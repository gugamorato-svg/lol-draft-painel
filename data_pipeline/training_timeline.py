"""Shared historical replay: labels become available after a map ends.

Oracle's date + gamelength is an estimate of the earliest possible availability,
not a recorded publication time. Missing/invalid durations are never replaced
with zero. A historical replay cannot use today's undated Gol.gg aggregates.
No fitting, file writes or network access happens in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import warnings

import numpy as np
import pandas as pd

from data_pipeline.temporal_context import TemporalContext

MAX_DURATION_SECONDS = 10800
SERIES_GAP_HOURS = 48
POLICY = "finished_before_prediction_v1_no_undated_prior"


@dataclass
class TrainingTimeline:
    games: pd.DataFrame
    features: pd.DataFrame
    context: object
    audit: dict


@dataclass
class TemporalSplit:
    train: np.ndarray
    test: np.ndarray
    split_at: int
    audit: dict


def _utc(value):
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError("Timestamp ausente no treino")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def build_training_timeline(df, use_prior=False, *, as_of=None, context_factory=TemporalContext):
    """Return aligned rows/features and a fully drained snapshot as of one cutoff.

    A completed validation map may update the state of later validation maps:
    that is prequential prediction, not training the regression on those labels.
    Equal start/end timestamps are conservative: the result is not yet used.
    """
    if use_prior:
        warnings.warn("--prior ignorado no treino: Gol.gg não tem snapshot/data de disponibilidade; "
                      "o replay e o snapshot usam somente Oracle temporal.", RuntimeWarning, stacklevel=2)
    as_of = _utc(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    games = df.copy()
    for name in ("gameid", "date", "gamelength", "blue_won"):
        if name not in games:
            raise ValueError(f"Coluna obrigatória ausente: {name}")
    if games["gameid"].isna().any() or games["gameid"].astype(str).duplicated().any():
        raise ValueError("gameid ausente/duplicado no dataset; não descartar silenciosamente")
    games["gameid"] = games["gameid"].astype(str)
    games["date"] = pd.to_datetime(games["date"], utc=True, errors="raise")
    if games["date"].isna().any():
        raise ValueError("Data ausente no dataset")
    if not games["blue_won"].isin([0, 1]).all():
        raise ValueError("Resultado ausente/inválido no dataset")
    duration = pd.to_numeric(games["gamelength"], errors="coerce")
    valid_duration = duration.notna() & np.isfinite(duration) & (duration > 0) & (duration <= MAX_DURATION_SECONDS)
    games["available_at"] = games["date"] + pd.to_timedelta(duration.where(valid_duration), unit="s")
    known = valid_duration & (games["available_at"] < as_of)
    audit = {
        "policy": POLICY, "as_of_utc": as_of.isoformat(), "source_rows": len(games),
        "excluded_invalid_duration": int((~valid_duration).sum()),
        "excluded_not_finished_as_of": int((valid_duration & ~known).sum()),
        "prior": "disabled_undated_Golgg", "prior_requested": bool(use_prior),
        "availability_limitation": "date+gamelength estimates earliest end; Oracle publication lag/pauses are not recorded",
    }
    games = games.loc[known].sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)
    if games.empty:
        raise ValueError("Nenhum resultado encerrado com duração válida antes do as_of")
    ctx, queue, rows = context_factory(), [], []
    for i, game in enumerate(games.to_dict("records")):
        while queue and queue[0][0] < game["date"]:
            _, _, finished = heapq.heappop(queue)
            ctx.update(finished)
        rows.append(ctx.build_features(game))
        heapq.heappush(queue, (game["available_at"], i, game))
    # Includes the last completed maps in the saved lookup, in completion order.
    while queue:
        _, _, finished = heapq.heappop(queue)
        ctx.update(finished)
    features = pd.DataFrame(rows)
    if features.empty or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features vazias/não finitas; não substituir silenciosamente por zero")
    audit.update({"retained_games": len(games), "first_start": games["date"].min().isoformat(),
                  "last_start": games["date"].max().isoformat(),
                  "latest_result_available_at": games["available_at"].max().isoformat(),
                  "dataset_sha256": hashlib.sha256(games.to_json(orient="records", date_format="iso").encode()).hexdigest()})
    return TrainingTimeline(games, features, ctx, audit)


def _text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def infer_series_groups(games):
    """Use an explicit series ID when present; otherwise conservative pair blocks.

    OE's aggregated table has no MatchId. Consecutive meetings of the same pair
    in the same competition within 48 h share a group, even across midnight or
    two Bo1 rematches. This can over-purge; it cannot prove identity of a series
    resuming after >48 h. Missing both team ID and name is an error.
    """
    last, groups = {}, []
    for game in games.to_dict("records"):
        explicit = next((_text(game.get(k)) for k in ("series_id", "match_id", "MatchId") if _text(game.get(k))), "")
        competition = (_text(game.get("league")), _text(game.get("year")), _text(game.get("split")))
        if explicit:
            groups.append(("explicit", *competition, explicit))
            continue
        # Names avoid splitting a series when OE omits an ID on only one map.
        teams = [_text(game.get(f"{side}_teamname")) or _text(game.get(f"{side}_teamid")) for side in ("blue", "red")]
        if not all(teams) or teams[0] == teams[1]:
            raise ValueError("Times ausentes/iguais; não é possível separar séries no holdout")
        key = (*competition, *sorted(teams))
        start = _utc(game["date"])
        prev = last.get(key)
        group_id = prev[1] if prev and start - prev[0] <= pd.Timedelta(hours=SERIES_GAP_HOURS) else game["gameid"]
        last[key] = (start, group_id)
        groups.append(("inferred", *key, group_id))
    return groups


def temporal_split(games, train_fraction=.8, test_end_fraction=1.0):
    """Purged chronological split with complete series on opposite sides.

    The boundary uses starts only. Labels that finish at/after the boundary,
    and all training members of any validation series, are excluded from fit.
    For a development window, series crossing its end are excluded entirely
    from development scoring, so their later maps stay in the final holdout.
    """
    if not 0 < train_fraction < test_end_fraction <= 1:
        raise ValueError("Frações de split inválidas")
    if len(games) < 3:
        raise ValueError("Poucos mapas para avaliação temporal")
    dates = pd.to_datetime(games["date"], utc=True, errors="raise")
    ends = pd.to_datetime(games["available_at"], utc=True, errors="raise")
    if not dates.is_monotonic_increasing:
        raise ValueError("O split exige jogos ordenados pela agenda temporal")
    split_at = max(1, min(len(games)-1, int(len(games)*train_fraction)))
    cutoff = dates.iloc[split_at]
    test_end = dates.iloc[int(len(games)*test_end_fraction)] if test_end_fraction < 1 else None
    groups = infer_series_groups(games)
    test = (dates >= cutoff).to_numpy(copy=True)
    if test_end is not None:
        after = (dates >= test_end).to_numpy()
        future_groups = {g for g, active in zip(groups, after) if active}
        test &= (dates < test_end).to_numpy()
        test &= np.array([g not in future_groups for g in groups])
    test_groups = {g for g, active in zip(groups, test) if active}
    past = (dates < cutoff).to_numpy()
    ended = (ends < cutoff).fillna(False).to_numpy()
    separate = np.array([g not in test_groups for g in groups])
    train = past & ended & separate
    if not train.any() or not test.any():
        raise ValueError("Treino/validação vazios após purga temporal e de séries")
    audit = {"policy": "chronological_purged_series_v1", "cutoff_utc": cutoff.isoformat(),
             "test_end_exclusive": test_end.isoformat() if test_end is not None else None,
             "train_games": int(train.sum()), "test_games": int(test.sum()),
             "purged_unfinished": int((past & ~ended).sum()),
             "purged_same_series": int((past & ended & ~separate).sum()),
             "validation_series": len(test_groups),
             "series_rule": "explicit ID or league/season + unordered team pair, gaps <=48h; inferred IDs can over-purge",
             "metric_scope": "historical descriptive validation; feature rules were previously explored, not a fresh confirmatory test"}
    return TemporalSplit(np.flatnonzero(train), np.flatnonzero(test), int(np.flatnonzero(test)[0]), audit)


def metadata_extra(timeline, split):
    """Provenance shared by weights and snapshot; no change to registrar API."""
    return {"training_timeline": timeline.audit, "evaluation_split": split.audit,
            "treino_ate": timeline.games["available_at"].max().strftime("%Y-%m-%d %H:%M:%S")}
