"""Treina modelo 'team-only' usando apenas features que o app PyQt6 conhece.

Como o app nao recebe picks/bans (so nomes de times e jogadores), o modelo
de 10 features nao pode ser aplicado direto. Treinamos um modelo separado
com 4 features que dependem so de time/player:
    - rating_diff       (Elo)
    - recent_form_diff  (WR janela 10 do time)
    - h2h_diff          (WR direto smoothed)
    - player_form_diff  (WR janela 10 dos jogadores)

Salva tambem o snapshot do TemporalContext apos processar tudo, para o
app carregar e usar como lookup table.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.oracle_loader import load_years
from data_pipeline.training_timeline import build_training_timeline, temporal_split, metadata_extra

DATA_DIR = Path(__file__).parent.parent / "data"
TEAM_FEATURES = ["rating_diff", "recent_form_diff", "h2h_diff", "player_form_diff"]


def run(years, only_tier1: bool, use_prior: bool):
    df = load_years(years, only_tier1=only_tier1, all_leagues=False if only_tier1 else None)
    print(f"[*] {len(df)} jogos carregados")

    timeline = build_training_timeline(df, use_prior=use_prior)
    df, ctx = timeline.games, timeline.context
    team_name_to_id = {}
    player_name_to_id = {}
    player_name_to_ids = defaultdict(set)
    roles = ("top", "jng", "mid", "bot", "sup")

    print(f"[*] Mapeando nomes do snapshot temporal...")
    for i, row in enumerate(df.itertuples(index=False), 1):
        g = row._asdict()
        # Mapping de nomes -> ids (mais recente sobrescreve antigos)
        team_name_to_id[g["blue_teamname"].strip().lower()] = g["blue_teamid"]
        team_name_to_id[g["red_teamname"].strip().lower()]  = g["red_teamid"]

        for side in ("blue", "red"):
            for r in roles:
                pname = g.get(f"{side}_playername_{r}")
                pid   = g.get(f"{side}_player_{r}")
                # jogador sem playerid no Oracle vem NaN: nao pode virar o id "nan"
                # (todos os sem-id caiam no mesmo "jogador")
                if (pname and pid and not (isinstance(pname, float) and pname != pname)
                        and not (isinstance(pid, float) and pid != pid)):
                    name = str(pname).strip().lower()
                    player_name_to_id[name] = str(pid)
                    player_name_to_ids[name].add(str(pid))

        if i % 2000 == 0:
            print(f"    {i}/{len(df)}")

    print(f"[OK] {ctx.stats()}")

    X_df = timeline.features[TEAM_FEATURES]
    X = X_df.values
    y = df["blue_won"].to_numpy(dtype=int)

    split = temporal_split(df)
    split_at = split.split_at
    X_train, X_test = X[split.train], X[split.test]
    y_train, y_test = y[split.train], y[split.test]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    p_train = model.predict_proba(X_train)[:, 1]

    print(f"\n[*] Modelo team-only ({len(TEAM_FEATURES)} features):")
    print(f"    acc treino: {accuracy_score(y_train, (p_train>=0.5).astype(int)):.3f}")
    print(f"    acc teste:  {accuracy_score(y_test, (p_test>=0.5).astype(int)):.3f}")
    print(f"    log loss:   {log_loss(y_test, p_test, labels=[0, 1]):.3f}")
    print(f"    Brier:      {brier_score_loss(y_test, p_test):.4f}")

    print(f"\n[*] Coeficientes (intercept={model.intercept_[0]:+.3f}):")
    for name, coef in zip(TEAM_FEATURES, model.coef_[0]):
        bar = "*" * min(40, int(abs(coef) * 30))
        sign = "+" if coef >= 0 else "-"
        print(f"    {name:<22} {sign}{abs(coef):.3f}  {bar}")

    # Salva coeficientes team-only. 13/10 (auditoria): os de cima sao do ajuste nos 80% (so p/
    # avaliar); o que vai p/ producao e o mesmo modelo re-ajustado com TODO o passado.
    from models.model_meta import ajuste_final, registrar
    final = ajuste_final(X, y)
    coef_path = DATA_DIR / "model_team_only.csv"
    with open(coef_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "coefficient"])
        w.writerow(["__intercept__", final.intercept_[0]])
        for name, coef in zip(TEAM_FEATURES, final.coef_[0]):
            w.writerow([name, coef])
    registrar("model_team_only.csv", df["date"], split_at,
              {"log_loss": log_loss(y_test, p_test, labels=[0, 1]), "acc": accuracy_score(y_test, (p_test >= 0.5).astype(int)),
               "brier": brier_score_loss(y_test, p_test)}, extra=metadata_extra(timeline, split))
    print(f"[OK] {coef_path} (pesos com todo o passado; metadados em data/model_meta.json)")

    # Calibration buckets
    print(f"\n[*] Calibration:")
    p_fav = np.where(p_test >= 0.5, p_test, 1 - p_test)
    y_fav = np.where(p_test >= 0.5, y_test, 1 - y_test)
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80)]
    for lo, hi in buckets:
        mask = (p_fav >= lo) & (p_fav < hi)
        n = mask.sum()
        if n == 0: continue
        real = y_fav[mask].mean() * 100
        exp = p_fav[mask].mean() * 100
        flag = " <- armadilha" if (lo == 0.50 and (real - exp) < -3) else ""
        print(f"    [{lo*100:.0f}-{hi*100:.0f}%]  n={n:>4}  real={real:>5.1f}%  esperado={exp:>5.1f}%  delta={real-exp:+5.1f}pp{flag}")

    # Salva snapshot do contexto para o app usar como lookup
    snapshot = {
        "team_rating":     dict(ctx.team_rating),
        "team_form":       {k: list(v) for k, v in ctx.team_form.items()},
        "player_form":     {k: list(v) for k, v in ctx.player_form.items()},
        "h2h":             dict(ctx.h2h),
        "champion_matchups": dict(ctx.champion_matchups),
        "synergies":       dict(ctx.synergies),
        "golgg_champion_matchups": dict(ctx.prior_champion_matchups),
        "golgg_synergies": dict(ctx.prior_synergies),
        "counter_synergy_mode": "oracle_first_golgg_fallback",
        "champion_pickrate": dict(ctx.champion_pickrate),
        "champion_recent_meta": {k: dict(v) for k, v in ctx.champion_recent_meta.items()},
        "player_champion": dict(ctx.player_champion),
        "team_early_profile": {k: dict(v) for k, v in ctx.team_early_profile.items()},
        "team_objective_profile": {k: dict(v) for k, v in ctx.team_objective_profile.items()},
        "team_side": dict(ctx.team_side),
        "team_series": dict(ctx.team_series),
        "team_champion": dict(ctx.team_champion),
        "patch_champion": dict(ctx.patch_champion),
        "league_champion": dict(ctx.league_champion),
        "league_rating": dict(ctx.league_rating),
        "cross_league_games": list(ctx.cross_league_games),
        "team_league_games": {k: dict(v) for k, v in ctx.team_league_games.items()},
        "player_total_games": dict(ctx.player_total_games),
        "player_champion_kda": {k: dict(v) for k, v in ctx.player_champion_kda.items()},
        "champion_role_kda": {k: dict(v) for k, v in ctx.champion_role_kda.items()},
        "team_name_to_id": team_name_to_id,
        "player_name_to_id": player_name_to_id,
        "player_name_to_ids": {name: sorted(ids) for name, ids in player_name_to_ids.items()},
        "context_settings": {"league_in_rating": bool(ctx.league_in_rating)},
        "training_timeline": timeline.audit,
        "snapshot_as_of": timeline.audit["as_of_utc"],
        "latest_result_available_at": timeline.audit["latest_result_available_at"],
        "n_games_seen":    ctx.n_games_seen,
        "years":           years,
    }
    snapshot_path = DATA_DIR / "temporal_state.pkl"
    with open(snapshot_path, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"[OK] Snapshot salvo em {snapshot_path} ({snapshot_path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--tier1", action="store_true")
    ap.add_argument("--prior", action="store_true",
                    help="Usa gol.gg como prior dos matchups")
    args = ap.parse_args()
    run(args.years, only_tier1=args.tier1, use_prior=args.prior)


if __name__ == "__main__":
    main()


