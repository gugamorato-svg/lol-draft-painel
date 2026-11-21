"""Retreina o modelo de WR AO VIVO (data/live_wr_model.csv).

Logistica unica: estado do jogo (gold/xp/kills/tempo) + logit(P_comp pre-draft).
- P_comp e previsto fora da amostra, com ajuste mensal usando apenas jogos
  encerrados antes do mes. Nunca le os pesos de producao ajustados no futuro.
- O contexto so incorpora um jogo depois do encerramento. Sem prior sem data.
- Snapshots do Oracle @10/15/20/25, perspectiva do blue.
- Valida no holdout (ano mais recente) ANTES de salvar: se o log loss vier ruim
  (> LL_MAX) o modelo atual NAO e sobrescrito. Backup do anterior sempre.
- Features = as que live_wr.py consome: gd, xpd, kd, tn, gd_t, cl.

Uso: python scripts/train_live_wr.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import glob
import heapq
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from data_pipeline.oracle_loader import load_years, list_available_years
from data_pipeline.temporal_context import TemporalContext
from scripts.train_prematch_sweep import PREMATCH_RAW_FEATURES, EXCLUDED_FEATURES

DATA = ROOT / "data"
OUT = DATA / "live_wr_model.csv"
FEATS = ["gd", "xpd", "kd", "tn", "gd_t", "cl"]
LL_MAX = 0.52   # modelo bom hoje ~0.45; acima disso algo quebrou -> nao salva
MIN_PAST_GAMES = 500
PREMATCH_CAP = 1.6


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def temporal_features(df, context_factory=TemporalContext):
    """Features disponiveis no inicio do mapa; resultados so apos gamelength.

    Duracao ausente/invalida: nao atualiza contexto nem permite usar esse label
    no ajuste upstream. Nao presume que o resultado estava disponivel no inicio.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise")
    df = df.sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)
    if df["gameid"].duplicated().any():
        raise ValueError("gameid duplicado no dataset")
    ctx, queue, rows, available = context_factory(), [], [], []
    for i, game in enumerate(df.itertuples(index=False)):
        g = game._asdict()
        start = g["date"]
        while queue and queue[0][0] <= start:
            _, _, finished = heapq.heappop(queue)
            ctx.update(finished)
        rows.append(ctx.build_features(g))
        duration = pd.to_numeric(g.get("gamelength"), errors="coerce")
        end = start + pd.Timedelta(seconds=float(duration)) if pd.notna(duration) and 0 < duration <= 10800 else pd.NaT
        available.append(end)
        if pd.notna(end):
            heapq.heappush(queue, (end, i, g))
    df["available_at"] = pd.to_datetime(available, utc=True)
    return df, pd.DataFrame(rows).fillna(0.0)


def past_only_probabilities(df, features, min_games=MIN_PAST_GAMES, model_factory=None):
    """P_comp mensal fora da amostra, sem utilizar labels do mes previsto.

    O primeiro periodo e aquecimento (NaN), nao uma previsao in-sample.
    Retorna auditoria dos cortes para cada ajuste.
    """
    from sklearn.linear_model import LogisticRegression
    model_factory = model_factory or (lambda: LogisticRegression(C=1.0, max_iter=1000))
    X = features.copy()
    X["pre_match_block"] = X["pre_match_raw"].clip(-PREMATCH_CAP, PREMATCH_CAP)
    excluded = set(EXCLUDED_FEATURES + PREMATCH_RAW_FEATURES + ["pre_match_raw"])
    cols = [c for c in X if c not in excluded]
    dates = pd.to_datetime(df["date"], utc=True)
    available = pd.to_datetime(df["available_at"], utc=True)
    months = dates.dt.strftime("%Y-%m")
    probs = pd.Series(np.nan, index=df.index, dtype=float)
    folds = []
    for month in sorted(months.unique()):
        cutoff = pd.Timestamp(month + "-01", tz="UTC")
        train = (dates < cutoff) & (available < cutoff)
        test = months == month
        if train.sum() < min_games or df.loc[train, "blue_won"].nunique() < 2:
            continue
        model = model_factory().fit(X.loc[train, cols], df.loc[train, "blue_won"])
        probs.loc[test] = np.clip(model.predict_proba(X.loc[test, cols])[:, 1], 0.1, 0.9)
        folds.append({"month": month, "train_games": int(train.sum()), "predicted_games": int(test.sum()),
                      "latest_training_result": available[train].max().isoformat(),
                      "prediction_cutoff": cutoff.isoformat()})
    return probs, folds


def split_live_validation(dataset, test_year):
    """Segundo estagio tambem exige resultado disponivel antes da virada."""
    cutoff = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
    starts = pd.to_datetime(dataset["date"], utc=True)
    ends = pd.to_datetime(dataset["available_at"], utc=True)
    return (dataset[(starts < cutoff) & (ends < cutoff)],
            dataset[starts.dt.year == test_year])


def build_dataset(years):
    print(f"[*] P_comp mensal fora da amostra ({years})...")
    df, features = temporal_features(load_years(years, only_tier1=False))
    probs, folds = past_only_probabilities(df, features)
    P = pd.DataFrame({"gameid": df["gameid"].astype(str), "win": df["blue_won"].astype(int),
                      "p_comp": probs, "year": df["date"].dt.year, "date": df["date"],
                      "available_at": df["available_at"]}).dropna()
    print(f"[*] {len(P)} probabilidades fora da amostra; {len(df)-len(P)} jogos de aquecimento excluidos")

    cols = ["gameid", "position", "side"]
    for t in (10, 15, 20, 25):
        cols += [f"golddiffat{t}", f"xpdiffat{t}", f"killsat{t}", f"deathsat{t}"]
    files = sorted(glob.glob(str(DATA / "oracle_elixir" / "*_lol.csv")))
    raw = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files],
                    ignore_index=True)
    blue = raw[(raw["position"] == "team") & (raw["side"] == "Blue")].copy()
    blue["gameid"] = blue["gameid"].astype(str)
    blue = blue.drop_duplicates("gameid", keep="last")
    m = blue.merge(P, on="gameid", how="inner", validate="one_to_one")

    rows = []
    for t in (10, 15, 20, 25):
        s = m.dropna(subset=[f"golddiffat{t}"])
        rows.append(pd.DataFrame({
            "gd": pd.to_numeric(s[f"golddiffat{t}"], errors="coerce") / 1000,
            "xpd": pd.to_numeric(s[f"xpdiffat{t}"], errors="coerce") / 1000,
            "kd": pd.to_numeric(s[f"killsat{t}"], errors="coerce")
                  - pd.to_numeric(s[f"deathsat{t}"], errors="coerce"),
            "tn": (t - 17.5) / 7.5,
            "cl": _logit(s["p_comp"].values),
            "win": s["win"].values,
            "year": s["year"].values,
            "gameid": s["gameid"].values,
            "date": s["date"].values,
            "available_at": s["available_at"].values,
        }))
    L = pd.concat(rows, ignore_index=True).dropna()
    L["gd_t"] = L["gd"] * L["tn"]
    L.attrs["upstream_folds"] = folds
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="valida mas nao salva")
    ap.add_argument("--report", type=Path, help="salva diagnostico da validacao, inclusive em dry-run")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    years = list_available_years()
    L = build_dataset(years)
    test_year = max(years)
    tr, te = split_live_validation(L, test_year)
    if tr.empty or te.empty or tr["win"].nunique() < 2:
        raise ValueError("treino/validacao temporal insuficientes")
    if set(tr["gameid"]) & set(te["gameid"]):
        raise ValueError("mapa presente em treino e validacao")
    print(f"[*] {len(L)} snapshots | treino {len(tr)} | holdout {test_year}: {len(te)}")

    mv = LogisticRegression(C=1.0, max_iter=1000).fit(tr[FEATS], tr["win"])
    pte = mv.predict_proba(te[FEATS])[:, 1]
    ll = log_loss(te["win"], pte)
    report = {"validated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
              "evaluation": "ultimo ano; pesos live fixos, P_comp mensal com passado encerrado",
              "upstream_prior": "sem Golgg: nao ha snapshot historico datado para reconstruir disponibilidade",
              "feed_limitation": "Oracle usa XP e minuto exatos; feed pode estimar XP/tempo. Esta validacao nao comprova calibracao no feed real.",
              "train_snapshots": len(tr), "test_snapshots": len(te), "test_year": test_year,
              "test_games": int(te["gameid"].nunique()), "log_loss": float(ll),
              "ll_max": LL_MAX, "passed": bool(np.isfinite(ll) and ll <= LL_MAX),
              "upstream_folds": L.attrs.get("upstream_folds", [])}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[*] holdout {test_year}: log loss {ll:.4f}")
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        mm = (pte >= lo) & (pte < hi)
        if mm.sum():
            print(f"    prev {lo*100:.0f}-{hi*100:.0f}%: real {100*te['win'].values[mm].mean():.1f}% (n{mm.sum()})")

    if not np.isfinite(ll) or ll > LL_MAX:
        print(f"[ERRO] log loss {ll:.4f} > {LL_MAX} — algo quebrou. Modelo atual MANTIDO.")
        return 2

    final = LogisticRegression(C=1.0, max_iter=1000).fit(L[FEATS], L["win"])
    coefs = dict(zip(FEATS, final.coef_[0]))
    print(f"[*] coefs finais: intercept {final.intercept_[0]:+.3f} | "
          + " ".join(f"{k} {v:+.3f}" for k, v in coefs.items()))

    if args.dry_run:
        print("[dry-run] nao salvo.")
        return 0

    if OUT.exists():
        bk = DATA / "_backup_live_wr"
        bk.mkdir(exist_ok=True)
        shutil.copy2(OUT, bk / f"live_wr_model_{datetime.now():%Y%m%d_%H%M%S}.csv")
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "coefficient"])
        w.writerow(["__intercept__", final.intercept_[0]])
        for k in FEATS:
            w.writerow([k, coefs[k]])
    tmp.replace(OUT)
    (DATA / "live_wr_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] salvo {OUT.name} (backup do anterior em data/_backup_live_wr/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
