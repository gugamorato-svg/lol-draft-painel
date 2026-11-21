"""Treina modelo de WR usando Oracle's Elixir + contexto temporal.

Fluxo:
  1. Carrega CSVs de N anos, ordena por data
  2. Loop temporal: para cada jogo,
       features = ctx.build_features(game)   # so estado passado
       ctx.update(game)                       # depois atualiza
  3. Split 80/20 temporal (NAO random) -- 80% mais antigos = treino
  4. LogisticRegression com regularizacao L2 leve
  5. Reajusta em todo passado encerrado; salva SOMENTE em diretório experimental
  6. Imprime calibration buckets (espelha analyze_calibration.py do Dota)

Uso:
    python scripts/train_model.py             # 2024 + 2025
    python scripts/train_model.py --years 2025
    python scripts/train_model.py --tier1     # so ligas tier 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.oracle_loader import load_years
from data_pipeline.training_timeline import build_training_timeline, temporal_split, metadata_extra

DATA_DIR = Path(__file__).parent.parent / "data"


def extract_features_temporal(df: pd.DataFrame, use_prior: bool = False):
    """Compatibilidade: a agenda compartilhada aguarda o encerramento do mapa."""
    timeline = build_training_timeline(df, use_prior=use_prior)
    metadata = timeline.games[["gameid", "date", "league", "patch"]].to_dict("records")
    return (timeline.features.to_dict("records"), timeline.games["blue_won"].astype(int).tolist(),
            metadata, timeline.context)


def train(years, only_tier1: bool, output_dir: Path, use_prior: bool = False):
    output_dir = Path(output_dir)
    if output_dir.resolve() == DATA_DIR.resolve():
        raise ValueError("train_model é experimental; não pode sobrescrever os pesos de produção em data/. "
                         "Use train_prematch_sweep para o modelo implantado.")
    df = load_years(years, only_tier1=only_tier1, all_leagues=False if only_tier1 else None)
    print(f"[*] {len(df)} jogos carregados (anos={years}, tier1={only_tier1}, prior={use_prior})")

    timeline = build_training_timeline(df, use_prior=use_prior)
    df, X_df = timeline.games, timeline.features
    meta = df[["gameid", "date", "league", "patch"]].to_dict("records")
    feature_names = list(X_df.columns)
    X = X_df.values
    y = df["blue_won"].to_numpy(dtype=int)

    # Split temporal 80/20
    split = temporal_split(df)
    split_at = split.split_at
    X_train, X_test = X[split.train], X[split.test]
    y_train, y_test = y[split.train], y[split.test]
    meta_test = [meta[i] for i in split.test]

    print(f"\n[*] Split temporal:")
    print(f"    {split.audit}")

    # Validacao das features (lição do bug do Dota)
    print(f"\n[*] Features != 0 em {len(X_train)} jogos de treino:")
    for j, name in enumerate(feature_names):
        nonzero = int(np.sum(np.abs(X_train[:, j]) > 1e-9))
        pct = 100 * nonzero / len(X_train)
        flag = " !!!ARMADILHA" if pct < 50 else ""
        print(f"    {name:<25} {nonzero:>6} ({pct:>5.1f}%){flag}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test  = model.predict_proba(X_test)[:, 1]

    acc_train = accuracy_score(y_train, (p_train >= 0.5).astype(int))
    acc_test  = accuracy_score(y_test,  (p_test  >= 0.5).astype(int))
    ll_test   = log_loss(y_test, p_test, labels=[0, 1])
    brier     = brier_score_loss(y_test, p_test)

    print(f"\n[*] Metricas:")
    print(f"    accuracy treino: {acc_train:.3f}")
    print(f"    accuracy teste:  {acc_test:.3f}")
    print(f"    log loss teste:  {ll_test:.3f}")
    print(f"    Brier teste:     {brier:.4f}")

    # Coeficientes
    print(f"\n[*] Coeficientes aprendidos (intercept={model.intercept_[0]:+.3f}):")
    coefs = sorted(
        zip(feature_names, model.coef_[0]),
        key=lambda x: -abs(x[1]),
    )
    for name, coef in coefs:
        bar = "*" * min(40, int(abs(coef) * 30))
        sign = "+" if coef >= 0 else "-"
        print(f"    {name:<25} {sign}{abs(coef):.3f}  {bar}")

    # O fit da avaliação nunca é confundido com o fit final. A saída antiga
    # tinha features incompatíveis com produção e sobrescrevia seu mesmo CSV.
    from models.model_meta import ajuste_final, registrar
    final = ajuste_final(X, y)
    output_dir.mkdir(parents=True, exist_ok=True)
    coef_path = output_dir / "model_coefficients.csv"
    with open(coef_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "coefficient"])
        w.writerow(["__intercept__", final.intercept_[0]])
        for name, coef in zip(feature_names, final.coef_[0]):
            w.writerow([name, coef])
    registrar("model_coefficients.csv", df["date"], split_at,
              {"log_loss": ll_test, "acc": acc_test, "brier": brier},
              path=output_dir / "model_meta.json", data_dir=output_dir,
              extra={"experimental_only": True, **metadata_extra(timeline, split)})
    print(f"\n[OK] Coeficientes salvos em {coef_path}")

    # Calibration buckets (armadilha 50-55%)
    print(f"\n[*] Calibration buckets (teste, P(blue_wins)):")
    print(f"    {'bucket':<18} {'jogos':>6} {'real_wr':>10} {'esperado':>10} {'delta':>9}")
    buckets = [(0.20, 0.45), (0.45, 0.50), (0.50, 0.55),
               (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80)]
    # Reflete: como modelo cospe P(blue_wins), olhamos do ponto de vista do favorito
    p_fav = np.where(p_test >= 0.5, p_test, 1 - p_test)
    y_fav_wins = np.where(p_test >= 0.5, y_test, 1 - y_test)
    for lo, hi in buckets:
        mask = (p_fav >= lo) & (p_fav < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"    [{lo*100:>3.0f}-{hi*100:<3.0f}%]      {n:>6}  ---")
            continue
        real = y_fav_wins[mask].mean() * 100
        exp  = p_fav[mask].mean() * 100
        delta = real - exp
        flag = " <- ARMADILHA" if (lo == 0.50 and delta < -3) else ""
        print(f"    [{lo*100:>3.0f}-{hi*100:<3.0f}%]      {n:>6}  {real:>9.1f}% {exp:>9.1f}% {delta:>+8.1f}pp{flag}")

    # Salva predicoes do teste
    test_pred_path = output_dir / "test_predictions.csv"
    with open(test_pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gameid", "date", "league", "patch",
                    "p_blue", "blue_won"])
        for i, m in enumerate(meta_test):
            w.writerow([m["gameid"], m["date"], m["league"], m["patch"],
                        f"{p_test[i]:.4f}", int(y_test[i])])
    print(f"[OK] Predicoes do teste salvas em {test_pred_path}")

    return {
        "acc_train": acc_train, "acc_test": acc_test,
        "log_loss_test": ll_test, "brier_test": brier,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--tier1", action="store_true",
                    help="So ligas tier 1 (LPL, LCK, LEC, LCS, LTA, Worlds, MSI)")
    ap.add_argument("--prior", action="store_true",
                    help="Compatibilidade: prior sem data é ignorado com aviso")
    ap.add_argument("--output-dir", type=Path, default=DATA_DIR / "archive" / "train_model",
                    help="Diretório experimental (data/ diretamente é recusado)")
    args = ap.parse_args()

    train(args.years, only_tier1=args.tier1, output_dir=args.output_dir,
          use_prior=args.prior)


if __name__ == "__main__":
    main()
