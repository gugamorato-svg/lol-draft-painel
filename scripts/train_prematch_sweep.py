"""Varre caps do pre_match_block para achar o 'meio termo'.

Ideia (descoberta do Dota): rating/h2h/forma sao 'pre-jogo'. O mercado ja
precifica isso. Consolidamos num pre_match_block e CAPAMOS para que nao
dominem a previsao, deixando espaco pro draft (lane/counter/synergy/comfort).

Mas no LoL nao queremos derrubar o pre demais -> varremos caps e escolhemos
um meio termo que:
  - mantem acuracia perto do baseline
  - aumenta o peso relativo das features de draft
  - nao reintroduz a armadilha 50-55%

Roda o loop temporal UMA vez; o cap e aplicado depois (barato).

Uso:
    python scripts/train_prematch_sweep.py --prior
    python scripts/train_prematch_sweep.py --prior --save-cap 1.8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.oracle_loader import load_years
from data_pipeline.training_timeline import build_training_timeline, temporal_split, metadata_extra

DATA_DIR = Path(__file__).parent.parent / "data"

# Features que entram DENTRO do pre_match_block (removidas como entrada direta)
PREMATCH_RAW_FEATURES = [
    "player_form_diff", "rating_diff", "recent_form_diff", "h2h_diff",
]
# Features de draft (queremos que ganhem espaco)
DRAFT_FEATURES = [
    "lane_matchup_diff", "last_pick_counter_diff", "synergy_diff",
    "counter_diff", "player_champion_diff", "jungle_mid_synergy_diff",
    "bot_2v2_synergy_diff", "flex_matchup_diff",
]
# Features REMOVIDAS deliberadamente do modelo.
#  - series_context_diff: WR historico por game number do BO; amostra pequena
#    por time, sinal duvidoso.
#  - last_pick_counter_diff: TRAINING/INFERENCE MISMATCH. Em treino, a lista
#    blue_picks_draft vem em ordem de DRAFT (pick1..pick5 do OE), entao [-1]
#    e o ultimo pick de verdade. Em live, a Riot devolve participantMetadata
#    em ordem de SLOT do player (top..sup), entao [-1] = support. A feature
#    aprendeu uma coisa e em producao aplica em outra. Alem disso, em 2026
#    blue tambem pode ter last pick, quebrando a heuristica side-based.
EXCLUDED_FEATURES = [
    "series_context_diff",       # WR por game-number do BO; amostra pequena/duvidosa
    "last_pick_counter_diff",    # mismatch ordem draft vs slot no live (ver acima)
    # near-zero / colineares — curados fora do modelo de producao (jun/2026).
    # (o loop temporal ainda os calcula; so nao entram no fit/CSV.)
    "flex_matchup_diff",         # colinear com counter/lane, coef ~0
    "ban_score_diff",            # coef ~0; tambem tinha mismatch de formula treino/live
    "jungle_mid_synergy_diff",   # coef ~0
    "bot_2v2_synergy_diff",      # coef ~0
    # constantes mortas — redundantes com o intercept
    "side_advantage",            # sempre 1.0
    "first_pick_diff",           # 0 no live (blue sempre picka 1o); ~constante no treino
    # SINAL INVERTIDO por colinearidade (auditoria 23/07): natural +, fit vira -
    # porque rating_diff/pre_match_block/objective ja carregam a forca do time.
    "team_patch_meta_diff",       # univariado +1.54, fit -0.29
    "side_specific_team_wr_diff", # univariado +0.89, fit -0.23
    # 23/09: KDA no campeao entrou SO no draft+comfort (validado la); no
    # completo o estudo de 22/07 deu incremental ~0.
    "comfort_kda_diff",
]


def run_temporal(years, only_tier1, use_prior):
    df = load_years(years, only_tier1=only_tier1, all_leagues=False if only_tier1 else None)
    print(f"[*] {len(df)} jogos. Loop temporal (1x)...")
    timeline = build_training_timeline(df, use_prior=use_prior)
    print(f"[OK] {timeline.context.stats()}")
    return timeline


def _fit_eval(X_df, y, feature_names, split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
    X = X_df[feature_names].values
    Xtr, Xte = X[split.train], X[split.test]
    ytr, yte = y[split.train], y[split.test]
    m = LogisticRegression(C=1.0, max_iter=1000)
    m.fit(Xtr, ytr)
    ptr = m.predict_proba(Xtr)[:, 1]
    pte_raw = m.predict_proba(Xte)[:, 1]
    pte = np.clip(pte_raw, .1, .9)  # mesma saída de FULL em trained_predictor
    return {
        "model": m,
        "acc_train": accuracy_score(ytr, (ptr >= 0.5)),
        "acc_test":  accuracy_score(yte, (pte >= 0.5)),
        "log_loss":  log_loss(yte, pte, labels=[0, 1]),
        "brier":     brier_score_loss(yte, pte),
        "log_loss_raw": log_loss(yte, pte_raw, labels=[0, 1]),
        "brier_raw": brier_score_loss(yte, pte_raw),
        "p_test":    pte,
        "y_test":    yte,
        "coefs":     dict(zip(feature_names, m.coef_[0])),
        "intercept": m.intercept_[0],
    }


def _calib_underdog(p_test, y_test):
    """Mede calibracao na faixa de underdog (P(blue) < 0.45) e armadilha 50-55."""
    out = {}
    # underdog: prob < 0.45
    for lo, hi, name in [(0.20, 0.45, "underdog"), (0.50, 0.55, "armadilha"),
                         (0.55, 0.65, "favorito_leve")]:
        # do ponto de vista do favorito
        p_fav = np.where(p_test >= 0.5, p_test, 1 - p_test)
        y_fav = np.where(p_test >= 0.5, y_test, 1 - y_test)
        if name == "underdog":
            mask = (p_test >= lo) & (p_test < hi)  # blue como underdog
            if mask.sum() == 0:
                out[name] = None
                continue
            real = y_test[mask].mean() * 100
            exp = p_test[mask].mean() * 100
        else:
            mask = (p_fav >= lo) & (p_fav < hi)
            if mask.sum() == 0:
                out[name] = None
                continue
            real = y_fav[mask].mean() * 100
            exp = p_fav[mask].mean() * 100
        out[name] = (int(mask.sum()), real, exp, real - exp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--tier1", action="store_true")
    ap.add_argument("--prior", action="store_true")
    ap.add_argument("--save-cap", type=float, default=None,
                    help="Se setado, salva model_coefficients.csv com esse cap")
    args = ap.parse_args()

    if args.save_cap is not None and (not np.isfinite(args.save_cap) or args.save_cap <= 0):
        ap.error("--save-cap exige um cap positivo e finito, escolhido antes desta execução")
    timeline = run_temporal(args.years, args.tier1, args.prior)
    df, X_df = timeline.games, timeline.features
    y = df["blue_won"].to_numpy(dtype=int)
    # A varredura nunca vê os 20% finais. Com --save-cap só UMA variante já
    # especificada é avaliada nesse holdout; o script não escolhe cap por ele.
    split = temporal_split(df) if args.save_cap is not None else temporal_split(df, .6, .8)
    print(f"[*] Validação: {split.audit}")

    all_features = [
        c for c in X_df.columns
        if c != "pre_match_raw" and c not in EXCLUDED_FEATURES
    ]

    # Feature set com bloco: tira os 4 crus, adiciona pre_match_block
    block_features = [f for f in all_features if f not in PREMATCH_RAW_FEATURES]
    block_features.append("pre_match_block")

    if args.save_cap is None:
        base = _fit_eval(X_df, y, all_features, split)
        print(f"BASELINE desenvolvimento: LL={base['log_loss']:.4f}, Brier={base['brier']:.4f}")
        print("SWEEP DESCRITIVO: treino inicial60%, desenvolvimento60–80%; holdout final não consultado")
        print(f"{'cap':>6} {'acc_dev':>8} {'brier':>7} {'logloss':>8} {'Sumdrft':>7}")
        for cap in [1.0, 1.3, 1.6, 2.0, 2.5, 99.0]:
            Xc = X_df.copy()
            Xc["pre_match_block"] = X_df["pre_match_raw"].clip(-cap, cap)
            r = _fit_eval(Xc, y, block_features, split)
            draft_sum = sum(abs(r["coefs"].get(f, 0)) for f in DRAFT_FEATURES)
            print(f"{cap:>6.1f} {r['acc_test']:>8.4f} {r['brier']:>7.4f} {r['log_loss']:>8.4f} {draft_sum:>7.2f}")
        return

    # Salva se pedido
    if args.save_cap is not None:
        cap = args.save_cap
        Xc = X_df.copy()
        Xc["pre_match_block"] = X_df["pre_match_raw"].clip(-cap, cap)
        r = _fit_eval(Xc, y, block_features, split)
        # 13/10 (auditoria): `r` e o ajuste nos 80% (so p/ avaliar). Producao = mesmo modelo
        # re-ajustado com TODO o passado.
        from models.model_meta import ajuste_final, registrar
        final = ajuste_final(Xc[block_features].values, y)
        coefs = dict(zip(block_features, final.coef_[0]))
        coef_path = DATA_DIR / "model_coefficients.csv"
        with open(coef_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature", "coefficient"])
            w.writerow(["__intercept__", final.intercept_[0]])
            for name in block_features:
                w.writerow([name, coefs[name]])
            # tambem grava o cap usado e os pesos internos (metadados)
            w.writerow(["__prematch_cap__", cap])
        registrar("model_coefficients.csv", df["date"], split.split_at,
                  {"log_loss": r["log_loss"], "acc": r["acc_test"], "brier": r["brier"],
                   "log_loss_raw": r["log_loss_raw"], "brier_raw": r["brier_raw"]},
                  extra={"cap": cap, "evaluation_probability_clip": [.1, .9], **metadata_extra(timeline, split)})
        print(f"\n[OK] Salvo model_coefficients.csv com pre_match_block cap={cap} (pesos com todo o passado)")
        print(f"     avaliação temporal purgada: acc_test={r['acc_test']:.4f}  brier={r['brier']:.4f}")
        print(f"\n  Coeficientes finais (nenhum cap escolhido pelo holdout):")
        for fdraft in DRAFT_FEATURES:
            n = coefs.get(fdraft, 0)
            print(f"    {fdraft:<25} {n:+.3f}")
        print(f"    {'pre_match_block':<25}        -> {coefs['pre_match_block']:+.3f}")


if __name__ == "__main__":
    main()
