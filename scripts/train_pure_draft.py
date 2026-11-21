"""Treina 3 niveis de modelo "draft only" com graus de pureza crescentes.

NIVEL 1 (atual cap=0): exclui so o pre_match_block. Mantem features de time
        que codificam forca (early_game, objective_control etc) -> VAZA.

NIVEL 2 (limpo): exclui o pre_match_block E todas as features de TIME
        (early_game, objective, side_specific, team_patch_meta, league_champ).
        Mantem features de player+champ (comfort).

NIVEL 3 (ultra puro): exclui tudo de time E tudo de player. So features
        que dependem dos CHAMPIONS picados (lane, counter, synergy, patch_wr,
        champion_meta). 100% draft-only.

Salva 3 CSVs separados:
  data/archive/model_coefficients_zero_leaky.csv (nivel 1 - debug/leaky)
  data/model_coefficients_pure_draft.csv   (nivel 2 - limpo)
  data/model_coefficients_champs_only.csv  (nivel 3 - ultra puro)

29/10/2026 (dono): a FORCA DA LIGA (league_strength_diff) e informacao do TIME (liga de casa),
nao do draft, e era o maior pedaco do "draft puro" em jogo entre ligas (KC x CTBC no Worlds:
62% -> 52% sem ela). Os niveis 2 e 3 OFICIAIS agora ficam SEM ela; as versoes COM ela continuam
treinadas em arquivos proprios (estrategias "draft + forca da liga" e o insumo da candidata do
teste prospectivo):
  data/model_coefficients_zero_liga.csv          (nivel 2 + forca da liga)
  data/model_coefficients_champs_only_liga.csv   (nivel 3 + forca da liga)
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.oracle_loader import load_years
from data_pipeline.training_timeline import build_training_timeline, temporal_split, metadata_extra

DATA_DIR = Path(__file__).parent.parent / "data"
ARCHIVE_DIR = DATA_DIR / "archive"

# Features sempre excluidas (mismatch, redundancia ou inputs zerados em live)
ALWAYS_EXCLUDED = [
    "series_context_diff", "last_pick_counter_diff", "first_pick_diff",
    "side_advantage",
    "bot_2v2_synergy_diff", "flex_matchup_diff",
    "jungle_mid_synergy_diff", "ban_score_diff",
]
PREMATCH_RAW = ["player_form_diff", "rating_diff", "recent_form_diff", "h2h_diff"]

# Features que VAZAM identidade do time (forca pela ID, nao pelo draft)
TEAM_LEAKAGE = [
    "team_patch_meta_diff",       # WR do TIME nesses champs
    "side_specific_team_wr_diff", # WR do TIME nesse lado
    "early_game_team_diff",       # gold/xp/cs medio do TIME @15
    "objective_control_diff",     # drag/herald/baron do TIME
    "league_champion_wr_diff",    # codifica liga (Gen.G = LCK)
]

# Forca da liga de casa dos times (29/10): informacao de TIME, fora dos modelos "draft" oficiais
LEAGUE_STRENGTH = ["league_strength_diff"]

# Features dependentes do PLAYER (capturam comfort/skill individual)
PLAYER_DEPENDENT = [
    "player_champion_diff",
    "player_champion_rate_diff",
    "sample_size_confidence_diff",
    "comfort_kda_diff",   # 23/09: KDA no campeao (so no nivel 2 = draft + comfort)
]


def fit(X_df, y, feats, split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
    X = X_df[feats].values
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X[split.train], y[split.train])
    pte_raw = m.predict_proba(X[split.test])[:, 1]
    pte = np.clip(pte_raw, .1, .9)  # mesma saída COMFORT/champs em produção
    yte = y[split.test]
    # 13/10 (auditoria): `m` e o ajuste nos 80% (so p/ avaliar). Os pesos SALVOS sao do mesmo
    # modelo re-ajustado com TODO o passado (models/model_meta.py).
    from models.model_meta import ajuste_final
    final = ajuste_final(X, y)
    return {
        "acc": accuracy_score(yte, pte >= 0.5),
        "brier": brier_score_loss(yte, pte),
        "ll": log_loss(yte, pte, labels=[0, 1]),
        "ll_raw": log_loss(yte, pte_raw, labels=[0, 1]),
        "brier_raw": brier_score_loss(yte, pte_raw),
        "coef": dict(zip(feats, final.coef_[0])),
        "b0": final.intercept_[0],
        "coef_avaliacao": dict(zip(feats, m.coef_[0])),
        "n": len(feats),
        "p": pte, "y": yte,
    }


def buckets(p, y):
    p_fav = np.where(p >= 0.5, p, 1 - p)
    y_fav = np.where(p >= 0.5, y, 1 - y)
    rows = []
    for lo, hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.80),(0.80,1.01)]:
        m = (p_fav >= lo) & (p_fav < hi)
        n = int(m.sum())
        if n == 0:
            rows.append((lo, hi, 0, 0, 0, 0)); continue
        real = y_fav[m].mean()*100
        exp = p_fav[m].mean()*100
        rows.append((lo, hi, n, real, exp, real-exp))
    return rows


def save_coefs(result, feats, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "coefficient"])
        w.writerow(["__intercept__", result["b0"]])
        for name in feats:
            w.writerow([name, result["coef"][name]])
        w.writerow(["__prematch_cap__", 0.0])


def main():
    from models.model_meta import registrar
    df = load_years([2024, 2025, 2026], only_tier1=False)
    print(f"[*] {len(df)} jogos. Loop temporal...")
    timeline = build_training_timeline(df)
    df, X_df = timeline.games, timeline.features
    y = df["blue_won"].to_numpy(dtype=int)
    X_df["pre_match_block"] = 0.0  # cap=0
    split = temporal_split(df)

    base = lambda extra: [
        c for c in X_df.columns
        if c != "pre_match_raw"
        and c != "pre_match_block"
        and c not in PREMATCH_RAW
        and c not in ALWAYS_EXCLUDED
        and c not in extra
    ] + ["pre_match_block"]

    print("\n" + "="*82)
    print("3 NIVEIS DE 'DRAFT ONLY'")
    print("="*82)

    # NIVEL 1: atual (sem prematch_block, mas com features de time) -- VAZA
    feats_l1 = base([])
    r1 = fit(X_df, y, feats_l1, split)
    print(f"\nNIVEL 1 (com leakage)  n={r1['n']:>2}  acc={r1['acc']:.4f}  brier={r1['brier']:.4f}")
    ARCHIVE_DIR.mkdir(exist_ok=True)
    save_coefs(r1, feats_l1, ARCHIVE_DIR / "model_coefficients_zero_leaky.csv")
    print("  [debug/leaky] -> archive/model_coefficients_zero_leaky.csv")

    def salvar(r, feats, arquivo, rotulo):
        save_coefs(r, feats, DATA_DIR / arquivo)
        registrar(arquivo, df["date"], split.split_at,
                  {"log_loss": r["ll"], "acc": r["acc"], "brier": r["brier"],
                   "log_loss_raw": r["ll_raw"], "brier_raw": r["brier_raw"]},
                  extra={"evaluation_probability_clip": [.1, .9], **metadata_extra(timeline, split)})
        print(f"  {rotulo:34s} n={r['n']:>2}  acc={r['acc']:.4f}  brier={r['brier']:.4f}  ll={r['ll']:.4f} -> {arquivo}")

    # NIVEL 2 (draft + comfort do jogador). OFICIAL sem a forca da liga; _liga com ela.
    feats_l2 = base(TEAM_LEAKAGE + LEAGUE_STRENGTH)
    r2 = fit(X_df, y, feats_l2, split)
    salvar(r2, feats_l2, "model_coefficients_zero.csv", "NIVEL 2 draft+comfort")
    feats_l2l = base(TEAM_LEAKAGE)
    r2l = fit(X_df, y, feats_l2l, split)
    salvar(r2l, feats_l2l, "model_coefficients_zero_liga.csv", "NIVEL 2 draft+comfort+forca liga")

    # NIVEL 3 (so campeoes). OFICIAL sem a forca da liga; _liga com ela.
    feats_l3 = base(TEAM_LEAKAGE + PLAYER_DEPENDENT + LEAGUE_STRENGTH)
    r3 = fit(X_df, y, feats_l3, split)
    salvar(r3, feats_l3, "model_coefficients_champs_only.csv", "NIVEL 3 draft puro")
    feats_l3l = base(TEAM_LEAKAGE + PLAYER_DEPENDENT)
    r3l = fit(X_df, y, feats_l3l, split)
    salvar(r3l, feats_l3l, "model_coefficients_champs_only_liga.csv", "NIVEL 3 draft+forca liga")

    # Buckets dos 3
    print("\n" + "="*82)
    print("BUCKETS DE CALIBRACAO (ponto de vista do favorito)")
    print("="*82)
    b1, b2, b3 = buckets(r1['p'], r1['y']), buckets(r2['p'], r2['y']), buckets(r3['p'], r3['y'])
    print(f"{'bucket':<10} {'| NIVEL 1':<22} {'| NIVEL 2':<22} {'| NIVEL 3':<22}")
    for (lo,hi,n1,r_1,e1,d1), (_,_,n2,r_2,e2,d2), (_,_,n3,r_3,e3,d3) in zip(b1, b2, b3):
        tag = f"[{lo*100:.0f}-{hi*100:.0f}%]"
        s1 = f"n{n1:<4} {r_1:4.1f}/{e1:4.1f}/{d1:+5.1f}" if n1 else "--"
        s2 = f"n{n2:<4} {r_2:4.1f}/{e2:4.1f}/{d2:+5.1f}" if n2 else "--"
        s3 = f"n{n3:<4} {r_3:4.1f}/{e3:4.1f}/{d3:+5.1f}" if n3 else "--"
        print(f"{tag:<10} | {s1:<20} | {s2:<20} | {s3:<20}")


if __name__ == "__main__":
    main()
