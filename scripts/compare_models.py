"""Compara modelo ANTIGO (4 pre-match soltas) vs NOVO (pre_match_block cap 1.6).

Roda o loop temporal 1x, treina os dois, imprime formula + buckets de calibracao.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.oracle_loader import load_years
from data_pipeline.temporal_context import TemporalContext

PREMATCH_RAW = ["player_form_diff", "rating_diff", "recent_form_diff", "h2h_diff"]
CAP = 1.6


def fit(X_df, y, feats, split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
    X = X_df[feats].values
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X[:split], y[:split])
    pte = m.predict_proba(X[split:])[:, 1]
    yte = y[split:]
    return {
        "acc": accuracy_score(yte, pte >= 0.5),
        "brier": brier_score_loss(yte, pte),
        "ll": log_loss(yte, pte),
        "p": pte, "y": yte,
        "coef": dict(zip(feats, m.coef_[0])),
        "b0": m.intercept_[0],
    }


def buckets(p, y):
    """Buckets do ponto de vista do FAVORITO."""
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


def main():
    df = load_years([2024, 2025, 2026], only_tier1=False)
    print(f"[*] {len(df)} jogos. Loop temporal...")
    ctx = TemporalContext.with_prior(use_golgg=True)
    rows, labels = [], []
    for i, g in enumerate(df.itertuples(index=False), 1):
        d = g._asdict()
        rows.append(ctx.build_features(d)); labels.append(int(d["blue_won"]))
        ctx.update(d)
        if i % 4000 == 0: print(f"    {i}/{len(df)}")
    X_df = pd.DataFrame(rows); y = np.array(labels)
    split = int(len(df)*0.8)

    all_feats = [c for c in X_df.columns if c != "pre_match_raw"]
    old = fit(X_df, y, all_feats, split)

    block_feats = [f for f in all_feats if f not in PREMATCH_RAW] + ["pre_match_block"]
    Xc = X_df.copy()
    Xc["pre_match_block"] = X_df["pre_match_raw"].clip(-CAP, CAP)
    new = fit(Xc, y, block_feats, split)

    print("\n" + "="*64)
    print("METRICAS")
    print("="*64)
    print(f"{'':16}{'ANTIGO':>12}{'NOVO(cap1.6)':>14}")
    print(f"{'accuracy':16}{old['acc']:>12.4f}{new['acc']:>14.4f}")
    print(f"{'brier':16}{old['brier']:>12.4f}{new['brier']:>14.4f}")
    print(f"{'log_loss':16}{old['ll']:>12.4f}{new['ll']:>14.4f}")

    print("\n" + "="*64)
    print("FORMULA - features pre-jogo")
    print("="*64)
    print("ANTIGO (4 soltas):")
    for f in PREMATCH_RAW:
        print(f"    {f:<22} {old['coef'][f]:+.4f}")
    print(f"  soma |coef| pre = {sum(abs(old['coef'][f]) for f in PREMATCH_RAW):.3f}")
    print("\nNOVO (1 bloco capado):")
    print(f"    {'pre_match_block':<22} {new['coef']['pre_match_block']:+.4f}  (cap {CAP})")

    print("\n" + "="*64)
    print("FORMULA - features de draft (peso que cada uma ganhou)")
    print("="*64)
    draft = ["lane_matchup_diff","last_pick_counter_diff","player_champion_diff",
             "synergy_diff","counter_diff","jungle_mid_synergy_diff","bot_2v2_synergy_diff"]
    print(f"{'feature':<26}{'ANTIGO':>10}{'NOVO':>10}")
    for f in draft:
        print(f"    {f:<22}{old['coef'].get(f,0):>10.4f}{new['coef'].get(f,0):>10.4f}")

    print("\n" + "="*64)
    print("CALIBRATION BUCKETS (ponto de vista do favorito)")
    print("="*64)
    bo, bn = buckets(old['p'], old['y']), buckets(new['p'], new['y'])
    print(f"{'bucket':<12}{'| ANTIGO real/esp/delta':<30}{'| NOVO real/esp/delta'}")
    for (lo,hi,no,ro,eo,do_), (_,_,nn,rn,en,dn) in zip(bo, bn):
        tag = f"[{lo*100:.0f}-{hi*100:.0f}%]"
        olds = f"n{no:<4} {ro:4.1f}/{eo:4.1f}/{do_:+5.1f}" if no else "--"
        news = f"n{nn:<4} {rn:4.1f}/{en:4.1f}/{dn:+5.1f}" if nn else "--"
        print(f"{tag:<12}| {olds:<28}| {news}")


if __name__ == "__main__":
    main()
