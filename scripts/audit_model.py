"""AUDITORIA MESTRA do modelo de WR de LoL — testa teorias transferidas do Dota.

Roda UM walk temporal (pipeline de producao) e sobre o holdout 80/20 testa:
  A. CALIBRACAO: modelo e sub-confiante (comprimido pro centro)? Platt-scaling ajuda?
     (Dota descobriu sub-confianca e +4pp de boost pagou.)
  B. LEAKAGE: algum bucket de prob alta tem WR ~100% (bandeira vermelha do Dota)?
  D. ABLACAO: para cada feature, delta log loss OOS removendo-a + poder univariado.
     Acha feature morta (|coef| alto != util) e os drivers reais.
  E. RATING: peso/uso do rating_diff (Dota: rating so vale em mismatch).
  F. Reporta coeficientes fitados vs producao.

Uso: python scripts/audit_model.py [--years 2024 2025 2026] [--prior]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.oracle_loader import load_years
from data_pipeline.temporal_context import TemporalContext


def temporal_matrix(df, use_prior):
    ctx = TemporalContext.with_prior(use_golgg=use_prior) if use_prior else TemporalContext()
    rows, labels, meta = [], [], []
    for game in df.itertuples(index=False):
        g = game._asdict()
        rows.append(ctx.build_features(g))
        labels.append(int(g["blue_won"]))
        meta.append({"date": g["date"], "league": g["league"]})
        ctx.update(g)
    X = pd.DataFrame(rows)
    return X, np.array(labels), meta


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    m = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-m)/d, (c+m)/d)


def calibration_report(p_test, y_test):
    print("\n=== A. CALIBRACAO (ponto de vista do FAVORITO) ===")
    p_fav = np.where(p_test >= 0.5, p_test, 1 - p_test)
    y_fav = np.where(p_test >= 0.5, y_test, 1 - y_test)
    buckets = [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.80),(0.80,1.01)]
    print(f"    {'bucket':<12}{'n':>6}{'real':>9}{'previsto':>10}{'delta':>9}  {'Wilson95':>16}")
    for lo, hi in buckets:
        m = (p_fav >= lo) & (p_fav < hi)
        n = int(m.sum())
        if n == 0:
            print(f"    [{lo*100:.0f}-{hi*100:.0f}%]{'':<4}{n:>6}     ---"); continue
        real = y_fav[m].mean(); exp = p_fav[m].mean()
        wl, wh = wilson(int(y_fav[m].sum()), n)
        flag = "  <-- SUB-CONF" if real - exp > 0.03 and wl > exp else ("  <-- OVER" if exp - real > 0.03 and wh < exp else "")
        print(f"    [{lo*100:.0f}-{hi*100:.0f}%]{'':<4}{n:>6}{real*100:>8.1f}%{exp*100:>9.1f}%{(real-exp)*100:>+8.1f}pp  [{wl*100:>4.1f},{wh*100:>4.1f}]{flag}")


def platt_test(p_tr, y_tr, p_te, y_te):
    """Fit Platt (logistic 1D) no TREINO, aplica no teste. slope>1 => sub-confiante."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    eps = 1e-6
    logit_tr = np.log(np.clip(p_tr, eps, 1-eps) / (1 - np.clip(p_tr, eps, 1-eps))).reshape(-1, 1)
    logit_te = np.log(np.clip(p_te, eps, 1-eps) / (1 - np.clip(p_te, eps, 1-eps))).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs")  # sem regularizacao ~ Platt puro
    cal.fit(logit_tr, y_tr)
    slope = cal.coef_[0][0]; intercept = cal.intercept_[0]
    p_te_cal = cal.predict_proba(logit_te)[:, 1]
    print("\n=== A2. PLATT-SCALING (fit no treino, aplicado no teste) ===")
    print(f"    slope={slope:.3f} intercept={intercept:+.3f}  "
          f"({'SUB-confiante (esticar ajuda)' if slope > 1.05 else 'OVER-confiante' if slope < 0.95 else 'bem escalado'})")
    print(f"    log loss teste: cru {log_loss(y_te, p_te):.4f} -> Platt {log_loss(y_te, p_te_cal):.4f} "
          f"(delta {log_loss(y_te, p_te_cal)-log_loss(y_te, p_te):+.4f})")
    return p_te_cal


def leakage_report(p_test, y_test):
    print("\n=== B. LEAKAGE (bandeira: bucket alto com WR ~100%) ===")
    p_fav = np.where(p_test >= 0.5, p_test, 1 - p_test)
    y_fav = np.where(p_test >= 0.5, y_test, 1 - y_test)
    for lo in [0.70, 0.80, 0.90]:
        m = p_fav >= lo
        n = int(m.sum())
        if n >= 10:
            wr = y_fav[m].mean()*100
            flag = "  <-- SUSPEITO" if wr > 95 and n >= 30 else ""
            print(f"    prob>={lo*100:.0f}%: n={n:<5} WR={wr:.1f}%{flag}")
        else:
            print(f"    prob>={lo*100:.0f}%: n={n} (poucos)")
    print("    (>95% em n>=30 = suspeita de leakage; senao calibracao honesta)")


def ablation_report(X, y, split, base_ll):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    print("\n=== D. ABLACAO + PODER UNIVARIADO (delta log loss OOS ao remover) ===")
    cols = list(X.columns)
    Xtr, Xte = X.values[:split], X.values[split:]
    ytr, yte = y[:split], y[split:]
    full = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr)
    coefs = dict(zip(cols, full.coef_[0]))
    results = []
    for j, c in enumerate(cols):
        keep = [k for k in range(len(cols)) if k != j]
        m = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr[:, keep], ytr)
        ll_wo = log_loss(yte, m.predict_proba(Xte[:, keep])[:, 1])
        # univariado
        mu = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr[:, [j]], ytr)
        ll_uni = log_loss(yte, mu.predict_proba(Xte[:, [j]])[:, 1])
        results.append((c, coefs[c], ll_wo - base_ll, ll_uni))
    print(f"    {'feature':<28}{'coef':>8}{'dLL remov':>11}{'LL univ':>10}")
    print(f"    {'(LL_null='+format(log_loss(yte, np.full(len(yte), ytr.mean())),'.4f')+')':<28}")
    for c, coef, dll, lluni in sorted(results, key=lambda r: r[2], reverse=True):
        tag = ""
        if dll <= 0.0002 and abs(coef) > 0.05:
            tag = "  MORTA? (coef alto, remover nao piora)"
        elif dll > 0.002:
            tag = "  <-- DRIVER"
        print(f"    {c:<28}{coef:>+8.3f}{dll:>+11.5f}{lluni:>10.4f}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--prior", action="store_true")
    args = ap.parse_args()

    print(f"[*] Carregando {args.years} (prior={args.prior})...")
    df = load_years(args.years, only_tier1=False).sort_values("date").reset_index(drop=True)
    X, y, meta = temporal_matrix(df, args.prior)
    split = int(len(df) * 0.8)
    print(f"    {len(df)} jogos | teste {len(df)-split} ({df['date'].iloc[split].date()} -> {df['date'].iloc[-1].date()})")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, accuracy_score
    Xtr, Xte = np.nan_to_num(X.values[:split]), np.nan_to_num(X.values[split:])
    ytr, yte = y[:split], y[split:]
    model = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr)
    p_tr = model.predict_proba(Xtr)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    base_ll = log_loss(yte, p_te)
    print(f"    baseline: acc {accuracy_score(yte,(p_te>=0.5).astype(int))*100:.2f}% | log loss {base_ll:.4f}")

    calibration_report(p_te, yte)
    platt_test(p_tr, ytr, p_te, yte)
    leakage_report(p_te, yte)
    ablation_report(pd.DataFrame(np.nan_to_num(X.values), columns=list(X.columns)), y, split, base_ll)


if __name__ == "__main__":
    main()
