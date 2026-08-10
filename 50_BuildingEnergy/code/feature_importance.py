"""Quick feature-importance analysis to source the "45% of domain
feature importance" claim in the paper.

Trains XGBoost (Domain features, seed=42) and reports:
  * top-N domain features by gain importance
  * share of total domain-feature importance held by the top-3
    (dT_indoor_outdoor, THI_out, spatial_T_range)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODEL_CONFIGS, RESULTS_DIR, SEEDS
from data_loader import DOMAIN_FEATURE_COLS, get_feature_matrix
from train import _stratified_split
from xgboost import XGBRegressor


def main() -> int:
    X, y = get_feature_matrix("Domain")
    seed = 42
    X_tr, X_te, y_tr, y_te = _stratified_split(X, y, seed)

    cfg = dict(MODEL_CONFIGS["XGB"])
    cfg["random_state"] = seed
    model = XGBRegressor(**cfg)
    model.fit(X_tr, y_tr)

    importances = model.feature_importances_
    fi = pd.Series(importances, index=X.columns)

    # Domain features only
    dom_fi = fi[DOMAIN_FEATURE_COLS].sort_values(ascending=False)
    dom_total = dom_fi.sum()
    dom_share = dom_fi / dom_total  # share within domain features

    print("=== Domain feature importance (XGBoost, seed=42) ===")
    print(f"{'feature':<22} {'importance':>10} {'% of domain':>12}")
    for name, val in dom_fi.items():
        print(f"{name:<22} {val:>10.4f} {dom_share[name]*100:>11.1f}%")

    # Top-3 by importance
    top3 = dom_fi.head(3)
    top3_sum = top3.sum()
    top3_share = top3_sum / dom_total
    print(f"\nTop-3 domain features: {list(top3.index)}")
    print(f"Top-3 sum of importance: {top3_sum:.4f}")
    print(f"Top-3 share of domain importance: {top3_share*100:.1f}%")

    # Specifically the three named in the paper
    paper_top3 = ["dT_indoor_outdoor", "THI_out", "spatial_T_range"]
    paper_sum = dom_fi[paper_top3].sum()
    paper_share = paper_sum / dom_total
    print(f"\nPaper-named top-3: {paper_top3}")
    print(f"Sum: {paper_sum:.4f}")
    print(f"Share of domain importance: {paper_share*100:.1f}%")

    # Average over all 7 seeds for robustness -- track both the
    # paper-named top-3 and the actual top-3 by importance.
    print("\n=== Averaging over 7 seeds ===")
    shares = []          # paper-named top-3 share
    actual_top3_shares = []  # actual top-3 share per seed
    actual_top3_names = []
    all_dom_fi = {c: [] for c in DOMAIN_FEATURE_COLS}

    for s in SEEDS:
        X_tr2, X_te2, y_tr2, y_te2 = _stratified_split(X, y, s)
        cfg2 = dict(MODEL_CONFIGS["XGB"])
        cfg2["random_state"] = s
        m = XGBRegressor(**cfg2)
        m.fit(X_tr2, y_tr2)
        fi2 = pd.Series(m.feature_importances_, index=X.columns)
        dom2 = fi2[DOMAIN_FEATURE_COLS]
        for c in DOMAIN_FEATURE_COLS:
            all_dom_fi[c].append(float(dom2[c]))
        # paper-named share
        sh = dom2[paper_top3].sum() / dom2.sum()
        shares.append(sh)
        # actual top-3
        top3 = dom2.sort_values(ascending=False).head(3)
        actual_top3_shares.append(top3.sum() / dom2.sum())
        actual_top3_names.append(list(top3.index))
        print(f"  seed={s}: paper_top3={sh*100:.1f}%  "
              f"actual_top3={actual_top3_shares[-1]*100:.1f}%  "
              f"names={actual_top3_names[-1]}")

    shares_arr = np.array(shares)
    actual_arr = np.array(actual_top3_shares)
    print(f"\nPaper-named top-3 mean share: {shares_arr.mean()*100:.1f}%")
    print(f"Actual top-3 mean share:      {actual_arr.mean()*100:.1f}%")

    # Mean importance per feature across seeds
    mean_fi = {c: float(np.mean(v)) for c, v in all_dom_fi.items()}

    # Persist
    out = {
        "paper_top3_features": paper_top3,
        "paper_top3_shares_per_seed": [float(s) for s in shares],
        "paper_top3_mean_share": float(shares_arr.mean()),
        "paper_top3_std_share": float(shares_arr.std(ddof=1)),
        "actual_top3_shares_per_seed": [float(s) for s in actual_top3_shares],
        "actual_top3_names_per_seed": actual_top3_names,
        "actual_top3_mean_share": float(actual_arr.mean()),
        "actual_top3_std_share": float(actual_arr.std(ddof=1)),
        "seed42_actual_top3": list(top3.index),
        "seed42_actual_top3_share": float(top3_share),
        "mean_feature_importance": mean_fi,
    }
    out_path = RESULTS_DIR / "feature_importance_share.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
