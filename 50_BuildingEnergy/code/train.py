"""Main experiment: train four tree models on Raw vs Domain features
across seven random seeds and record test-set R^2.

Output files (all under results/):
  * per_seed_results.csv   -- one row per (feature_set, model, seed)
  * per_seed_results.json  -- same data, nested dict
  * summary.json           -- mean/std R^2 per (feature_set, model)
                              plus Wilcoxon signed-rank p-values
  * run_log.txt            -- human-readable log of every fit

Run from the code/ directory:
    python train.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wilcoxon

from config import (
    FEATURE_SETS,
    HOUR_BIN_COUNT,
    MODEL_CONFIGS,
    MODEL_ORDER,
    RESULTS_DIR,
    SEEDS,
    TEST_SIZE,
)
from data_loader import get_feature_matrix

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    print(f"[WARN] XGBoost unavailable: {exc}")
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception as exc:  # pragma: no cover
    print(f"[WARN] LightGBM unavailable: {exc}")
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception as exc:  # pragma: no cover
    print(f"[WARN] CatBoost unavailable: {exc}")
    CatBoostRegressor = None


LOG_PATH = RESULTS_DIR / "run_log.txt"


def _open_log():
    return open(LOG_PATH, "w", encoding="utf-8")


def _stratified_split(X: pd.DataFrame, y: pd.Series, seed: int):
    """Stratified 80/20 split keyed on hour-of-day bins.

    The paper states: "random 80/20 train/test splits stratified by
    hour-of-day bins".  We reconstruct hour-of-day from the circular
    encodings present in the Domain feature set; for the Raw set we
    re-derive it from the raw timestamp via data_loader.
    """
    # hour-of-day is recoverable from hour_cos alone (unique per hour in
    # [0,23]); but to be safe and fully deterministic we re-read the
    # timestamp here.
    from data_loader import load_raw_frame

    ts = load_raw_frame()["date"]
    hour = ts.dt.hour.to_numpy()

    # Bin into HOUR_BIN_COUNT bins (one per hour, 0..23).
    bins = np.clip(hour, 0, HOUR_BIN_COUNT - 1).astype(int)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(X.to_numpy(), bins))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
    )


def _make_model(name: str, seed: int):
    cfg = dict(MODEL_CONFIGS[name])
    cfg["random_state"] = seed
    if name == "XGB":
        if XGBRegressor is None:
            return None
        return XGBRegressor(**cfg)
    if name == "LGB":
        if LGBMRegressor is None:
            return None
        return LGBMRegressor(**cfg)
    if name == "Cat":
        if CatBoostRegressor is None:
            return None
        return CatBoostRegressor(**cfg)
    if name == "RF":
        return RandomForestRegressor(**cfg)
    raise ValueError(name)


def run_one(feature_set: str, model_name: str, seed: int, log) -> dict:
    """Train one (feature_set, model, seed) combination and return a
    result record.  Returns {"r2": None, "error": msg} on failure."""
    log.write(
        f"[{time.strftime('%H:%M:%S')}] start  fs={feature_set:<6} "
        f"model={model_name:<3} seed={seed}\n"
    )
    log.flush()
    t0 = time.time()
    try:
        X, y = get_feature_matrix(feature_set)
        X_tr, X_te, y_tr, y_te = _stratified_split(X, y, seed)

        model = _make_model(model_name, seed)
        if model is None:
            msg = f"{model_name} not installed; skipped"
            log.write(f"    SKIP: {msg}\n")
            return {"feature_set": feature_set, "model": model_name,
                    "seed": seed, "r2": None, "error": msg}

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        r2 = float(r2_score(y_te, preds))
        elapsed = time.time() - t0
        log.write(
            f"    done  r2={r2:.6f}  ({elapsed:.1f}s)\n"
        )
        log.flush()
        return {"feature_set": feature_set, "model": model_name,
                "seed": seed, "r2": r2, "error": None}
    except Exception as exc:  # pragma: no cover
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        log.write(f"    ERROR after {elapsed:.1f}s: {exc}\n{tb}\n")
        log.flush()
        return {"feature_set": feature_set, "model": model_name,
                "seed": seed, "r2": None, "error": str(exc)}


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = _open_log()
    log.write(f"Experiment start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"seeds={SEEDS}\nmodels={MODEL_ORDER}\n"
              f"feature_sets={FEATURE_SETS}\n\n")

    print(f"Running {len(MODEL_ORDER) * len(FEATURE_SETS) * len(SEEDS)} "
          f"fits...")

    records: list[dict] = []
    for feature_set in FEATURE_SETS:
        for model_name in MODEL_ORDER:
            for seed in SEEDS:
                rec = run_one(feature_set, model_name, seed, log)
                records.append(rec)
                if rec["r2"] is None:
                    print(f"  SKIP/ERR fs={feature_set} model={model_name} "
                          f"seed={seed}: {rec['error']}")
                else:
                    print(f"  fs={feature_set:<6} model={model_name:<3} "
                          f"seed={seed} r2={rec['r2']:.4f}")

    # --- Persist per-seed records ----------------------------------------
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "per_seed_results.csv", index=False)

    nested: dict[str, dict[str, dict[int, float]]] = {}
    for fs in FEATURE_SETS:
        nested[fs] = {}
        for m in MODEL_ORDER:
            sub = df[(df["feature_set"] == fs) & (df["model"] == m)]
            nested[fs][m] = {
                int(row["seed"]): (None if pd.isna(row["r2"])
                                   else float(row["r2"]))
                for _, row in sub.iterrows()
            }
    with open(RESULTS_DIR / "per_seed_results.json", "w") as fh:
        json.dump(nested, fh, indent=2)

    # --- Summary (mean / std / Wilcoxon) ---------------------------------
    summary: dict[str, dict[str, dict]] = {}
    for fs in FEATURE_SETS:
        summary[fs] = {}
        for m in MODEL_ORDER:
            vals = np.array(
                [v for v in nested[fs][m].values() if v is not None],
                dtype=float,
            )
            if vals.size == 0:
                summary[fs][m] = {"R2": None, "std": None, "n_seeds": 0}
                continue
            summary[fs][m] = {
                "R2": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "n_seeds": int(vals.size),
                "all_r2": [float(v) for v in vals],
            }

    # Wilcoxon signed-rank: Raw vs Domain for each model, across seeds.
    summary["wilcoxon"] = {}
    for m in MODEL_ORDER:
        raw_vals = [v for v in nested["Raw"][m].values() if v is not None]
        dom_vals = [v for v in nested["Domain"][m].values() if v is not None]
        n = min(len(raw_vals), len(dom_vals))
        if n >= 3:
            raw_arr = np.array(raw_vals[:n])
            dom_arr = np.array(dom_vals[:n])
            diffs = dom_arr - raw_arr
            if np.all(diffs == 0):
                p = 1.0
                stat = 0.0
            else:
                try:
                    stat, p = wilcoxon(dom_arr, raw_arr,
                                       alternative="greater")
                except Exception:
                    stat, p = float("nan"), float("nan")
            summary["wilcoxon"][m] = {
                "n_pairs": int(n),
                "statistic": float(stat) if not np.isnan(stat) else None,
                "p_value": float(p) if not np.isnan(p) else None,
                "all_positive": bool(np.all(diffs > 0)),
            }
        else:
            summary["wilcoxon"][m] = {"n_pairs": int(n),
                                      "statistic": None,
                                      "p_value": None,
                                      "all_positive": None}

    with open(RESULTS_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # --- Console summary --------------------------------------------------
    log.write("\n=== Summary ===\n")
    print("\n=== Summary (mean R^2 +/- std) ===")
    print(f"{'Model':<6} {'Raw':<22} {'Domain':<22} delta")
    for m in MODEL_ORDER:
        r = summary["Raw"][m]
        d = summary["Domain"][m]
        if r["R2"] is None or d["R2"] is None:
            line = f"{m:<6} (missing)"
        else:
            line = (f"{m:<6} {r['R2']:.4f}+/-{r['std']:.4f}   "
                    f"{d['R2']:.4f}+/-{d['std']:.4f}   "
                    f"+{(d['R2'] - r['R2']):.4f}")
        print(line)
        log.write(line + "\n")

    log.write("\nWilcoxon p-values (Domain > Raw):\n")
    for m in MODEL_ORDER:
        w = summary["wilcoxon"][m]
        log.write(f"  {m}: p={w['p_value']} n={w['n_pairs']} "
                  f"all_pos={w['all_positive']}\n")

    log.write(f"\nDone {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.close()
    print(f"\nResults written to: {RESULTS_DIR}")
    print(f"  - per_seed_results.csv")
    print(f"  - per_seed_results.json")
    print(f"  - summary.json")
    print(f"  - run_log.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
