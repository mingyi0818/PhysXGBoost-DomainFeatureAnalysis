"""Configuration for building energy prediction experiments.

Defines paths, model hyperparameters, seeds, and feature groups used by
train.py.  All numbers reported in the paper must trace back to a result
file produced with the settings in this module.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "energy.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Experiment controls -------------------------------------------------
# Seven independent seeds, matching the paper ("7 independent seeds, 42-48").
SEEDS = [42, 43, 44, 45, 46, 47, 48]

TEST_SIZE = 0.20  # 80/20 train/test split
HOUR_BIN_COUNT = 24  # stratification bins (hour-of-day)

# --- Model hyperparameters (shared across all seeds) ---------------------
N_ESTIMATORS = 300
MAX_DEPTH_BOOST = 6
MAX_DEPTH_RF = 12
LEARNING_RATE = 0.05
N_JOBS = -1

MODEL_CONFIGS = {
    "XGB": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH_BOOST,
        "learning_rate": LEARNING_RATE,
        "n_jobs": N_JOBS,
        "random_state": None,  # set per-seed in train.py
        "verbosity": 0,
        "tree_method": "hist",
    },
    "LGB": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH_BOOST,
        "learning_rate": LEARNING_RATE,
        "n_jobs": N_JOBS,
        "random_state": None,
        "verbose": -1,
    },
    "Cat": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH_BOOST,
        "learning_rate": LEARNING_RATE,
        "random_state": None,
        "verbose": False,
    },
    "RF": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH_RF,
        "n_jobs": N_JOBS,
        "random_state": None,
    },
}

MODEL_ORDER = ["XGB", "LGB", "Cat", "RF"]
FEATURE_SETS = ["Raw", "Domain"]

# --- Sensor groups (used by data_loader) ---------------------------------
INDOOR_TEMP_COLS = [f"T{i}" for i in range(1, 10)]
INDOOR_RH_COLS = [f"RH_{i}" for i in range(1, 10)]
