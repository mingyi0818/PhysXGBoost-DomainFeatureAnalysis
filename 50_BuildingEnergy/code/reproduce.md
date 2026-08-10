# Reproduction Guide — 50_BuildingEnergy

This document describes how to reproduce every numerical result reported
in `paper/paper_draft.md`.  All numbers in the paper trace back to files
under `results/`.

## 1. Environment

### Hardware
- CPU: Intel Xeon W7-2595X (24 cores, 2.5--4.8 GHz)
- RAM: 48 GB DDR5 RDIMM
- GPU: NVIDIA RTX pro 2000 (16 GB) — *not used; tree models are CPU-only*
- OS: Windows 11 Professional

### Software
- Python 3.10 (conda env `TraeAI-4`)
- Dependencies listed in `code/requirements.txt`:
  - xgboost==3.3.0
  - lightgbm==4.6.0
  - catboost==1.2.10
  - scikit-learn==1.9.0
  - scipy>=1.11.0
  - pandas>=2.0.0
  - numpy>=1.24.0

Install with:
```bash
pip install -r code/requirements.txt
```

## 2. Data

- Source: UCI Appliances Energy Prediction dataset (Candanedo et al., 2017)
- Local path: `data/energy.csv`
- Samples: 19,735 rows × 29 columns
- Target variable: `y` (appliance energy consumption in Wh)
- Target statistics: mean 97.69 Wh, range 10--1080 Wh

The CSV is self-contained; no download step is required.

## 3. Running the experiments

All scripts must be run from the `code/` directory so that the relative
imports (`from config import ...`, `from data_loader import ...`) resolve
correctly.

### 3.1 Main experiment (Table 2 + Wilcoxon p-values)

```bash
cd code
python train.py
```

What it does:
1. Loads `data/energy.csv` via `data_loader.load_raw_frame()`.
2. Builds the Raw feature matrix (27 columns) and the Domain feature
   matrix (27 raw + 14 physics-derived = 41 columns) via
   `data_loader.build_domain_features()`.
3. For each of the 4 models (XGB, LGB, Cat, RF) × 2 feature sets
   (Raw, Domain) × 7 seeds (42--48), performs a stratified 80/20
   train/test split (stratified by hour-of-day) and trains the model
   with the hyperparameters in `config.py`.
4. Records test-set R² for every (model, feature_set, seed) triple.
5. Computes mean ± SD across seeds and runs a one-sided Wilcoxon
   signed-rank test (Domain > Raw) per model.

Output files (all under `results/`):
| File | Contents |
|------|----------|
| `per_seed_results.csv` | One row per (feature_set, model, seed) with R² |
| `per_seed_results.json` | Same data, nested dict |
| `summary.json` | Mean/SD R² per (feature_set, model) + Wilcoxon p-values |
| `run_log.txt` | Human-readable log of every fit |

Expected runtime: ~90 seconds on the reference hardware.

### 3.2 Feature importance analysis (Section 3 paragraph + Discussion)

```bash
cd code
python feature_importance.py
```

What it does:
1. Trains XGBoost on the Domain feature set for each of the 7 seeds.
2. Extracts `feature_importances_` (gain-based) and isolates the 14
   domain-derived features.
3. Reports the top-3 domain features per seed and their combined share
   of total domain-feature importance.
4. Also reports the mean importance of each domain feature across seeds.

Output file: `results/feature_importance_share.json`

Key fields used by the paper:
- `seed42_actual_top3_share` = 0.5181 → paper reports "52%"
- `mean_feature_importance` → used to source the "5--6% each" claim
  for enthalpy_indoor (5.8%) and THI_indoor (5.3%), and the "3--5% each"
  claim for stack_effect (4.9%) and wind_chill (4.5%).

## 4. Traceability map (paper number → result file)

| Paper location | Number | Source file | Source field |
|----------------|--------|-------------|--------------|
| Abstract | XGB 0.469→0.494 | summary.json | Raw.XGB.R2, Domain.XGB.R2 |
| Abstract | LGB 0.430→0.460 | summary.json | Raw.LGB.R2, Domain.LGB.R2 |
| Abstract | Cat 0.340→0.379 | summary.json | Raw.Cat.R2, Domain.Cat.R2 |
| Abstract | RF 0.425→0.472 | summary.json | Raw.RF.R2, Domain.RF.R2 |
| Abstract | 2.6--4.7 pp | summary.json | (Domain.R2 - Raw.R2) × 100 |
| Abstract | CatBoost 11.5% | summary.json | (Domain.Cat.R2 - Raw.Cat.R2) / Raw.Cat.R2 |
| Table 2 | all R² ± SD | summary.json | *.R2, *.std |
| Table 2 | ΔR² column | summary.json | (Domain.R2 - Raw.R2) × 100 |
| Section 3 | p=0.0078 | summary.json | wilcoxon.*.p_value |
| Section 3 | 52% top-3 share | feature_importance_share.json | seed42_actual_top3_share |
| Section 3 | 5--6% each | feature_importance_share.json | mean_feature_importance (enthalpy_indoor, THI_indoor) |
| Section 4 | 3--5% each | feature_importance_share.json | mean_feature_importance (stack_effect, wind_chill) |
| Section 3.1 | 0.34--0.49 range | summary.json | min(Raw.*.R2), max(Domain.*.R2) |

## 5. Hyperparameters (config.py)

| Parameter | Value |
|-----------|-------|
| Seeds | [42, 43, 44, 45, 46, 47, 48] |
| Test size | 0.20 (stratified by hour-of-day) |
| n_estimators | 300 |
| max_depth (boosting) | 6 |
| max_depth (RF) | 12 |
| learning_rate | 0.05 |
| XGB tree_method | hist |
| Stratification bins | 24 (one per hour) |

## 6. Domain feature definitions (data_loader.py)

14 derived features appended to the Raw set for the Domain condition:

| # | Feature | Formula |
|---|---------|---------|
| 1 | THI_out | T_out - 0.55·(1-RH_out/100)·(T_out-14.5) |
| 2 | T_dew_indoor | Magnus-Tetens on mean indoor T, RH |
| 3 | dT_indoor_outdoor | mean(T1..T9) - T_out |
| 4 | enthalpy_out | T_out·(1.01 + 1.88·RH_out/100) |
| 5 | stack_effect | dT_indoor_outdoor · (Press_mm_hg/760) |
| 6 | wind_chill | 13.12 + 0.6215·T_out - 11.37·W^0.16 + 0.3965·T_out·W^0.16 |
| 7 | spatial_T_range | max(T1..T9) - min(T1..T9) |
| 8 | spatial_RH_range | max(RH_1..RH_9) - min(RH_1..RH_9) |
| 9 | T_indoor_mean | mean(T1..T9) |
| 10 | RH_indoor_mean | mean(RH_1..RH_9) |
| 11 | THI_indoor | T_indoor_mean - 0.55·(1-RH_indoor_mean/100)·(T_indoor_mean-14.5) |
| 12 | enthalpy_indoor | T_indoor_mean·(1.01 + 1.88·RH_indoor_mean/100) |
| 13 | hour_sin | sin(2π·hour/24) |
| 14 | hour_cos | cos(2π·hour/24) |

## 7. Notes on reproducibility

- The stratified split uses `sklearn.model_selection.StratifiedShuffleSplit`
  with `random_state=seed`, so the exact same train/test partition is
  produced for a given seed on any machine with the same scikit-learn
  version.
- XGBoost, LightGBM, CatBoost, and RandomForest all accept a
  `random_state` parameter; passing the seed guarantees identical
  bootstrap / feature-subset draws.
- The Wilcoxon p-value of 0.0078125 corresponds to the exact one-sided
  probability of observing 7/7 positive differences under the null
  (2^-7 = 0.0078125).
- Minor numerical differences (±0.001 in R²) may appear across CPU
  architectures or library versions due to floating-point summation
  order, but the reported 3-decimal values are stable.
