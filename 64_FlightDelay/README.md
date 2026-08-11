# FlightFeat: Data Leakage Diagnosis in Flight Delay Prediction via Causal Feature Analysis

> A causal feature analysis approach for diagnosing data leakage in flight delay prediction. This work investigates near-perfect prediction accuracy (AUC > 0.9999) and identifies causally problematic features through information-theoretic and causal inference techniques.

**Task**: Classification | **Target**: Flight delay label (delayed/not delayed) | **Primary Metric**: AUC

## Dataset

| Item | Detail |
|------|--------|
| Name | Flight Delay (source TBD) |
| File | `data/ (directory missing - data file needs to be obtained)` |
| Size | N/A (data file not present) |
| Source | Flight delay dataset - source to be determined. The data/ directory is currently missing and must be populated before experiments can be re-run. |
| Task | Classification |
| Target | Flight delay label (delayed/not delayed) |
| Metric | AUC |

## Method

PhysXGBoost framework: domain feature engineering + tree models. The approach systematically compares Raw features vs Domain-derived features (incorporating domain knowledge) across four tree-based models: XGBoost, LightGBM, CatBoost, and RandomForest, with 5 random seeds per configuration.

## Directory Structure

```
64_FlightDelay/
├── data/          # Dataset files
├── code/          # Source code
├── results/       # Experimental results (JSON/CSV)
├── paper/         # Paper draft
└── plots/         # Figures (PNG, 300 DPI)
```

## Key Results

| Metric | Best Model | Value |
|--------|-----------|-------|
| AUC | LightGBM (Domain features, AUC near 1.0 - suspected leakage) | AUC = 0.9999936 |

> All metrics are computed on the **test set** (20% holdout) and averaged across 5 random seeds [42, 123, 456, 789, 2024]. Results are sourced from `results/summary.json`.

### Result Files

| File | Description |
|------|-------------|
| `summary.json` | Main results: mean/std metrics for Raw vs Domain features, Wilcoxon test p-values |

## Environment Requirements

- Python 3.10+
- OS: Windows 11 Professional (tested)
- CPU: Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- RAM: 48 GB DDR5 RDIMM
- GPU: NVIDIA RTX 2000 Pro (16 GB) -- not required for tree models

### Python Dependencies

```
xgboost>=2.0.0, lightgbm>=4.0.0, catboost>=1.2.0, scikit-learn>=1.3.0, pandas>=2.0.0, numpy>=1.24.0, scipy>=1.11.0, matplotlib>=3.7.0
```

## How to Reproduce

See [reproduce.md](reproduce.md) for detailed step-by-step instructions.

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
