# Building Physics-Derived Feature Augmentation for Tree-Based Appliance Energy Prediction

> A building physics-derived feature augmentation approach for predicting appliance energy consumption. Domain features based on thermodynamic principles, temperature gradients, and temporal patterns are compared against raw features across tree-based models.

**Task**: Regression | **Target**: Appliances | **Primary Metric**: R2

## Dataset

| Item | Detail |
|------|--------|
| Name | Appliances Energy Prediction (UCI ML Repository) |
| File | `data/energy.csv` |
| Size | 4.22 MB |
| Source | UCI Machine Learning Repository - Appliances Energy Prediction Dataset. ~19,735 samples with 29 features covering temperature/humidity sensors, weather data, and energy usage over 4.5 months. |
| Task | Regression |
| Target | Appliances |
| Metric | R2 |

## Method

PhysXGBoost framework: domain feature engineering + tree models. The approach systematically compares Raw features vs Domain-derived features (incorporating domain knowledge) across four tree-based models: XGBoost, LightGBM, CatBoost, and RandomForest, with 5 random seeds per configuration.

## Directory Structure

```
50_BuildingEnergy/
├── data/          # Dataset files
├── code/          # Source code
├── results/       # Experimental results (JSON/CSV)
├── paper/         # Paper draft
└── plots/         # Figures (PNG, 300 DPI)
```

## Key Results

| Metric | Best Model | Value |
|--------|-----------|-------|
| R2 | XGBoost (Domain features) | R2 = 0.4942 |

> All metrics are computed on the **test set** (20% holdout) and averaged across 5 random seeds [42, 123, 456, 789, 2024]. Results are sourced from `results/summary.json`.

### Result Files

| File | Description |
|------|-------------|
| `summary.json` | Main results: mean/std metrics for Raw vs Domain features, Wilcoxon test p-values |
| `comprehensive_results.json` | Ablation and sensitivity analysis |
| `per_seed_results.json` | Per-seed detailed results for each (model, feature_set, seed) |
| `per_seed_results.csv` | Per-seed results in CSV format |
| `additional_metrics.json` | Additional metrics (Accuracy, F1, RMSE, MAE, 95% CI, Cohen's d) |
| `feature_importance_share.json` | Feature importance share analysis |
| `run_log.txt` | Experiment run log |

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
