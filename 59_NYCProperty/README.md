# NYCPropFeat: Real Estate Domain Feature Analysis for Property Price Prediction with Information-Theoretic Saturation Bounds

> A real estate domain feature engineering approach for predicting NYC property sale prices with information-theoretic saturation analysis. Domain features capturing spatial, building, and market characteristics are compared against raw features, with SHAP analysis and saturation bounds providing theoretical guarantees.

**Task**: Regression | **Target**: SALE PRICE | **Primary Metric**: R2

## Dataset

| Item | Detail |
|------|--------|
| Name | NYC Property Sales (NYC Department of Finance) |
| File | `data/nyc_property_sales.csv` |
| Size | 12.99 MB |
| Source | NYC Open Data - NYC Property Sales Dataset from the Department of Finance. Property sale records for NYC boroughs with building characteristics, tax class, and sale price information. |
| Task | Regression |
| Target | SALE PRICE |
| Metric | R2 |

## Method

PhysXGBoost framework: domain feature engineering + tree models. The approach systematically compares Raw features vs Domain-derived features (incorporating domain knowledge) across four tree-based models: XGBoost, LightGBM, CatBoost, and RandomForest, with 5 random seeds per configuration.

## Directory Structure

```
59_NYCProperty/
├── data/          # Dataset files
├── code/          # Source code
├── results/       # Experimental results (JSON/CSV)
├── paper/         # Paper draft
└── plots/         # Figures (PNG, 300 DPI)
```

## Key Results

| Metric | Best Model | Value |
|--------|-----------|-------|
| R2 | XGBoost (Domain features) | R2 = 0.6592 |

> All metrics are computed on the **test set** (20% holdout) and averaged across 5 random seeds [42, 123, 456, 789, 2024]. Results are sourced from `results/summary.json`.

### Result Files

| File | Description |
|------|-------------|
| `summary.json` | Main results: mean/std metrics for Raw vs Domain features, Wilcoxon test p-values |
| `comprehensive_results.json` | Ablation and sensitivity analysis |
| `per_seed_results.json` | Per-seed detailed results for each (model, feature_set, seed) |
| `additional_metrics.json` | Additional metrics (Accuracy, F1, RMSE, MAE, 95% CI, Cohen's d) |
| `shap_and_saturation.json` | SHAP values and saturation analysis |

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
