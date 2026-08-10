# CDMFeat: Customer Behavior Feature Analysis for Repeat Purchase Prediction

> A customer behavior feature engineering approach for predicting repeat purchase probability using the CDNOW dataset. Domain features capturing RFM (recency, frequency, monetary) patterns and customer lifetime value indicators are compared against raw features across tree-based models.

**Task**: Classification | **Target**: target (repeat purchase: 0/1) | **Primary Metric**: AUC

## Dataset

| Item | Detail |
|------|--------|
| Name | CDNOW (Agrawal & Schreiber, 2001) |
| File | `data/cdnow.csv` |
| Size | 11.28 MB |
| Source | CDNOW dataset - a well-known benchmark for customer behavior analysis (originally from cdnow.com). ~69,659 transaction records with customer purchase history data. |
| Task | Classification |
| Target | target (repeat purchase: 0/1) |
| Metric | AUC |

## Method

PhysXGBoost framework: domain feature engineering + tree models. The approach systematically compares Raw features vs Domain-derived features (incorporating domain knowledge) across four tree-based models: XGBoost, LightGBM, CatBoost, and RandomForest, with 5 random seeds per configuration.

## Directory Structure

```
58_CDNOW/
├── data/          # Dataset files
├── code/          # Source code
├── results/       # Experimental results (JSON/CSV)
├── paper/         # Paper draft
└── plots/         # Figures (PNG, 300 DPI)
```

## Key Results

| Metric | Best Model | Value |
|--------|-----------|-------|
| AUC | LightGBM (Domain features) | AUC = 0.8275 |

> All metrics are computed on the **test set** (20% holdout) and averaged across 5 random seeds [42, 123, 456, 789, 2024]. Results are sourced from `results/summary.json`.

### Result Files

| File | Description |
|------|-------------|
| `summary.json` | Main results: mean/std metrics for Raw vs Domain features, Wilcoxon test p-values |
| `additional_metrics.json` | Additional metrics (Accuracy, F1, RMSE, MAE, 95% CI, Cohen's d) |

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
