# FinFeat: Systematic Domain-Derived Feature Augmentation for Tree-Based Bank Marketing Prediction

> A domain feature engineering approach for bank marketing prediction using tree-based models. The experiment systematically compares Raw vs Domain-derived features across four gradient boosting and ensemble methods to quantify the impact of domain knowledge injection.

**Task**: Classification | **Target**: y (term deposit subscription: yes/no) | **Primary Metric**: AUC, F1

## Dataset

| Item | Detail |
|------|--------|
| Name | Bank Marketing (UCI ML Repository) |
| File | `data/bank_marketing.csv` |
| Size | 3.23 MB |
| Source | UCI Machine Learning Repository - Bank Marketing Dataset (https://archive.ics.uci.edu/dataset/222/bank+marketing). ~45,000 samples with 21 attributes covering client demographics, financial status, campaign contacts, and macroeconomic indicators. |
| Task | Classification |
| Target | y (term deposit subscription: yes/no) |
| Metric | AUC, F1 |

## Method

PhysXGBoost framework: domain feature engineering + tree models. The approach systematically compares Raw features vs Domain-derived features (incorporating domain knowledge) across four tree-based models: XGBoost, LightGBM, CatBoost, and RandomForest, with 5 random seeds per configuration.

## Directory Structure

```
46_FlightDelay_PhysXGBoost/
├── data/          # Dataset files
├── code/          # Source code
├── results/       # Experimental results (JSON/CSV)
├── paper/         # Paper draft
└── plots/         # Figures (PNG, 300 DPI)
```

## Key Results

| Metric | Best Model | Value |
|--------|-----------|-------|
| AUC, F1 | LightGBM (Raw features) | AUC = 0.9388, F1 = 0.5921 |

> All metrics are computed on the **test set** (20% holdout) and averaged across 5 random seeds [42, 123, 456, 789, 2024]. Results are sourced from `results/summary.json`.

### Result Files

| File | Description |
|------|-------------|
| `summary.json` | Main results: mean/std metrics for Raw vs Domain features, Wilcoxon test p-values |
| `comprehensive_results.json` | Ablation and sensitivity analysis |
| `per_seed_results.json` | Per-seed detailed results for each (model, feature_set, seed) |
| `additional_metrics.json` | Additional metrics (Accuracy, F1, RMSE, MAE, 95% CI, Cohen's d) |
| `additional_metrics_v2.json` | Extended additional metrics v2 |

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
