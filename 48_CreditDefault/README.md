# InfoRedund: An Information-Theoretic Framework Explaining When Domain Features Fail for Tree-Based Credit Default Prediction

> An information-theoretic analysis of when domain feature engineering fails to improve tree-based models. Using credit default prediction as a case study, this work demonstrates that domain features can be redundant when the original feature space already saturates the mutual information with the target.

**Task**: Classification | **Target**: default.payment.next.month | **Primary Metric**: AUC

## Dataset

| Item | Detail |
|------|--------|
| Name | Default of Credit Card Clients (UCI ML Repository) |
| File | `data/credit_default.csv` |
| Size | 2.60 MB |
| Source | UCI Machine Learning Repository - Default of Credit Card Clients Dataset. 30,000 samples with 24 features covering credit limit, payment history, bill statements, and demographic information. |
| Task | Classification |
| Target | default.payment.next.month |
| Metric | AUC |

## Method

PhysXGBoost framework: domain feature engineering + tree models. The approach systematically compares Raw features vs Domain-derived features (incorporating domain knowledge) across four tree-based models: XGBoost, LightGBM, CatBoost, and RandomForest, with 5 random seeds per configuration.

## Directory Structure

```
48_CreditDefault/
├── data/          # Dataset files
├── code/          # Source code
├── results/       # Experimental results (JSON/CSV)
├── paper/         # Paper draft
└── plots/         # Figures (PNG, 300 DPI)
```

## Key Results

| Metric | Best Model | Value |
|--------|-----------|-------|
| AUC | CatBoost (Raw = Domain; features are information-redundant) | AUC = 0.7802 |

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
