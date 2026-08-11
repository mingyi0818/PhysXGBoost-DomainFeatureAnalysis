# PowerConsFeat: Data Leakage Detection in Household Power Consumption Prediction via Information-Theoretic Feature Analysis

> An information-theoretic feature analysis approach for household power consumption prediction. This work investigates near-perfect prediction accuracy (R2 > 0.999) and diagnoses potential data leakage through causal and information-theoretic feature analysis.

**Task**: Regression | **Target**: Global_active_power | **Primary Metric**: R2

## Dataset

| Item | Detail |
|------|--------|
| Name | Individual Household Electric Power Consumption (UCI ML Repository) |
| File | `data/power.csv` |
| Size | 71.54 MB |
| Source | UCI Machine Learning Repository - Individual Household Electric Power Consumption Dataset. ~2,075,259 measurements with 7 features collected over 4 years at 1-minute resolution from a household in Sceaux, France. |
| Task | Regression |
| Target | Global_active_power |
| Metric | R2 |

## Method

PhysXGBoost framework: domain feature engineering + tree models. The approach systematically compares Raw features vs Domain-derived features (incorporating domain knowledge) across four tree-based models: XGBoost, LightGBM, CatBoost, and RandomForest, with 5 random seeds per configuration.

## Directory Structure

```
56_PowerConsumption/
├── data/          # Dataset files
├── code/          # Source code
├── results/       # Experimental results (JSON/CSV)
├── paper/         # Paper draft
└── plots/         # Figures (PNG, 300 DPI)
```

## Key Results

| Metric | Best Model | Value |
|--------|-----------|-------|
| R2 | RandomForest (Domain features) | R2 = 0.9998 |

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
