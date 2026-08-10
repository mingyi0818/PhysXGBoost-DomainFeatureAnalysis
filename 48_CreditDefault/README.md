# PhysXGBoost: Domain Feature Engineering for Credit Default Prediction

> A domain feature engineering approach for credit card default prediction, comparing Raw vs Domain-derived financial risk features across four tree-based models.

**Task**: Binary Classification | **Target**: default payment next month | **Primary Metric**: AUC

## Data Integrity Notice

> NOTE: Results show Raw and Domain features produce nearly identical AUC scores, indicating domain features did not improve performance for this dataset. Only 2 result files are available (no per_seed_results.json or comprehensive_results.json).

## Dataset

| Item | Detail |
|------|--------|
| Name | Default of Credit Card Clients (UCI ML Repository) |
| File | `data/credit_default.csv` |
| Size | 2.60 MB |
| Source | UCI Machine Learning Repository - Default of Credit Card Clients Dataset (https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). 30,000 samples with 24 features. |
| Task | Binary Classification |
| Target | default payment next month |
| Metric | AUC |

## Environment Requirements

- Python 3.10+
- OS: Windows 11 Professional (tested)
- CPU: Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- RAM: 48 GB DDR5 RDIMM
- GPU: NVIDIA RTX 2000 Pro (16 GB) — not required for tree models

### Python Dependencies

```
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
```

## Directory Structure

```
48_CreditDefault/
├── code/
│   ├── run_experiments.py
├── data/
│   └── credit_default.csv
├── results/
│   ├── summary.json
│   └── additional_metrics.json
├── plots/
│   ├── fig1_architecture.png
│   ├── fig2_performance_comparison.png
│   ├── fig3_feature_importance.png
│   ├── fig4_multi_metric_comparison.png
│   └── fig5_training_time.png
├── paper/
│   └── paper_draft.md
├── reference/
│   └── REFERENCE_MATERIALS.md
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install xgboost lightgbm catboost scikit-learn scipy matplotlib pandas numpy
```

### 2. Verify data is present

```bash
# Verify credit_default.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 48_CreditDefault
```

### 4. Check results

```bash
# Results saved to results/ directory
# Key file: results/summary.json
```

## Result Files

| File | Description |
|------|-------------|
| `summary.json` | Main results: mean/std metrics for Raw vs Domain features, Wilcoxon test p-values |
| `additional_metrics.json` | Supplementary metrics (95% CI, Cohen's d, etc.) |

## Reproduction

For detailed reproduction instructions, see [reproduce.md](reproduce.md).

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
