# PhysXGBoost: Domain Feature Engineering for Household Power Consumption Prediction

> A domain feature engineering approach for predicting household electric power consumption. The experiment compares Raw vs Domain features across four tree-based models.

**Task**: Regression | **Target**: Global_active_power | **Primary Metric**: R2

## Data Integrity Notice

> WARNING: Only 1 result file (summary.json) and 2 plots are available. No per_seed_results.json, comprehensive_results.json, or additional_metrics.json. The experiment appears incomplete - additional result files and plots (ablation, sensitivity, training time) should be generated. R2 scores are very high (0.996-0.999) which may indicate data leakage in the time-series split.

## Dataset

| Item | Detail |
|------|--------|
| Name | Individual Household Electric Power Consumption (UCI ML Repository) |
| File | `data/power.csv` |
| Size | 71.54 MB |
| Source | UCI Machine Learning Repository - Individual Household Electric Power Consumption Dataset (https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). ~2 million samples with measurements at one-minute resolution. |
| Task | Regression |
| Target | Global_active_power |
| Metric | R2 |

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
56_PowerConsumption/
├── code/
│   ├── run_experiments.py
├── data/
│   └── power.csv
├── results/
│   └── summary.json
├── plots/
│   ├── fig1_architecture.png
│   └── fig2_performance_comparison.png
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
# Verify power.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 56_PowerConsumption
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

## Reproduction

For detailed reproduction instructions, see [reproduce.md](reproduce.md).

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
