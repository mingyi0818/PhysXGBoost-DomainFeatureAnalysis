# PhysXGBoost: Domain Feature Engineering for Customer Purchase Prediction (CDNOW)

> A domain feature engineering approach for predicting customer purchase behavior using the CDNOW dataset. Domain features capture RFM (Recency, Frequency, Monetary) patterns and customer lifecycle indicators.

**Task**: Classification | **Target**: target (purchase behavior indicator) | **Primary Metric**: AUC

## Data Integrity Notice

> NOTE: Only 2 result files are available (no per_seed_results.json or comprehensive_results.json). Domain features show minimal improvement over Raw features (AUC 0.827 -> 0.827).

## Dataset

| Item | Detail |
|------|--------|
| Name | CDNOW customer transaction dataset |
| File | `data/cdnow.csv` |
| Size | 11.28 MB |
| Source | CDNOW dataset (Fader & Hardie, 2001), originally from the Center for Retail Management at Wharton. Contains transaction-level purchase history for 2,357 customers. |
| Task | Classification |
| Target | target (purchase behavior indicator) |
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
58_CDNOW/
├── code/
│   ├── run_experiments.py
├── data/
│   └── cdnow.csv
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
# Verify cdnow.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 58_CDNOW
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
