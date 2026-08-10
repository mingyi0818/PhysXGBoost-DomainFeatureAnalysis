# PhysXGBoost: Domain Feature Engineering for Hotel Booking Cancellation Prediction

> A domain feature engineering approach for predicting hotel booking cancellations. The experiment compares Raw vs Domain features across four tree-based models.

**Task**: Binary Classification | **Target**: is_canceled | **Primary Metric**: AUC

## Data Integrity Notice

> WARNING: Only 1 result file (summary.json) and 2 plots are available. No per_seed_results.json, comprehensive_results.json, or additional_metrics.json. The experiment appears incomplete - additional result files and plots (ablation, sensitivity, training time) should be generated.

## Dataset

| Item | Detail |
|------|--------|
| Name | Hotel Booking Demand (Kaggle) |
| File | `data/hotel.csv` |
| Size | 15.18 MB |
| Source | Hotel Booking Demand dataset, originally published in Antonio et al. (2019), available via Kaggle (https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). ~119,000 booking records with 32 features. |
| Task | Binary Classification |
| Target | is_canceled |
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
63_HotelBooking/
├── code/
│   ├── run_experiments.py
├── data/
│   └── hotel.csv
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
# Verify hotel.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 63_HotelBooking
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
