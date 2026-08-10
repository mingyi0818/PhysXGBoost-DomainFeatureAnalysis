# PhysXGBoost: Real Estate Domain Features for California Housing Price Prediction

> A real estate domain feature engineering approach for predicting California housing prices. Domain features include distances to major cities, room-bedroom ratios, age-based indicators, population density, and income-based derived features.

**Task**: Regression | **Target**: MedHouseVal (median house value) | **Primary Metric**: R2

## Data Integrity Notice

> NOTE: Only 2 result files are available (no per_seed_results.json or comprehensive_results.json). Domain features show modest improvement (R2 0.835 -> 0.841 for XGB).

## Dataset

| Item | Detail |
|------|--------|
| Name | California Housing (scikit-learn / StatLib) |
| File | `data/california_housing.csv` |
| Size | 1.82 MB |
| Source | StatLib repository (Pace & Barry, 1997), accessed via scikit-learn. 20,640 samples with 9 features including MedInc, HouseAge, AveRooms, Latitude, Longitude. |
| Task | Regression |
| Target | MedHouseVal (median house value) |
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
55_CalHousing/
├── code/
│   ├── run_experiments.py
├── data/
│   └── california_housing.csv
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
# Verify california_housing.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 55_CalHousing
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
