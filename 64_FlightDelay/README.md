# PhysXGBoost: Domain Feature Engineering for Flight Delay Prediction

> A domain feature engineering approach for predicting flight delays. The experiment compares Raw vs Domain features across four tree-based models.

**Task**: Classification | **Target**: Unknown | **Primary Metric**: AUC

## Data Integrity Notice

> CRITICAL: The data/ directory is completely missing. No data file exists for this direction. The results/summary.json shows AUC scores near 1.0 for all models, which is suspiciously high and likely indicates data leakage or an incorrect experiment setup. Only 1 result file and 2 plots are available. This direction requires data acquisition and experiment re-running before any results can be trusted.

## Dataset

| Item | Detail |
|------|--------|
| Name | Flight Delay (source TBD) |
| File | `NO DATA FILE FOUND - data/ directory does not exist` |
| Size | 0.00 MB |
| Source | Unknown - no data file is present in the data/ directory. The data/ directory is missing entirely. |
| Task | Classification |
| Target | Unknown |
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
64_FlightDelay/
├── code/
│   ├── run_experiments.py
├── data/
│   └── (missing)
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

> **WARNING**: No data file found. Please obtain the dataset first.


### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 64_FlightDelay
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
