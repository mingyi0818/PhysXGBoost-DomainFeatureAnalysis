# PhysXGBoost: Physics-Informed Domain Features for Building Energy Prediction

> A physics-informed feature engineering approach that derives 14 thermodynamic domain features (enthalpy, THI, stack effect, wind chill, etc.) for appliance energy consumption prediction. Compares Raw vs Domain features across four tree-based models with 7-seed statistical validation.

**Task**: Regression | **Target**: y (appliance energy consumption in Wh) | **Primary Metric**: R2

## Dataset

| Item | Detail |
|------|--------|
| Name | Appliances Energy Prediction (UCI ML Repository) |
| File | `data/energy.csv` |
| Size | 4.22 MB |
| Source | UCI Machine Learning Repository - Appliances Energy Prediction Dataset (https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction). 19,735 samples with 29 columns (Candanedo et al., 2017). |
| Task | Regression |
| Target | y (appliance energy consumption in Wh) |
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
50_BuildingEnergy/
├── code/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_importance.py
│   ├── run_experiments.py
│   ├── train.py
│   ├── verify_traceability.py
│   ├── reproduce.md
│   ├── requirements.txt
├── data/
│   └── energy.csv
├── results/
│   ├── summary.json
│   ├── per_seed_results.json
│   ├── per_seed_results.csv
│   ├── feature_importance_share.json
│   ├── additional_metrics.json
│   └── run_log.txt
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
pip install -r code/requirements.txt
```

### 2. Verify data is present

```bash
# Verify energy.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 50_BuildingEnergy
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
| `per_seed_results.json` | Per-seed breakdown of all metrics |
| `per_seed_results.csv` | Per-seed results in CSV format |
| `feature_importance_share.json` | Feature importance analysis for domain features |
| `additional_metrics.json` | Supplementary metrics (95% CI, Cohen's d, etc.) |
| `run_log.txt` | Experiment run log with timestamps |

## Reproduction

For detailed reproduction instructions, see [reproduce.md](reproduce.md).

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
