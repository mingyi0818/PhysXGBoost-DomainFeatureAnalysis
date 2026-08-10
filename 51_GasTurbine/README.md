# PhysXGBoost: Thermodynamic Domain Features for Gas Turbine NOx Emission Prediction

> A thermodynamic domain feature engineering approach for predicting gas turbine NOx emissions. Domain features include air density, temperature ratios, pressure ratios, combustion efficiency, and thermal efficiency proxies derived from physical relationships.

**Task**: Regression | **Target**: NOX | **Primary Metric**: R2

## Dataset

| Item | Detail |
|------|--------|
| Name | Gas Turbine CO and NOx Emission Dataset (UCI ML Repository) |
| File | `data/gasturbine.csv` |
| Size | 2.67 MB |
| Source | UCI Machine Learning Repository - Gas Turbine CO and NOx Emission Dataset (https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set). 36,733 samples with 11 sensor features. |
| Task | Regression |
| Target | NOX |
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
51_GasTurbine/
├── code/
│   ├── run_experiments.py
├── data/
│   └── gasturbine.csv
├── results/
│   ├── summary.json
│   ├── comprehensive_results.json
│   ├── per_seed_results.json
│   ├── nox_summary.json
│   └── additional_metrics.json
├── plots/
│   ├── fig1_architecture.png
│   ├── fig2_performance_comparison.png
│   ├── fig3_ablation_results.png
│   ├── fig4_sensitivity_analysis.png
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
# Verify gasturbine.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 51_GasTurbine
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
| `comprehensive_results.json` | Extended metrics including all model-variant-seed combinations |
| `per_seed_results.json` | Per-seed breakdown of all metrics |
| `nox_summary.json` | NOx-specific analysis summary |
| `additional_metrics.json` | Supplementary metrics (95% CI, Cohen's d, etc.) |

## Reproduction

For detailed reproduction instructions, see [reproduce.md](reproduce.md).

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
