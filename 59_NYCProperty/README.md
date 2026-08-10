# PhysXGBoost: Real Estate Domain Features for NYC Property Sale Price Prediction

> A real estate domain feature engineering approach for predicting New York City property sale prices. Domain features include location-based indicators, building age, unit density, and tax-class interactions. Includes SHAP analysis and feature saturation analysis.

**Task**: Regression | **Target**: SALE PRICE | **Primary Metric**: R2

## Dataset

| Item | Detail |
|------|--------|
| Name | NYC Property Sales (Kaggle / NYC Department of Finance) |
| File | `data/nyc_property_sales.csv` |
| Size | 12.99 MB |
| Source | New York City Department of Finance Rolling Sales dataset, available via Kaggle (https://www.kaggle.com/datasets/new-york-city/nyc-property-sales). Contains property sale records across NYC boroughs. |
| Task | Regression |
| Target | SALE PRICE |
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
59_NYCProperty/
├── code/
│   ├── run_experiments.py
├── data/
│   └── nyc_property_sales.csv
├── results/
│   ├── summary.json
│   ├── comprehensive_results.json
│   ├── per_seed_results.json
│   ├── shap_and_saturation.json
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
# Verify nyc_property_sales.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 59_NYCProperty
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
| `shap_and_saturation.json` | SHAP values and feature saturation analysis |
| `additional_metrics.json` | Supplementary metrics (95% CI, Cohen's d, etc.) |

## Reproduction

For detailed reproduction instructions, see [reproduce.md](reproduce.md).

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
