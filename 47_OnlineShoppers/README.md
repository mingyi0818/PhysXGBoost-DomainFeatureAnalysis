# PhysXGBoost: Domain Feature Engineering for Online Shopper Purchase Intention Prediction

> A comparative study of domain-engineered features vs raw features across four tree-based models for predicting online shopping purchase intention. Domain features capture browsing behavior patterns, seasonal effects, and user engagement metrics.

**Task**: Binary Classification | **Target**: y (purchase intention: 0/1) | **Primary Metric**: AUC

## Dataset

| Item | Detail |
|------|--------|
| Name | Online Shoppers Intention (UCI ML Repository) |
| File | `data/online_shoppers.csv` |
| Size | 1.07 MB |
| Source | UCI Machine Learning Repository - Online Shoppers Purchasing Intention Dataset (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). 12,330 sessions with 18 features. |
| Task | Binary Classification |
| Target | y (purchase intention: 0/1) |
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
47_OnlineShoppers/
├── code/
│   ├── run_experiments.py
├── data/
│   └── online_shoppers.csv
├── results/
│   ├── summary.json
│   ├── comprehensive_results.json
│   ├── per_seed_results.json
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
# Verify online_shoppers.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 47_OnlineShoppers
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
| `additional_metrics.json` | Supplementary metrics (95% CI, Cohen's d, etc.) |

## Reproduction

For detailed reproduction instructions, see [reproduce.md](reproduce.md).

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
