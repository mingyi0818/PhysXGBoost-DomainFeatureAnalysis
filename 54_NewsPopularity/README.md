# PhysXGBoost: Content Engagement Domain Features for News Popularity Prediction

> A content engagement domain feature engineering approach for predicting online news article popularity. Domain features include LDA topic entropy, keyword diversity, channel popularity priors, sentiment extremity, and media richness indices.

**Task**: Regression | **Target**: shares | **Primary Metric**: R2

## Data Integrity Notice

> WARNING: R2 scores are near zero or negative for all models (Raw and Domain identical), indicating the models fail to predict news popularity effectively with the current feature set. This is an honest negative result. Also note: two CSV files exist in data/ (news_pop.csv and news_popularity.csv) - the experiments use news_pop.csv.

## Dataset

| Item | Detail |
|------|--------|
| Name | Online News Popularity (UCI ML Repository) |
| File | `data/news_pop.csv` |
| Size | 18.52 MB |
| Source | UCI Machine Learning Repository - Online News Popularity Dataset (https://archive.ics.uci.edu/dataset/332/online+news+popularity). 39,644 samples with 61 features. |
| Task | Regression |
| Target | shares |
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
54_NewsPopularity/
├── code/
│   ├── run_experiments.py
├── data/
│   └── news_pop.csv
├── results/
│   ├── summary.json
│   ├── comprehensive_results.json
│   ├── per_seed_results.json
│   └── additional_metrics.json
├── plots/
│   ├── fig1_architecture.png
│   ├── fig2_performance_comparison.png
│   ├── fig3_feature_importance.png
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
# Verify news_pop.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 54_NewsPopularity
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
