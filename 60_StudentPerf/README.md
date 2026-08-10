# PhysXGBoost: Educational Domain Features for Student Performance Prediction

> An educational domain feature engineering approach for predicting student academic performance. Domain features include study efficiency, parental education composites, social-drinking scores, health-lifestyle indicators, and social-study balance metrics.

**Task**: Regression | **Target**: G3 (final grade) | **Primary Metric**: R2

## Data Integrity Notice

> NOTE: R2 scores are low (0.17-0.28), which is expected for student performance prediction with limited features. G1 and G2 (prior grades) are dropped to avoid leakage, making prediction more challenging. Domain features show mixed results.

## Dataset

| Item | Detail |
|------|--------|
| Name | Student Performance (UCI ML Repository) |
| File | `data/student.csv` |
| Size | 0.06 MB |
| Source | UCI Machine Learning Repository - Student Performance Dataset (https://archive.ics.uci.edu/dataset/320/student+performance). 649 samples (math + Portuguese) with 33 features (Cortez & Silva, 2008). |
| Task | Regression |
| Target | G3 (final grade) |
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
60_StudentPerf/
├── code/
│   ├── run_experiments.py
├── data/
│   └── student.csv
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
# Verify student.csv exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py --direction 60_StudentPerf
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
