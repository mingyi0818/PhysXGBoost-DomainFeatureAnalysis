# PhysXGBoost: Domain Feature Augmentation for Tree-based Models

A comprehensive study of domain-specific feature engineering for tree-based ensemble models (XGBoost, LightGBM, CatBoost, Random Forest) across 19 tabular datasets spanning diverse domains.

## Overview

This repository contains the source code, experimental results, and paper drafts for 19 research directions, each investigating whether domain-specific feature engineering improves the performance of tree-based models on a specific tabular dataset.

## Research Directions

| # | Direction | Dataset | Task | Key Finding |
|---|-----------|---------|------|-------------|
| 1 | 44_Energy_Anomaly | SGCC | Classification | TCR-AD anomaly detection |
| 2 | 46_FlightDelay_PhysXGBoost | Bank Marketing | Classification | Domain features show minimal gain |
| 3 | 47_OnlineShoppers | Online Shoppers | Classification | AUC 0.923-0.930, minimal domain feature gain |
| 4 | 48_CreditDefault | Credit Card Default | Classification | Zero gain - information saturation |
| 5 | 49_Superconductor | Superconductivity | Regression | Domain features improve R² |
| 6 | 50_BuildingEnergy | Building Energy | Regression | Energy prediction with domain features |
| 7 | 51_GasTurbine | Gas Turbine | Regression | Thermodynamic domain features |
| 8 | 52_CCPP | Combined Cycle Power Plant | Regression | Thermodynamic feature analysis |
| 9 | 53_BikeSharing | Bike Sharing | Regression | Transportation domain features |
| 10 | 54_NewsPopularity | Online News | Regression | Negative result - R²≈0 |
| 11 | 55_CalHousing | California Housing | Regression | Real estate domain features |
| 12 | 56_PowerConsumption | Household Power | Regression | Data leakage investigation |
| 13 | 58_CDNOW | CDNOW | Classification | Customer lifetime value |
| 14 | 59_NYCProperty | NYC Property Sales | Regression | Real estate domain features |
| 15 | 60_StudentPerf | Student Performance | Regression | Educational domain features |
| 16 | 61_DryBean | Dry Bean | Classification | Domain features significantly improve AUC |
| 17 | 63_HotelBooking | Hotel Booking | Classification | Hotel cancellation prediction |
| 18 | 64_FlightDelay | Flight Delay | Classification | Data leakage investigation |
| 19 | 65_HR | IBM HR Analytics | Classification | Employee attrition prediction |

## Methodology

All directions follow the PhysXGBoost framework:
1. **Raw Features**: Original dataset features
2. **Domain Feature Engineering**: Domain-specific derived features
3. **Model Comparison**: XGBoost, LightGBM, CatBoost, Random Forest
4. **Statistical Validation**: 5 random seeds (42, 123, 456, 789, 2024), Wilcoxon signed-rank test, 95% CI, Cohen's d
5. **Analysis**: Ablation, sensitivity, robustness, SHAP feature importance

## Repository Structure

```
.
├── 44_Energy_Anomaly/
│   ├── README.md          # Direction-specific guide
│   ├── reproduce.md       # Reproduction instructions
│   ├── code/              # Source code
│   ├── data/              # Dataset
│   ├── results/           # Experimental results (JSON)
│   ├── plots/             # High-resolution figures (PNG, 300 DPI)
│   ├── paper/             # Paper draft
│   └── reference/         # Reference materials
├── 46_FlightDelay_PhysXGBoost/
│   └── ...
├── ... (17 more directions)
├── run_experiments.py     # Universal experiment runner
├── compute_additional_metrics.py  # Additional metrics computation
├── generate_figures.py    # Figure generation script
├── replace_placeholders.py  # PLACEHOLDER replacement script
└── requirements.txt       # Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments for a specific direction
python run_experiments.py --direction 51_GasTurbine

# Compute additional metrics
python compute_additional_metrics.py

# Generate figures
python generate_figures.py
```

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list

## Data Sources

All datasets are publicly available from UCI Machine Learning Repository or Kaggle. See each direction's `reproduce.md` for specific download instructions.

## Academic Integrity

- All experimental results are stored in `results/` directories as JSON files
- Every number in the paper drafts can be traced to a specific results file
- Negative results (54_NewsPopularity: R²≈0) and data leakage investigations (56_PowerConsumption, 64_FlightDelay) are reported honestly
- No data has been fabricated or manipulated

## License

This project is for academic research purposes.

## Citation

If you use this code, please cite the corresponding paper.
