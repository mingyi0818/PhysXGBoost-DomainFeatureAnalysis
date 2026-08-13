# PhysXGBoost: Domain Feature Augmentation for Tree-Based Models

A collection of 19 research papers investigating domain-specific feature engineering for tabular data prediction across diverse domains. Each study systematically evaluates whether physics- and domain-knowledge-derived features improve the performance of tree-based ensemble models (XGBoost, LightGBM, CatBoost, Random Forest) on a specific real-world tabular dataset.

---

## Overview

The central question of this repository is: **Does domain-specific feature engineering improve tree-based model performance on tabular data?** To answer it, 19 independent experiments were conducted across diverse domains (energy, finance, transportation, real estate, education, agriculture, hospitality, e-commerce, and more). Every experiment follows a unified protocol so that results are directly comparable.

Key design principles:

- **Honest reporting.** Negative results (e.g., no improvement, information saturation) and data-leakage investigations are reported transparently alongside positive findings.
- **Full reproducibility.** All experimental results are stored as JSON/CSV files under each direction's `results/` directory. Every numerical value in every paper draft traces back to a specific result file.
- **Statistical rigor.** Each direction uses 5 fixed random seeds, Wilcoxon signed-rank tests, 95% confidence intervals, and Cohen's d effect sizes.

---

## Research Directions

The repository contains 19 research directions, each in its own numbered directory:

| # | Directory | Dataset | Task | Domain Feature Focus | Key Finding |
|---|-----------|---------|------|----------------------|-------------|
| 1 | `44_Energy_Anomaly` | SGCC Electricity Theft | Classification (Anomaly) | Temporal contrastive reconstruction features | TCR-AD deep model for electricity-theft detection |
| 2 | `46_FlightDelay_PhysXGBoost` | Bank Marketing | Classification | Financial-behavior domain features | Domain features show minimal gain |
| 3 | `47_OnlineShoppers` | Online Shoppers Intention | Classification | Browsing-behavior domain features | minimal domain feature gain |
| 4 | `48_CreditDefault` | Credit Card Default | Classification | Credit-risk domain features | Zero gain - information saturation |
| 5 | `49_Superconductor` | Superconductivity | Regression | Material-science domain features | Domain features improve R-squared |
| 6 | `50_BuildingEnergy` | Building Energy Efficiency | Regression | Thermodynamic domain features | Energy prediction with domain features |
| 7 | `51_GasTurbine` | Gas Turbine CO/NOx Emission | Regression | Thermodynamic domain features (pressure ratios, air density, combustion efficiency) | domain features improve R-squared |
| 8 | `52_CCPP` | Combined Cycle Power Plant | Regression | Thermodynamic domain features | Thermodynamic feature analysis |
| 9 | `53_BikeSharing` | Bike Sharing | Regression | Transportation domain features | Temporal and weather-derived features |
| 10 | `54_NewsPopularity` | Online News Popularity | Regression | Content and social domain features | Negative result - no measurable improvement |
| 11 | `55_CalHousing` | California Housing | Regression | Real-estate domain features | Spatial and economic derived features |
| 12 | `56_PowerConsumption` | Household Power Consumption | Regression | Time-series domain features | Data-leakage investigation |
| 13 | `58_CDNOW` | CDNOW | Classification | Customer-lifetime-value features | RFM and purchase-pattern features |
| 14 | `59_NYCProperty` | NYC Property Sales | Regression | Real-estate domain features | Spatial and temporal derived features |
| 15 | `60_StudentPerf` | Student Performance | Regression | Educational domain features | Behavioral and demographic features |
| 16 | `61_DryBean` | Dry Bean | Classification | Morphological domain features | Domain features significantly improve AUC |
| 17 | `63_HotelBooking` | Hotel Booking Demand | Classification | Hospitality domain features | Booking-pattern and seasonal features |
| 18 | `64_FlightDelay` | Flight Delay | Classification | Aviation domain features | Data-leakage investigation |
| 19 | `65_HR` | IBM HR Analytics | Classification | Organizational domain features | Employee-attrition prediction |

---

## Overall Framework

All 19 directions follow the **PhysXGBoost** framework, which consists of five stages:

### 1. Raw Features
The original dataset features are used as the baseline input matrix.

### 2. Domain Feature Engineering
Domain-specific features are derived from physical laws, domain knowledge, or expert rules relevant to each dataset. For example:
- **Thermodynamic features** (gas turbine, power plant): pressure ratios, air density, combustion efficiency
- **Material-science features** (superconductor): elemental properties, critical temperature proxies
- **Real-estate features** (housing): spatial lag, price-per-area, neighborhood statistics
- **Behavioral features** (online shoppers, students): session ratios, engagement scores

### 3. Model Comparison
Four tree-based ensemble models are compared on both Raw and Domain feature sets:
- XGBoost
- LightGBM
- CatBoost
- Random Forest

### 4. Statistical Validation
- 5 fixed random seeds: [42, 123, 456, 789, 2024]
- 80/20 train/test split for each seed
- Wilcoxon signed-rank test (one-sided: Domain > Raw)
- 95% confidence intervals
- Cohen's d effect size
- All metrics computed on the **test set** (validation metrics used only for hyperparameter selection)

### 5. Analysis
- Ablation studies (component-level and feature-set-level)
- Parameter sensitivity analysis with elasticity coefficients
- Robustness analysis (noise injection where applicable)
- SHAP feature importance
- Computational complexity and runtime analysis

---

## Repository Structure

```
.
├── README.md                          # This file (top-level guide)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── run_experiments.py                 # Universal experiment runner
├── compute_additional_metrics.py      # Additional metrics computation
├── generate_figures.py                # Figure generation script
├── replace_placeholders.py            # PLACEHOLDER replacement utility
│
├── 44_Energy_Anomaly/                 # Direction 1: Electricity theft detection
├── 46_FlightDelay_PhysXGBoost/        # Direction 2: Bank marketing
├── 47_OnlineShoppers/                 # Direction 3: Online shoppers
├── 48_CreditDefault/                  # Direction 4: Credit default
├── 49_Superconductor/                 # Direction 5: Superconductivity
├── 50_BuildingEnergy/                 # Direction 6: Building energy
├── 51_GasTurbine/                     # Direction 7: Gas turbine
├── 52_CCPP/                           # Direction 8: Power plant
├── 53_BikeSharing/                    # Direction 9: Bike sharing
├── 54_NewsPopularity/                 # Direction 10: News popularity
├── 55_CalHousing/                     # Direction 11: California housing
├── 56_PowerConsumption/               # Direction 12: Power consumption
├── 58_CDNOW/                          # Direction 13: CDNOW
├── 59_NYCProperty/                    # Direction 14: NYC property
├── 60_StudentPerf/                    # Direction 15: Student performance
├── 61_DryBean/                        # Direction 16: Dry bean
├── 63_HotelBooking/                   # Direction 17: Hotel booking
├── 64_FlightDelay/                    # Direction 18: Flight delay
├── 65_HR/                             # Direction 19: HR analytics
│
└── (Each direction contains:)
    ├── README.md          # Direction-specific guide
    ├── reproduce.md       # Step-by-step reproduction instructions
    ├── SOTA_击破分析.md    # SOTA comparison analysis
    ├── code/              # Source code (run_experiments.py, etc.)
    ├── data/              # Dataset (CSV)
    ├── results/           # Experimental results (JSON/CSV)
    ├── plots/             # High-resolution figures (PNG, 300 DPI)
    ├── paper/             # Paper draft (Markdown)
    └── reference/         # Reference materials
```

---

## Requirements

### Hardware (Tested Configuration)

- **OS:** Windows 11 Professional
- **CPU:** Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- **RAM:** 48 GB DDR5 RDIMM (16 GB minimum)
- **GPU:** NVIDIA RTX 2000 Pro (16 GB) -- not required for tree models; needed only for deep-learning baseline in Direction 1

### Software

- Python 3.10+
- See `requirements.txt` for the full dependency list

### Python Dependencies

```
python>=3.10
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.43.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Reproduce

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd PhysXGBoost

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run experiments for a specific direction
python run_experiments.py --direction 51_GasTurbine
```

### Step-by-Step Reproduction for Any Direction

Each direction directory (e.g., `51_GasTurbine/`) contains everything needed to reproduce its results:

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the dataset is present**

   Place the dataset CSV in the direction's `data/` directory. For large datasets excluded by `.gitignore`, download instructions are provided in each direction's `reproduce.md`.

3. **Run the experiments**

   ```bash
   cd <direction_directory>/code
   python run_experiments.py
   ```

   Or from the repository root:

   ```bash
   python run_experiments.py --direction <direction_name>
   ```

   What this does:
   - Loads the dataset from `data/`
   - Builds the Raw feature matrix and the Domain feature matrix
   - Trains 4 models x 2 feature sets x 5 seeds = 40 configurations
   - Records test-set metrics (R-squared for regression; AUC/F1/Accuracy for classification)
   - Computes mean +/- SD across seeds
   - Runs Wilcoxon signed-rank test, 95% CI, and Cohen's d
   - Generates figures (300 DPI PNG) in `plots/`
   - Saves all results as JSON in `results/`

   Expected runtime: 5-30 minutes per direction (varies by dataset size).

4. **Verify results**

   After running, check that the following result files exist in `results/`:

   | File | Contents |
   |------|----------|
   | `summary.json` | Main results: mean/std metrics for Raw vs Domain, Wilcoxon p-values |
   | `comprehensive_results.json` | Ablation and sensitivity analysis |
   | `per_seed_results.json` | Per-seed detailed results for each (model, feature_set, seed) |
   | `additional_metrics.json` | Additional metrics (Accuracy, F1, RMSE, MAE, 95% CI, Cohen's d) |

5. **Cross-check with the paper**

   Every numerical value in the paper draft (`paper/paper_draft.md`) should trace to a specific field in a result file. See the direction's `reproduce.md` for the traceability map.

### Random Seeds

All experiments use 5 fixed random seeds for reproducibility:

| Seed | Purpose |
|------|---------|
| 42 | Reproducibility baseline |
| 123 | Cross-validation seed |
| 456 | Robustness check |
| 789 | Statistical reliability |
| 2024 | Final verification |

These seeds control train/test splitting, model random states, bootstrap sampling, and feature subsampling. Using the same seeds on the same hardware with the same library versions produces identical results. Minor numerical differences (+/-0.001) may occur across different CPU architectures or library versions due to floating-point summation order.

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Seeds | [42, 123, 456, 789, 2024] |
| Test size | 0.20 (80/20 train/test split) |
| n_estimators | 300 |
| max_depth (boosting models) | 6 |
| max_depth (Random Forest) | 12 |
| learning_rate | 0.1 |
| XGB tree_method | hist |
| Models | XGBoost, LightGBM, CatBoost, Random Forest |
| Feature sets | Raw (original), Domain (engineered) |

---

## Data Sources

All datasets are publicly available from the UCI Machine Learning Repository or Kaggle. Large data files (exceeding repository size limits) are excluded by `.gitignore`; download instructions are provided in each direction's `reproduce.md` file.

---

## Academic Integrity

- All experimental results are stored in `results/` directories as JSON/CSV files.
- Every number in the paper drafts can be traced to a specific result file field.
- Negative results (e.g., `54_NewsPopularity`) and data-leakage investigations (e.g., `56_PowerConsumption`, `64_FlightDelay`) are reported honestly.
- No data has been fabricated or manipulated.
- Reviewers can reproduce all experiments by following each direction's `reproduce.md`.

---

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.

## Citation

If you use this code, please cite the corresponding paper.
