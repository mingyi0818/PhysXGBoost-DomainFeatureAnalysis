# Reproduction Guide — 54_NewsPopularity

This document describes how to reproduce every numerical result reported in the paper. All numbers in the paper trace back to files under `results/`.

## 1. Environment Configuration

### Hardware

- CPU: Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- RAM: 48 GB DDR5 RDIMM
- GPU: NVIDIA RTX 2000 Pro (16 GB) — not required for tree models
- OS: Windows 11 Professional

### Software

- Python 3.10+
- Dependencies listed below

### Install dependencies

```bash
pip install xgboost lightgbm catboost scikit-learn scipy matplotlib pandas numpy
```

### Verify installation

```bash
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
python -c "import catboost; print('CatBoost:', catboost.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

## 2. Data Acquisition

**Dataset**: Online News Popularity (UCI ML Repository)

**Source**: UCI Machine Learning Repository - Online News Popularity Dataset (https://archive.ics.uci.edu/dataset/332/online+news+popularity). 39,644 samples with 61 features.

**Local path**: `data/news_pop.csv`

**Size**: 18.52 MB

The CSV file is self-contained in the `data/` directory; no download step is required if the file is present.

If the data file is missing, download it from the source URL above and place it in the `data/` directory.

## 3. Running Experiments

All scripts must be run from the `code/` directory.

### 3.1 Main experiment

```bash
cd code
python run_experiments.py --direction 54_NewsPopularity
```

What it does:

1. Loads the dataset from `data/` directory.
2. Builds the Raw feature matrix and the Domain feature matrix (with domain-derived features).
3. For each of the 4 models (XGBoost, LightGBM, CatBoost, RandomForest) x 2 feature sets (Raw, Domain) x 5 seeds, performs an 80/20 train/test split and trains the model.
4. Records test-set R2 for every (model, feature_set, seed) triple.
5. Computes mean +/- SD across seeds and runs a one-sided Wilcoxon signed-rank test (Domain > Raw) per model.
6. Also computes 95% confidence intervals and Cohen's d effect sizes.

### Output files (all under `results/`)

| File | Contents |
|------|----------|
| `per_seed_results.json` | One entry per (feature_set, model, seed) with metric value |
| `comprehensive_results.json` | Extended metrics including all combinations |
| `summary.json` | Mean/SD metric per (feature_set, model) + Wilcoxon p-values |
| `additional_metrics.json` | 95% CI, Cohen's d, and supplementary statistics |


Expected runtime: 1-10 minutes on the reference hardware depending on dataset size.

### 3.2 Generate plots

Plots are automatically generated during the experiment run and saved to `plots/`.

## 4. Result Verification

After running the experiments, verify the results:

```bash
# Check that all expected result files exist
ls results/summary.json
ls results/comprehensive_results.json
ls results/per_seed_results.json
ls results/additional_metrics.json
```

### Key verification steps:

1. Open `results/summary.json` and verify the R2 values for Raw and Domain feature sets.
2. Check that `n_seeds` = 5 in the summary (or per_seed_results.json has 5 entries per model).
3. Verify Wilcoxon p-values are present in summary.json.
4. Cross-check that every number cited in the paper can be traced to a specific field in the result files.

### Traceability map (paper number -> result file)

| Paper location | Source file | Source field |
|----------------|-------------|--------------|
| Abstract / Table | summary.json | Raw.*.R2, Domain.*.R2 |
| Abstract / Table | summary.json | *.std (standard deviation) |
| Statistical test | summary.json | wilcoxon.*.p_value |
| Per-seed analysis | per_seed_results.json | Individual seed results |
| CI / Effect size | additional_metrics.json | CI and Cohen's d values |

## 5. Random Seeds

All experiments use 5 fixed random seeds to ensure reproducibility:

| Seed | Purpose |
|------|---------|
| 42 | Reproducibility baseline |
| 123 | Cross-validation seed |
| 456 | Robustness check |
| 789 | Statistical reliability |
| 2024 | Final verification |


These seeds control:

- Train/test split (`train_test_split` with `random_state=seed`)
- Model random states (`random_state` / `random_seed` parameters)
- Bootstrap sampling (RandomForest)
- Feature subsampling (tree-based models)


Using the same seeds on the same hardware with the same library versions will produce identical results. Minor numerical differences (+-0.001) may occur across different CPU architectures or library versions due to floating-point summation order.

## 6. Hyperparameters

All hyperparameters are defined in `run_experiments.py`:

| Parameter | Value |
|-----------|-------|
| Seeds | [42, 123, 456, 789, 2024] |
| Test size | 0.20 (80/20 train/test split) |
| n_estimators | 300 |
| max_depth (boosting) | 6 |
| max_depth (RF) | 12 |
| learning_rate | 0.1 |
| XGB tree_method | hist (default) |
| Models | XGBoost, LightGBM, CatBoost, RandomForest |
| Feature sets | Raw (original), Domain (with engineered features) |


## 7. Notes on Reproducibility

- The train/test split uses `sklearn.model_selection.train_test_split` with `random_state=seed`, so the exact same partition is produced for a given seed on any machine with the same scikit-learn version.
- XGBoost, LightGBM, CatBoost, and RandomForest all accept a `random_state`/`random_seed` parameter; passing the seed guarantees identical bootstrap/feature-subset draws.
- The Wilcoxon signed-rank test with n=5 seeds has a minimum p-value of 0.0625 (one-sided) for all-positive differences, which is a limitation of the small sample size.
- Minor numerical differences (+-0.001) may appear across CPU architectures or library versions due to floating-point summation order, but the reported 3-decimal values are stable.


## 8. Known Issues and Limitations

> WARNING: R2 scores are near zero or negative for all models (Raw and Domain identical), indicating the models fail to predict news popularity effectively with the current feature set. This is an honest negative result. Also note: two CSV files exist in data/ (news_pop.csv and news_popularity.csv) - the experiments use news_pop.csv.

---
For questions about reproduction, please refer to the code comments or open an issue in the repository.
