# Reproduction Guide: 44_Energy_Anomaly

This document describes how to reproduce every numerical result reported in the paper. All numbers in the paper trace back to files under `results/`.

## 1. Environment Requirements

### Hardware

- CPU: Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- RAM: 48 GB DDR5 RDIMM (16 GB minimum)
- GPU: NVIDIA RTX 2000 Pro (16 GB) -- required for deep learning models
- OS: Windows 11 Professional

### Software

- Python 3.10+
- See `code/requirements.txt` for package versions (if available)

### Install dependencies

```bash
pip install "torch>=2.0.0" "xgboost>=2.0.0" "lightgbm>=4.0.0" "catboost>=1.2.0" "scikit-learn>=1.3.0" "scipy>=1.11.0" "matplotlib>=3.7.0" "pandas>=2.0.0" "numpy>=1.24.0"
```

Or, if `requirements.txt` is available:

```bash
pip install -r code/requirements.txt
```

### Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
python -c "import catboost; print('CatBoost:', catboost.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)" 
```

## 2. Data Preparation

**Dataset**: SGCC (State Grid Corporation of China) Electricity Theft Dataset

**Source**: State Grid Corporation of China (SGCC) electricity consumption records. Raw data path configured as D:\datasets\energy\SGCC in config.py.

**Local path**: `data/processed/X_s0.30.npy, data/processed/y_s0.30.npy`

**Size**: 50.29 MB

Preprocessed .npy files (30% sample) included. Raw SGCC dataset expected at external path D:\datasets\energy\SGCC.

> **WARNING**: Data may not be available. Please ensure the data file is present before running experiments.

## 3. Step-by-Step Reproduction

### 3.1 Install dependencies

```bash
pip install -r code/requirements.txt
```

If no `requirements.txt` is available, install packages manually:

```bash
pip install "torch>=2.0.0" "xgboost>=2.0.0" "lightgbm>=4.0.0" "catboost>=1.2.0" "scikit-learn>=1.3.0" "scipy>=1.11.0" "matplotlib>=3.7.0" "pandas>=2.0.0" "numpy>=1.24.0"
```

### 3.2 Place data file in `data/` directory

Ensure the dataset file is present at `data/processed/X_s0.30.npy, data/processed/y_s0.30.npy`.

### 3.3 Run experiments

```bash
cd code
python run_experiments.py
```



```bash
python run_experiments.py --direction 44_Energy_Anomaly
```

What it does:

1. Loads the dataset from `data/` directory.
2. Prepares the deep learning model and baselines.
3. For each of the models x 5 seeds, performs an 80/20 train/test split and trains the model.
4. Records test-set AUC, F1, Precision, Recall for every configuration.
5. Computes mean +/- SD across seeds and runs a one-sided Wilcoxon signed-rank test (Domain > Raw) per model (for tree-based models).
6. Also computes 95% confidence intervals and Cohen's d effect sizes (if available).

Expected runtime: Several hours on the reference hardware (GPU recommended)

### 3.4 Generate plots

```bash
cd code
python visualize.py
```

Plots are saved to `plots/` (or `results/plots/`).

## 4. Results Files Description

### Result Files

> No result files are currently available. Run the experiments to generate result files.

Expected output files after running:
| File | Contents |
|------|----------|
| `results/summary.json` | Main experimental results (Raw vs Domain features, 4 models x 5 seeds) |
| `results/comprehensive_results.json` | Ablation and sensitivity analysis (if available) |
| `results/additional_metrics.json` | Additional metrics (Accuracy, F1, RMSE, MAE, etc.) |
| `results/per_seed_results.json` | Per-seed detailed results |

## 5. Result Verification

> **WARNING**: No results are currently available. Run the experiments first to generate result files.

After running experiments, verify:

1. All expected result files exist in `results/`.
2. Every number in the paper can be traced to a specific field in the result files.
3. The number of seeds matches (5 seeds: 42, 123, 456, 789, 2024).

## 6. Random Seeds

All experiments use 5 fixed random seeds to ensure reproducibility:

| Seed | Purpose |
|------|---------|
| 42 | Reproducibility baseline |
| 123 | Cross-validation seed |
| 456 | Robustness check |
| 789 | Statistical reliability |
| 2024 | Final verification |

These seeds control:

- Neural network weight initialization
- Data shuffling and batching
- Dropout mask generation
- Train/validation/test split

Using the same seeds on the same hardware with the same library versions will produce identical results. Minor numerical differences (+/-0.001) may occur across different CPU architectures or library versions due to floating-point summation order.

## 7. Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 256 |
| Epochs | 50 |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Early stop patience | 10 |
| Embed dim | 128 |
| Conv kernel sizes | [3, 5, 7] |
| Contrastive temperature | 0.5 |
| Seeds | [42, 123, 456, 789, 2024] |

## 8. Notes on Reproducibility

- PyTorch operations are deterministic when `torch.manual_seed()` is set, but CUDA operations may introduce small non-determinism. Use `torch.backends.cudnn.deterministic = True` for full determinism.
- The preprocessed `.npy` files ensure the same data is used across runs.
- GPU memory requirements are modest (~4 GB) and fit within the 16 GB RTX 2000 Pro.

## 9. Known Issues and Limitations

> WARNING: The results/ directory does not exist yet - experiments have not been fully run or results were not saved. Preprocessed .npy files (30% sample) included. Raw SGCC dataset expected at external path D:\datasets\energy\SGCC.

---

For questions about reproduction, please refer to the code comments or open an issue in the repository.
