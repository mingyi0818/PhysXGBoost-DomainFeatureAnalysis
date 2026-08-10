# Reproduction Guide — 44_Energy_Anomaly

This document describes how to reproduce every numerical result reported in the paper. All numbers in the paper trace back to files under `results/`.

## 1. Environment Configuration

### Hardware

- CPU: Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- RAM: 48 GB DDR5 RDIMM
- GPU: NVIDIA RTX 2000 Pro (16 GB) — not required for tree models; used for deep learning models
- OS: Windows 11 Professional

### Software

- Python 3.10+
- Dependencies listed below

### Install dependencies

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

## 2. Data Acquisition

**Dataset**: SGCC (State Grid Corporation of China) Electricity Theft Dataset

**Source**: State Grid Corporation of China (SGCC) electricity consumption records. Raw data path configured as D:\datasets\energy\SGCC in config.py.

**Local path**: `data/processed/X_s0.30.npy`

**Size**: 50.29 MB

The preprocessed data (`.npy` files) is included in `data/processed/`. The raw SGCC dataset is expected at an external path configured in `code/config.py` (`DATA_DIR`). To use raw data, download the SGCC dataset and update `DATA_DIR` in `config.py`.

## 3. Running Experiments

All scripts must be run from the `code/` directory.

### 3.1 Main experiment

```bash
cd code
python run_experiments.py
```

What it does:

1. Loads preprocessed data from `data/processed/` (or raw SGCC data if available).
2. Trains the TCR-AD model and baselines (OCSVM, IForest, AE, VAE, DAGMM) for each seed.
3. Evaluates anomaly detection performance (AUC, F1, Precision, Recall).
4. Saves results to `results/tables/` and plots to `results/plots/`.

Expected runtime: Several hours on the reference hardware (GPU recommended).

## 4. Result Verification

> **WARNING**: No results are currently available. Run the experiments first to generate result files.

After running experiments, verify:

1. All expected result files exist in `results/`.
2. Every number in the paper can be traced to a specific field in the result files.
3. The number of seeds matches (5 seeds: 42, 123, 456, 789, 2024).

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

- Neural network weight initialization
- Data shuffling and batching
- Dropout mask generation
- Train/validation/test split


Using the same seeds on the same hardware with the same library versions will produce identical results. Minor numerical differences (+-0.001) may occur across different CPU architectures or library versions due to floating-point summation order.

## 6. Hyperparameters

See `code/config.py` for all hyperparameters. Key settings:

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


## 7. Notes on Reproducibility

- PyTorch operations are deterministic when `torch.manual_seed()` is set, but CUDA operations may introduce small non-determinism. Use `torch.backends.cudnn.deterministic = True` for full determinism.
- The preprocessed `.npy` files ensure the same data is used across runs.
- GPU memory requirements are modest (~4 GB) and fit within the 16 GB RTX 2000 Pro.


## 8. Known Issues and Limitations

> WARNING: The results/ directory does not exist yet - experiments have not been fully run or results were not saved. The checkpoints/tcrad_best.pth exists but no result files are available for traceability. The raw SGCC dataset is expected at an external path (D:\datasets\energy\SGCC) which may not be available on all machines. Only preprocessed .npy files (30% sample) are included in data/processed/.

---
For questions about reproduction, please refer to the code comments or open an issue in the repository.
