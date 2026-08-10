# Reproduction Guide: 44_Energy_Anomaly

This document describes how to reproduce every numerical result reported in the paper. All numbers in the paper trace back to files under `results/tables/`.

## 1. Environment Requirements

### Hardware

- CPU: Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- RAM: 48 GB DDR5 RDIMM (16 GB minimum)
- GPU: NVIDIA RTX 2000 Pro (16 GB) -- required for deep learning models
- OS: Windows 11 Professional

### Software

- Python 3.10+
- See `code/requirements.txt` for package versions

### Install dependencies

```bash
pip install -r code/requirements.txt
```

Or install manually:

```bash
pip install "torch>=2.0.0" "numpy>=1.22.0" "pandas>=1.4.0" "scikit-learn>=1.0.0" "scipy>=1.7.0" "matplotlib>=3.5.0"
```

### Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import scipy; print('SciPy:', scipy.__version__)"
```

## 2. Data Preparation

**Dataset**: SGCC (State Grid Corporation of China) Electricity Theft Dataset

**Source**: State Grid Corporation of China electricity consumption records. Raw data path configured as `D:\datasets\energy\SGCC` in `code/config.py`.

**Raw files expected**:
- `D:\datasets\energy\SGCC\after_preprocess_data.csv` -- consumption time series (42,372 consumers x 1,035 days)
- `D:\datasets\energy\SGCC\label.csv` -- anomaly labels (0=normal, 1=anomalous)

**Preprocessed files** (included in repository):
- `data/processed/X_s0.20.npy` -- 20% sampled consumers, shape (8474, 1035)
- `data/processed/y_s0.20.npy` -- corresponding labels, shape (8474,)

The preprocessing script samples 20% of consumers (SAMPLE_RATIO=0.2) to reduce memory usage while preserving the anomaly rate (~9.11%).

## 3. Step-by-Step Reproduction

### 3.1 Install dependencies

```bash
pip install -r code/requirements.txt
```

### 3.2 Place data files

Ensure preprocessed data is at `data/processed/X_s0.20.npy` and `data/processed/y_s0.20.npy`. If you have the raw SGCC dataset, run the data preprocessing first (see `code/data_loader.py` for the sampling logic).

### 3.3 Run main experiments

```bash
cd code
python run_experiments.py
```

What it does:

1. Loads the SGCC dataset from `data/processed/X_s0.20.npy` and `y_s0.20.npy`.
2. For each of 6 models (TCR-AD, OCSVM, IForest, AE, VAE, DAGMM) x 5 seeds:
   - Performs 70/15/15 train/validation/test split
   - Trains the model (TCR-AD uses mixed precision training)
   - Records test-set AUC-ROC, F1, Precision, Recall
3. Computes mean +/- std across 5 seeds
4. Runs paired t-test (TCR-AD vs each baseline) with 95% confidence intervals and Cohen's d
5. Runs ablation study (4 variants, seed 42)
6. Runs sensitivity analysis (embedding_dim: 32/64/128/256, contrastive_weight: 0.0/0.25/0.75/1.0, seed 42)
7. Computes complexity analysis (params, inference time)
8. Saves all results to `results/tables/`

**Expected runtime**: ~2-3 hours on the reference hardware (GPU required for TCR-AD).

### 3.4 Generate plots

```bash
cd code
python visualize.py
```

Plots are saved to `results/plots/`.

## 4. Results Files Description

### Result Files (all in `results/tables/`)

| File | Contents |
|------|----------|
| `summary.json` | Combined summary with all experiment results |
| `main_comparison_results.json` | Per-seed results: 6 models x 5 seeds (AUC, F1, Precision, Recall, train_time, params) |
| `main_comparison_summary.csv` | Summary statistics (mean, std) for all models |
| `ablation_results.json` | Component-level ablation (Full, w/o Time Encoder, w/o Freq Encoder, w/o Contrastive) |
| `ablation_results.csv` | CSV format of ablation results |
| `sensitivity_results.json` | Parameter sensitivity (embedding_dim, contrastive_weight) |
| `sensitivity_all.csv` | CSV format of sensitivity results |
| `complexity_analysis.json` | Model complexity (params, inference_time_ms) |
| `complexity_analysis.csv` | CSV format of complexity analysis |
| `statistical_tests.json` | Paired t-test results (TCR-AD vs each baseline, 5 seeds) |
| `statistical_tests.csv` | CSV format of statistical tests |

### Expected Results

**Main Comparison** (mean +/- std over 5 seeds):

| Model | AUC-ROC | F1-Score |
|-------|---------|----------|
| TCR-AD | 0.5038 +/- 0.0249 | 0.1485 +/- 0.0139 |
| IForest | 0.5256 +/- 0.0259 | 0.1583 +/- 0.0112 |
| OCSVM | 0.5130 +/- 0.0352 | 0.1585 +/- 0.0207 |
| DAGMM | 0.4986 +/- 0.0278 | 0.1470 +/- 0.0178 |
| AE | 0.4976 +/- 0.0283 | 0.1459 +/- 0.0169 |
| VAE | 0.4755 +/- 0.0217 | 0.1490 +/- 0.0111 |

**Note**: These are negative results. TCR-AD does not outperform the best baseline (IForest). All methods perform near random level (AUC ~ 0.5). This is honestly reported in the paper with analysis of causes.

## 5. Result Verification

After running experiments, verify:

1. All 12 result files exist in `results/tables/`.
2. Every number in the paper can be traced to a specific field in the result files:
   - Main comparison numbers: `summary.json` -> `main_comparison` array
   - Ablation numbers: `summary.json` -> `ablation` array
   - Sensitivity numbers: `summary.json` -> `sensitivity` array
   - Statistical test numbers: `summary.json` -> `statistical_tests` array
   - Complexity numbers: `summary.json` -> `complexity` array
3. The number of seeds matches (5 seeds: 42, 123, 456, 789, 2024).
4. Per-seed detailed results are in `main_comparison_results.json` -> `per_seed` array (30 entries = 6 models x 5 seeds).

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

Using the same seeds on the same hardware with the same library versions will produce identical results. Minor numerical differences (+/-0.001) may occur across different GPU architectures or library versions due to floating-point summation order.

## 7. Hyperparameters

All hyperparameters are defined in `code/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| SAMPLE_RATIO | 0.2 | 20% of consumers sampled |
| SEQ_LEN | 1035 | Original sequence length (days) |
| SUB_SEQ_LEN | 128 | Sub-sequence length for training |
| EMBED_DIM | 64 | Embedding dimension |
| TIME_ENCODER_HIDDEN | [64, 128] | Time encoder hidden layers |
| FREQ_ENCODER_HIDDEN | [64, 32] | Frequency encoder hidden layers |
| CONV_KERNEL_SIZES | [3, 5] | Convolution kernel sizes |
| N_HEADS | 4 | Number of attention heads |
| DROPOUT | 0.1 | Dropout rate |
| FREQ_N_BINS | 64 | Number of frequency bins |
| BATCH_SIZE | 1024 | Batch size |
| N_EPOCHS | 8 | Maximum training epochs |
| LEARNING_RATE | 1e-3 | Learning rate |
| WEIGHT_DECAY | 1e-5 | Weight decay |
| EARLY_STOP_PATIENCE | 5 | Early stopping patience |
| CONTRASTIVE_TEMPERATURE | 0.5 | NT-Xent temperature |
| CONTRASTIVE_WEIGHT | 0.5 | Contrastive loss weight |
| RECON_WEIGHT | 0.5 | Reconstruction loss weight |
| VAL_RATIO | 0.15 | Validation set ratio |
| TEST_RATIO | 0.15 | Test set ratio |
| Seeds | [42, 123, 456, 789, 2024] | Random seeds |

## 8. Notes on Reproducibility

- TCR-AD uses mixed precision training (`torch.cuda.amp`) for 2x speedup. This may introduce minor numerical differences compared to full precision training.
- The preprocessed `.npy` files ensure the same data is used across runs.
- GPU memory usage is approximately 4-6 GB, well within the 16 GB RTX 2000 Pro.
- PyTorch operations are deterministic when `torch.manual_seed()` is set, but CUDA operations may introduce small non-determinism. Use `torch.backends.cudnn.deterministic = True` for full determinism.
- The NT-Xent loss uses a float16-safe mask value (-1e4 instead of -1e9) to prevent overflow during mixed precision training.

## 9. Known Issues and Limitations

1. **Negative results**: TCR-AD does not outperform baselines on the SGCC dataset. All methods perform near random level (AUC ~ 0.5). This is honestly reported and analyzed in the paper.

2. **Sub-sequence sampling**: The original 1,035-day sequences are split into 128-day sub-sequences, which may lose long-term patterns. This is a known limitation discussed in the paper.

3. **Reconstruction-based scoring**: Anomaly scores based on reconstruction error may not capture all theft patterns, especially when anomalous patterns are similar to normal variations.

4. **Training convergence**: With only 8 epochs and early stopping (patience=5), the model may not fully converge. However, validation loss typically plateaus within 3-5 epochs.

5. **Data availability**: The SGCC dataset must be obtained separately. Preprocessed `.npy` files (20% sample) are included in the repository.

---

For questions about reproduction, please refer to the code comments or open an issue in the repository.
