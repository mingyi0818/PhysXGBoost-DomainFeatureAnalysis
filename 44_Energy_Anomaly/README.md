# TCR-AD: Temporal Contrastive Reconstruction for Anomaly Detection in Electricity Theft Detection

> A deep learning approach combining temporal contrastive learning with reconstruction-based anomaly detection for smart grid energy consumption data. The model (TCR-AD) uses multi-scale convolutional encoders in both time and frequency domains to detect anomalous electricity usage patterns.

**Task**: Anomaly Detection (Binary Classification) | **Target**: Anomaly label (0=normal, 1=anomalous) | **Primary Metric**: AUC-ROC, F1-Score

## Dataset

| Item | Detail |
|------|--------|
| Name | SGCC (State Grid Corporation of China) Electricity Theft Dataset |
| Raw Source | State Grid Corporation of China, 42,372 consumers with 1,035 days of daily consumption records |
| Sampled Data | 20% of consumers (8,474 sampled), ~9.11% anomaly rate |
| Files | `data/processed/X_s0.20.npy`, `data/processed/y_s0.20.npy` |
| Raw Data Path | `D:\datasets\energy\SGCC` (configured in `code/config.py`) |

## Method

TCR-AD combines temporal contrastive learning with reconstruction-based anomaly detection, using:
- **Time-domain encoder**: Multi-scale 1D-CNN with residual connections and multi-head self-attention
- **Frequency-domain encoder**: FFT-based spectral feature extraction
- **Adaptive gated fusion**: Learns to balance time and frequency representations
- **Joint loss**: NT-Xent contrastive loss + MSE reconstruction loss + BCE classification loss

Baselines: OCSVM, Isolation Forest (IForest), Autoencoder (AE), Variational Autoencoder (VAE), DAGMM.

## Directory Structure

```
44_Energy_Anomaly/
├── data/              # Dataset files (preprocessed .npy)
│   └── processed/     # X_s0.20.npy, y_s0.20.npy
├── code/              # Source code
│   ├── config.py      # All hyperparameters
│   ├── data_loader.py # Data loading and augmentation
│   ├── models.py      # TCR-AD and baseline model implementations
│   ├── train.py       # Training functions (mixed precision)
│   ├── run_experiments.py # Main experiment pipeline
│   ├── visualize.py   # Plot generation
│   └── requirements.txt
├── results/           # Experimental results
│   └── tables/        # All JSON/CSV result files
├── paper/             # Paper draft (paper_draft.md)
└── checkpoints/       # Model checkpoints
```

## Key Results

**Honest reporting of negative results**: TCR-AD does not outperform the best baseline. All methods perform near random level (AUC ~ 0.5).

| Model | AUC-ROC (mean +/- std) | F1-Score (mean +/- std) | Params | Inference (ms) |
|-------|----------------------|------------------------|--------|-----------------|
| TCR-AD | 0.5038 +/- 0.0249 | 0.1485 +/- 0.0139 | 412,067 | 1.89 |
| IForest | **0.5256 +/- 0.0259** | **0.1583 +/- 0.0112** | 0 | < 0.01 |
| OCSVM | 0.5130 +/- 0.0352 | 0.1585 +/- 0.0207 | 0 | N/A |
| DAGMM | 0.4986 +/- 0.0278 | 0.1470 +/- 0.0178 | 149,119 | 0.18 |
| AE | 0.4976 +/- 0.0283 | 0.1459 +/- 0.0169 | 148,416 | 0.11 |
| VAE | 0.4755 +/- 0.0217 | 0.1490 +/- 0.0111 | 156,672 | 0.14 |

Statistical tests (paired t-test, TCR-AD vs baselines): All p > 0.05, no significant difference.

### Ablation Study (Seed 42)

| Variant | AUC-ROC | F1-Score |
|---------|---------|----------|
| Full TCR-AD | 0.5219 | 0.1617 |
| w/o Time Encoder | 0.5522 | 0.1818 |
| w/o Freq Encoder | 0.5231 | 0.1657 |
| w/o Contrastive | 0.5241 | 0.1594 |

### Sensitivity Analysis (Seed 42)

| Parameter | Value | AUC-ROC |
|-----------|-------|---------|
| embedding_dim | 32 | 0.5211 |
| embedding_dim | 64 | 0.5199 |
| embedding_dim | 128 | 0.5203 |
| embedding_dim | 256 | 0.5212 |
| contrastive_weight | 0.0 | 0.5177 |
| contrastive_weight | 0.25 | 0.5217 |
| contrastive_weight | 0.75 | 0.5166 |
| contrastive_weight | 1.0 | 0.5174 |

## Result Files

All results are stored in `results/tables/`:

| File | Contents |
|------|----------|
| `summary.json` | Combined summary: main comparison, ablation, sensitivity, complexity, statistical tests |
| `main_comparison_results.json` | Per-seed detailed results for all 6 models x 5 seeds |
| `main_comparison_summary.csv` | Summary statistics (mean, std) for all models |
| `ablation_results.json` | Component-level ablation study results |
| `sensitivity_results.json` | Parameter sensitivity analysis results |
| `complexity_analysis.json` | Model complexity (params, inference time) |
| `statistical_tests.json` | Paired t-test results (TCR-AD vs each baseline) |
| `main_comparison.csv` | CSV format of main comparison |
| `ablation_results.csv` | CSV format of ablation results |
| `sensitivity_all.csv` | CSV format of sensitivity results |
| `complexity_analysis.csv` | CSV format of complexity analysis |
| `statistical_tests.csv` | CSV format of statistical tests |

## Environment Requirements

- Python 3.10+
- OS: Windows 11 Professional (tested)
- CPU: Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- RAM: 48 GB DDR5 RDIMM
- GPU: NVIDIA RTX 2000 Pro (16 GB) -- required for deep learning models

### Python Dependencies

```
torch>=2.0.0, numpy>=1.22.0, pandas>=1.4.0, scikit-learn>=1.0.0, scipy>=1.7.0, matplotlib>=3.5.0
```

## How to Reproduce

See [reproduce.md](reproduce.md) for detailed step-by-step instructions.

```bash
cd code
python run_experiments.py
```

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
