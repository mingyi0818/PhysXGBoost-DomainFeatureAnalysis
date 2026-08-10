# TCR-AD: Temporal Contrastive Reconstruction for Anomaly Detection in Electricity Theft Detection

> A deep learning approach combining temporal contrastive learning with reconstruction-based anomaly detection for smart grid energy consumption data. The model (TCR-AD) uses multi-scale convolutional encoders in both time and frequency domains to detect anomalous electricity usage patterns.

**Task**: Anomaly Detection (Binary Classification) | **Target**: Anomaly label (0=normal, 1=anomalous) | **Primary Metric**: AUC, F1, Precision, Recall

## Dataset

| Item | Detail |
|------|--------|
| Name | SGCC (State Grid Corporation of China) Electricity Theft Dataset |
| File | `data/processed/X_s0.30.npy, data/processed/y_s0.30.npy` |
| Size | 50.29 MB |
| Source | State Grid Corporation of China (SGCC) electricity consumption records. Raw data path configured as D:\datasets\energy\SGCC in config.py. |
| Task | Anomaly Detection (Binary Classification) |
| Target | Anomaly label (0=normal, 1=anomalous) |
| Metric | AUC, F1, Precision, Recall |

## Method

Deep learning model (TCR-AD) combining temporal contrastive learning with reconstruction-based anomaly detection, using multi-scale convolutional encoders in both time and frequency domains. Baselines include OCSVM, Isolation Forest, Autoencoder, VAE, and DAGMM.

## Directory Structure

```
44_Energy_Anomaly/
├── data/          # Dataset files
├── code/          # Source code
├── results/       # Experimental results (JSON/CSV)
├── paper/         # Paper draft
└── plots/         # Figures (PNG, 300 DPI)
```

## Key Results

> **WARNING**: No results are currently available. The `results/` directory does not exist. Run the experiments first to generate result files.

## Result Files

> No result files are currently available. Run the experiments to generate results.

## Environment Requirements

- Python 3.10+
- OS: Windows 11 Professional (tested)
- CPU: Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- RAM: 48 GB DDR5 RDIMM
- GPU: NVIDIA RTX 2000 Pro (16 GB) -- required for deep learning models

### Python Dependencies

```
torch>=2.0.0, xgboost>=2.0.0, lightgbm>=4.0.0, catboost>=1.2.0, scikit-learn>=1.3.0, pandas>=2.0.0, numpy>=1.24.0, scipy>=1.11.0, matplotlib>=3.7.0
```

## How to Reproduce

See [reproduce.md](reproduce.md) for detailed step-by-step instructions.

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
