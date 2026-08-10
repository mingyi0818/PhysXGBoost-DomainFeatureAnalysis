# TCR-AD: Temporal Contrastive Reconstruction for Energy Anomaly Detection

> A deep learning approach combining temporal contrastive learning with reconstruction-based anomaly detection for smart grid energy consumption data. The model (TCR-AD) uses multi-scale convolutional encoders in both time and frequency domains to detect anomalous electricity usage patterns.

**Task**: Anomaly Detection (Binary Classification) | **Target**: Anomaly label (0=normal, 1=anomalous) | **Primary Metric**: AUC, F1, Precision, Recall

## Data Integrity Notice

> WARNING: The results/ directory does not exist yet - experiments have not been fully run or results were not saved. The checkpoints/tcrad_best.pth exists but no result files are available for traceability. The raw SGCC dataset is expected at an external path (D:\datasets\energy\SGCC) which may not be available on all machines. Only preprocessed .npy files (30% sample) are included in data/processed/.

## Dataset

| Item | Detail |
|------|--------|
| Name | SGCC (State Grid Corporation of China) Electricity Theft Dataset |
| File | `data/processed/X_s0.30.npy, data/processed/y_s0.30.npy (preprocessed; raw data expected at external path)` |
| Size | 50.29 MB |
| Source | State Grid Corporation of China (SGCC) electricity consumption records. Raw data path configured as D:\datasets\energy\SGCC in config.py. |
| Task | Anomaly Detection (Binary Classification) |
| Target | Anomaly label (0=normal, 1=anomalous) |
| Metric | AUC, F1, Precision, Recall |

## Environment Requirements

- Python 3.10+
- OS: Windows 11 Professional (tested)
- CPU: Intel Xeon W7-2595X (24 cores, 2.5-4.8 GHz)
- RAM: 48 GB DDR5 RDIMM
- GPU: NVIDIA RTX 2000 Pro (16 GB) — not required for tree models

### Python Dependencies

```
torch>=2.0.0
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
44_Energy_Anomaly/
├── code/
│   ├── config.py
│   ├── data_loader.py
│   ├── models.py
│   ├── run_experiments.py
│   ├── train.py
│   ├── visualize.py
│   ├── requirements.txt
├── data/
│   ├── X_s0.30.npy
│   ├── y_s0.30.npy (preprocessed; raw data expected at external path)
├── results/
│   └── (not available)
├── plots/
│   └── (not available)
├── paper/
│   └── paper_draft.md
├── reference/
│   └── REFERENCE_MATERIALS.md
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r code/requirements.txt
```

### 2. Verify data is present

```bash
# Verify X_s0.30.npy exists in data/
# Verify y_s0.30.npy (preprocessed; raw data expected at external path) exists in data/
```

### 3. Run experiments

```bash
cd code
python run_experiments.py
```

### 4. Check results

```bash
# WARNING: No results directory found. Run experiments first.
```

## Result Files

> No result files are currently available. Run the experiments to generate results.

## Reproduction

For detailed reproduction instructions, see [reproduce.md](reproduce.md).

## Citation

If you use this code or data, please cite the original dataset source and this repository.

## License

This project is for academic research purposes. Dataset licenses follow their respective sources.
