"""Configuration for TCR-AD: Temporal Contrastive Reconstruction for Anomaly Detection."""

import os

# Paths
BASE_DIR = r'D:\ResearchPaperPrepare\44_Energy_Anomaly'
DATA_DIR = r'D:\datasets\energy\SGCC'
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'tables')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Dataset
DATA_FILE = os.path.join(DATA_DIR, 'after_preprocess_data.csv')
LABEL_FILE = os.path.join(DATA_DIR, 'label.csv')

# Data preprocessing
SEQ_LEN = 1035  # Original sequence length (days)
SUB_SEQ_LEN = 256  # Sub-sequence length for training
SAMPLE_RATIO = 0.3  # Ratio of customers to sample (for memory)
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NORMALIZE = True

# TCR-AD Model
EMBED_DIM = 128
TIME_ENCODER_HIDDEN = [128, 256, 128]
FREQ_ENCODER_HIDDEN = [128, 64]
CONV_KERNEL_SIZES = [3, 5, 7]  # Multi-scale kernels
N_HEADS = 4
DROPOUT = 0.1
USE_FREQ_ENCODER = True
USE_TIME_ENCODER = True
FREQ_N_BINS = 128  # Number of frequency bins to use

# Contrastive learning
CONTRASTIVE_TEMPERATURE = 0.5
CONTRASTIVE_WEIGHT = 0.5  # Weight for contrastive loss
RECON_WEIGHT = 0.5  # Weight for reconstruction loss
N_POSITIVE = 2  # Number of positive augmentations per sample

# Anomaly detection
ANOMALY_ALPHA = 0.5  # Weight for combining recon error and contrastive score
ANOMALY_THRESHOLD_PERCENTILE = 95  # Percentile for threshold

# Training
BATCH_SIZE = 256
N_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10

# Random seeds
RANDOM_SEEDS = [42, 123, 456, 789, 2024]

# Baselines
BASELINE_MODELS = [
    'OCSVM',       # One-Class SVM
    'IForest',     # Isolation Forest
    'AE',           # Autoencoder
    'VAE',          # Variational Autoencoder
    'DAGMM',        # Deep Autoencoding Gaussian Mixture Model
    'AnoGAN',       # Anomaly GAN
]

# Device
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'