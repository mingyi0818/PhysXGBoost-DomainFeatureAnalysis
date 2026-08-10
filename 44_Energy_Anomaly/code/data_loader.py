"""Data loader for SGCC electricity theft detection dataset."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


def load_sgcc_data(sample_ratio=SAMPLE_RATIO, random_state=42):
    """Load SGCC electricity theft detection dataset.
    
    Returns:
        X: numpy array of shape (n_samples, SEQ_LEN)
        y: numpy array of shape (n_samples,) - 0=normal, 1=anomaly
    """
    print(f"Loading SGCC data from {DATA_FILE}...")
    
    # Check if processed data exists (with sample ratio in filename)
    proc_x_path = os.path.join(PROCESSED_DIR, f'X_s{sample_ratio:.2f}.npy')
    proc_y_path = os.path.join(PROCESSED_DIR, f'y_s{sample_ratio:.2f}.npy')
    
    if os.path.exists(proc_x_path) and os.path.exists(proc_y_path):
        print("Loading cached processed data...")
        X = np.load(proc_x_path)
        y = np.load(proc_y_path)
        print(f"Loaded: X {X.shape}, y {y.shape}, anomalies: {y.sum()} ({y.mean()*100:.2f}%)")
        return X, y
    
    # Load raw data - skip header row, skip first column (index)
    df = pd.read_csv(DATA_FILE, header=None, skiprows=1, low_memory=False)
    labels = pd.read_csv(LABEL_FILE, header=0)
    
    print(f"Raw data shape: {df.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Convert to numpy - skip first column (customer index)
    X_raw = df.iloc[:, 1:].values.astype(np.float32)
    y_raw = labels.iloc[:, 0].values.astype(np.int64)
    
    # Ensure matching lengths
    n_samples = min(X_raw.shape[0], y_raw.shape[0])
    X_raw = X_raw[:n_samples]
    y_raw = y_raw[:n_samples]
    
    print(f"X: {X_raw.shape}, y: {y_raw.shape}, anomalies: {y_raw.sum()} ({y_raw.mean()*100:.2f}%)")
    
    # Replace inf/-inf with NaN, then fill with column mean
    X_raw = np.where(np.isfinite(X_raw), X_raw, np.nan)
    
    # Handle NaN values
    col_means = np.nanmean(X_raw, axis=0)
    inds = np.where(np.isnan(X_raw))
    X_raw[inds] = np.take(col_means, inds[1])
    
    # Sample if needed
    if sample_ratio < 1.0:
        n_samples = int(X_raw.shape[0] * sample_ratio)
        rng = np.random.RandomState(random_state)
        indices = rng.choice(X_raw.shape[0], n_samples, replace=False)
        X_raw = X_raw[indices]
        y_raw = y_raw[indices]
        print(f"After sampling ({sample_ratio*100:.0f}%): X {X_raw.shape}, anomalies: {y_raw.sum()} ({y_raw.mean()*100:.2f}%)")
    
    # Global min-max normalization (preserves overall consumption level)
    if NORMALIZE:
        x_min = np.nanmin(X_raw)
        x_max = np.nanmax(X_raw)
        if x_max > x_min:
            X = (X_raw - x_min) / (x_max - x_min)
        else:
            X = X_raw
    else:
        X = X_raw
    
    # Save processed data (with sample ratio in filename)
    proc_x_save = os.path.join(PROCESSED_DIR, f'X_s{sample_ratio:.2f}.npy')
    proc_y_save = os.path.join(PROCESSED_DIR, f'y_s{sample_ratio:.2f}.npy')
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(proc_x_save, X)
    np.save(proc_y_save, y_raw)
    
    print(f"Final: X {X.shape}, y {y_raw.shape}, anomalies: {y_raw.sum()} ({y_raw.mean()*100:.2f}%)")
    return X, y_raw


def create_data_loaders(X, y, batch_size=BATCH_SIZE, val_ratio=VAL_RATIO, 
                        test_ratio=TEST_RATIO, random_state=42):
    """Create train/val/test data loaders."""
    # Split into train+val and test
    test_size = int(X.shape[0] * test_ratio)
    val_size = int(X.shape[0] * val_ratio)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train anomaly ratio: {y_train.mean()*100:.2f}%")
    print(f"Val anomaly ratio: {y_val.mean()*100:.2f}%")
    print(f"Test anomaly ratio: {y_test.mean()*100:.2f}%")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, sub_seq_len=SUB_SEQ_LEN, augment=True)
    val_dataset = TimeSeriesDataset(X_val, y_val, sub_seq_len=SUB_SEQ_LEN, augment=False)
    test_dataset = TimeSeriesDataset(X_test, y_test, sub_seq_len=SUB_SEQ_LEN, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, (X_train, y_train, X_val, y_val, X_test, y_test)


class TimeSeriesDataset(Dataset):
    """Dataset for time series anomaly detection with augmentation."""
    
    def __init__(self, X, y, sub_seq_len=SUB_SEQ_LEN, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.sub_seq_len = sub_seq_len
        self.augment = augment
        self.n_samples = X.shape[0]
        self.seq_len = X.shape[1]
        
        # Pre-compute random sub-sequence start indices
        if self.seq_len > self.sub_seq_len:
            self.start_inds = np.random.randint(0, self.seq_len - self.sub_seq_len, size=self.n_samples)
        else:
            self.start_inds = np.zeros(self.n_samples, dtype=int)
            self.sub_seq_len = self.seq_len
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # Extract sub-sequence
        start = self.start_inds[idx]
        end = start + self.sub_seq_len
        if end > self.seq_len:
            x = x[-self.sub_seq_len:]
        else:
            x = x[start:end]
        
        x = x.unsqueeze(-1)  # Add channel dimension: (seq_len, 1)
        
        if self.augment:
            # Generate augmented views for contrastive learning
            x_aug1 = self._augment(x)
            x_aug2 = self._augment(x)
            return x, x_aug1, x_aug2, y
        
        return x, y
    
    def _augment(self, x):
        """Apply random augmentation."""
        aug = x.clone()
        
        # Random masking
        if np.random.random() > 0.5:
            mask_len = int(self.sub_seq_len * np.random.uniform(0.05, 0.15))
            mask_start = np.random.randint(0, self.sub_seq_len - mask_len)
            aug[mask_start:mask_start+mask_len] = 0
        
        # Random noise
        if np.random.random() > 0.5:
            noise_std = np.random.uniform(0.01, 0.05)
            aug += torch.randn_like(aug) * noise_std
        
        # Random scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            aug *= scale
        
        # Random time shift
        if np.random.random() > 0.5:
            shift = np.random.randint(-10, 10)
            aug = torch.roll(aug, shifts=shift, dims=0)
        
        return aug


def get_test_data_for_eval(X, y, sub_seq_len=SUB_SEQ_LEN):
    """Get full test data for evaluation with sub-sequences."""
    dataset = TimeSeriesDataset(X, y, sub_seq_len=sub_seq_len, augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loader


if __name__ == '__main__':
    X, y = load_sgcc_data(sample_ratio=0.1)
    train_loader, val_loader, test_loader, splits = create_data_loaders(X, y)
    print("Data loaders created successfully!")
    
    # Check a batch
    for batch in train_loader:
        x, x_aug1, x_aug2, y = batch
        print(f"Batch: x {x.shape}, x_aug1 {x_aug1.shape}, y {y.shape}")
        break