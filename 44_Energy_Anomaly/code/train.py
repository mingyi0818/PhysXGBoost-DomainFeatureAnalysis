"""Training script for TCR-AD and baseline models."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import json
import time
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from models import TCRAD, tcr_ad_loss, Autoencoder, VAE, DAGMM
from data_loader import load_sgcc_data, create_data_loaders


def set_seed(seed):
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_tcrad(model, train_loader, val_loader, device, n_epochs=N_EPOCHS, 
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=EARLY_STOP_PATIENCE):
    """Train TCR-AD model with mixed precision for speed."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Mixed precision training for 2x speedup
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'contrastive_loss': [], 'recon_loss': []}
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_c = 0
        train_r = 0
        n_batches = 0
        
        for batch in train_loader:
            x, x_aug1, x_aug2, y = batch
            x = x.to(device)
            x_aug1 = x_aug1.to(device)
            x_aug2 = x_aug2.to(device)
            y = y.to(device)
            
            # For reconstruction and contrastive: only use normal samples
            normal_mask = (y == 0)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    total_loss, c_loss, r_loss, cls_loss = tcr_ad_loss(
                        model, x, x_aug1, x_aug2, 
                        y_full=y, normal_mask=normal_mask, cls_weight=0.2
                    )
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss, c_loss, r_loss, cls_loss = tcr_ad_loss(
                    model, x, x_aug1, x_aug2, 
                    y_full=y, normal_mask=normal_mask, cls_weight=0.2
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += total_loss.item()
            train_c += c_loss.item()
            train_r += r_loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(device)
                x_aug1 = x + torch.randn_like(x) * 0.01
                x_aug2 = x + torch.randn_like(x) * 0.01
                if use_amp:
                    with torch.cuda.amp.autocast():
                        total_loss, _, _, _ = tcr_ad_loss(model, x, x_aug1, x_aug2)
                else:
                    total_loss, _, _, _ = tcr_ad_loss(model, x, x_aug1, x_aug2)
                val_loss += total_loss.item()
                val_n += 1
        
        avg_train_loss = train_loss / n_batches
        avg_val_loss = val_loss / val_n
        avg_c = train_c / n_batches
        avg_r = train_r / n_batches
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['contrastive_loss'].append(avg_c)
        history['recon_loss'].append(avg_r)
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} (C: {avg_c:.4f}, R: {avg_r:.4f}), Val Loss: {avg_val_loss:.4f}", flush=True)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'tcrad_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'tcrad_best.pth')))
    print(f"Best model from epoch {best_epoch+1}, val_loss: {best_val_loss:.4f}", flush=True)
    
    return model, history


def train_ae(model, train_loader, device, n_epochs=8, lr=1e-3):
    """Train autoencoder baseline."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        n = 0
        for batch in train_loader:
            x, _, _, y = batch
            mask = (y == 0)
            if mask.sum() == 0:
                continue
            x = x[mask].to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    recon = model(x)
                    loss = nn.MSELoss()(recon, x.squeeze(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon = model(x)
                loss = nn.MSELoss()(recon, x.squeeze(-1))
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            n += 1
        if (epoch+1) % 4 == 0:
            print(f"AE Epoch {epoch+1}/{n_epochs} - Loss: {train_loss/n:.4f}", flush=True)
    return model


def train_vae(model, train_loader, device, n_epochs=8, lr=1e-3):
    """Train VAE baseline."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        n = 0
        for batch in train_loader:
            x, _, _, y = batch
            mask = (y == 0)
            if mask.sum() == 0:
                continue
            x = x[mask].to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    recon, mu, logvar = model(x)
                    recon_loss = nn.MSELoss()(recon, x.squeeze(-1))
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                    loss = recon_loss + 0.1 * kl_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, mu, logvar = model(x)
                recon_loss = nn.MSELoss()(recon, x.squeeze(-1))
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                loss = recon_loss + 0.1 * kl_loss
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            n += 1
        if (epoch+1) % 4 == 0:
            print(f"VAE Epoch {epoch+1}/{n_epochs} - Loss: {train_loss/n:.4f}", flush=True)
    return model


def train_dagmm(model, train_loader, device, n_epochs=8, lr=1e-3):
    """Train DAGMM baseline."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        n = 0
        for batch in train_loader:
            x, _, _, y = batch
            mask = (y == 0)
            if mask.sum() == 0:
                continue
            x = x[mask].to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    recon, z, gamma = model(x)
                    recon_loss = nn.MSELoss()(recon, x.squeeze(-1))
                    loss = recon_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, z, gamma = model(x)
                recon_loss = nn.MSELoss()(recon, x.squeeze(-1))
                loss = recon_loss
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            n += 1
        if (epoch+1) % 4 == 0:
            print(f"DAGMM Epoch {epoch+1}/{n_epochs} - Loss: {train_loss/n:.4f}", flush=True)
    return model


def get_anomaly_scores_tcrad(model, loader, device):
    """Get TCR-AD anomaly scores."""
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            score = model.get_anomaly_score(x)
            scores.append(score.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def evaluate_anomaly_detection(scores, labels):
    """Evaluate anomaly detection performance."""
    # AUC-ROC
    auc = roc_auc_score(labels, scores)
    
    # Find best threshold
    thresholds = np.percentile(scores, np.linspace(50, 100, 100))
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}
    
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {
                'f1': f1,
                'precision': precision_score(labels, preds),
                'recall': recall_score(labels, preds)
            }
    
    return {
        'auc_roc': auc,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'best_precision': best_metrics.get('precision', 0),
        'best_recall': best_metrics.get('recall', 0)
    }


def evaluate_sklearn_baseline(model, X_test, y_test):
    """Evaluate sklearn-based baseline (OCSVM, IForest)."""
    scores = model.decision_function(X_test) if hasattr(model, 'decision_function') else -model.score_samples(X_test)
    scores = -scores  # Higher score = more anomalous
    return evaluate_anomaly_detection(scores, y_test)


if __name__ == '__main__':
    print("="*60, flush=True)
    print("TCR-AD Training", flush=True)
    print("="*60, flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    
    # Load data
    X, y = load_sgcc_data(sample_ratio=SAMPLE_RATIO)
    train_loader, val_loader, test_loader, splits = create_data_loaders(X, y)
    
    # Create model
    model = TCRAD().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TCR-AD parameters: {n_params:,}", flush=True)
    
    # Train
    model, history = train_tcrad(model, train_loader, val_loader, device, n_epochs=N_EPOCHS)
    
    # Evaluate
    scores, labels = get_anomaly_scores_tcrad(model, test_loader, device)
    results = evaluate_anomaly_detection(scores, labels)
    print(f"\nTest results: AUC-ROC={results['auc_roc']:.4f}, F1={results['best_f1']:.4f}", flush=True)
    print(f"Precision={results['best_precision']:.4f}, Recall={results['best_recall']:.4f}", flush=True)
