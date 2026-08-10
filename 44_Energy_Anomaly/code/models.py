"""TCR-AD: Temporal Contrastive Reconstruction for Anomaly Detection in Energy Consumption.

Core components:
1. Time-domain multi-scale encoder (1D-CNN with residual connections)
2. Frequency-domain encoder (FFT-based)
3. Contrastive learning head (NT-Xent loss)
4. Reconstruction decoder
5. Combined anomaly scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


class MultiScaleConv1D(nn.Module):
    """Multi-scale 1D convolutional encoder with residual connections."""
    
    def __init__(self, in_channels, hidden_dims, kernel_sizes, dropout=0.1):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        
        for k in kernel_sizes:
            blocks = []
            prev_ch = in_channels
            for h in hidden_dims:
                blocks.append(nn.Sequential(
                    nn.Conv1d(prev_ch, h, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(h),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout)
                ))
                prev_ch = h
            self.conv_blocks.append(nn.Sequential(*blocks))
        
        # Fusion layer - concatenate all multi-scale outputs (each has hidden_dims[-1] channels)
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dims[-1] * len(kernel_sizes), hidden_dims[-1], kernel_size=1),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        """x: (batch, channels, seq_len)"""
        outputs = []
        for block in self.conv_blocks:
            out = block(x)
            outputs.append(out)
        
        fused = torch.cat(outputs, dim=1)  # (batch, sum(hidden)*n_kernels, seq_len)
        fused = self.fusion(fused)
        return fused


class TimeEncoder(nn.Module):
    """Time-domain encoder with multi-scale convolutions and attention."""
    
    def __init__(self, in_channels=1, hidden_dims=TIME_ENCODER_HIDDEN, 
                 kernel_sizes=CONV_KERNEL_SIZES, embed_dim=EMBED_DIM, 
                 n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        
        self.multi_scale_conv = MultiScaleConv1D(
            in_channels, hidden_dims, kernel_sizes, dropout
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1], num_heads=n_heads, 
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dims[-1])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Project to embedding
        self.proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        """x: (batch, seq_len, 1)"""
        # (batch, seq_len, 1) -> (batch, 1, seq_len)
        x = x.transpose(1, 2)
        
        # Multi-scale conv
        conv_out = self.multi_scale_conv(x)  # (batch, hidden[-1], seq_len)
        
        # Self-attention
        attn_in = conv_out.transpose(1, 2)  # (batch, seq_len, hidden[-1])
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)
        attn_out = self.norm(attn_out + attn_in)  # Residual
        attn_out = attn_out.transpose(1, 2)  # (batch, hidden[-1], seq_len)
        
        # Global pooling
        pooled = self.global_pool(attn_out).squeeze(-1)  # (batch, hidden[-1])
        
        # Project
        emb = self.proj(pooled)  # (batch, embed_dim)
        
        return emb


class FreqEncoder(nn.Module):
    """Frequency-domain encoder using FFT."""
    
    def __init__(self, seq_len=SUB_SEQ_LEN, n_bins=FREQ_N_BINS, 
                 hidden_dims=FREQ_ENCODER_HIDDEN, embed_dim=EMBED_DIM):
        super().__init__()
        
        self.seq_len = seq_len
        self.n_bins = min(n_bins, seq_len // 2 + 1)
        
        # MLP for frequency features
        layers = []
        prev_dim = self.n_bins
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.freq_mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """x: (batch, seq_len, 1)"""
        # Compute FFT
        x_flat = x.squeeze(-1)  # (batch, seq_len)
        spectrum = torch.fft.rfft(x_flat, dim=-1)  # (batch, seq_len//2+1)
        magnitude = torch.abs(spectrum)  # (batch, seq_len//2+1)
        
        # Use first n_bins
        magnitude = magnitude[:, :self.n_bins]
        
        # Normalize
        magnitude = F.normalize(magnitude, p=2, dim=-1)
        
        # MLP
        emb = self.freq_mlp(magnitude)  # (batch, embed_dim)
        return emb


class TCRAD(nn.Module):
    """Temporal Contrastive Reconstruction for Anomaly Detection.
    
    Combines:
    - Time-domain contrastive learning
    - Frequency-domain feature learning
    - Reconstruction-based anomaly detection
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        self.time_encoder = TimeEncoder(
            in_channels=1,
            hidden_dims=TIME_ENCODER_HIDDEN,
            kernel_sizes=CONV_KERNEL_SIZES,
            embed_dim=EMBED_DIM,
            n_heads=N_HEADS,
            dropout=DROPOUT
        )
        
        self.freq_encoder = FreqEncoder(
            seq_len=SUB_SEQ_LEN,
            n_bins=FREQ_N_BINS,
            hidden_dims=FREQ_ENCODER_HIDDEN,
            embed_dim=EMBED_DIM
        )
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.BatchNorm1d(EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, EMBED_DIM)
        )
        
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(EMBED_DIM, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, SUB_SEQ_LEN)
        )
        
        # Classification head for semi-supervised learning
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        self.use_freq_encoder = USE_FREQ_ENCODER
        self.use_time_encoder = USE_TIME_ENCODER
    
    def encode(self, x):
        """Encode input to embedding.
        x: (batch, seq_len, 1)
        Returns: (batch, embed_dim)
        """
        time_emb = None
        freq_emb = None
        
        if self.use_time_encoder:
            time_emb = self.time_encoder(x)
        
        if self.use_freq_encoder:
            freq_emb = self.freq_encoder(x)
        
        if time_emb is not None and freq_emb is not None:
            # Adaptive fusion
            gate = self.fusion_gate(torch.cat([time_emb, freq_emb], dim=-1))
            emb = gate[:, 0:1] * time_emb + gate[:, 1:2] * freq_emb
        elif time_emb is not None:
            emb = time_emb
        else:
            emb = freq_emb
        
        return emb
    
    def forward(self, x):
        """Forward pass.
        x: (batch, seq_len, 1)
        Returns: (emb, recon, proj, logits)
        """
        emb = self.encode(x)  # (batch, embed_dim)
        proj = self.contrastive_head(emb)  # (batch, embed_dim)
        recon = self.decoder(emb)  # (batch, seq_len)
        logits = self.classifier(emb).squeeze(-1)  # (batch,)
        return emb, recon, proj, logits
    
    def get_anomaly_score(self, x):
        """Compute anomaly score using reconstruction error only.
        x: (batch, seq_len, 1)
        Returns: (batch,) anomaly scores
        """
        emb, recon, proj, logits = self.forward(x)
        
        # Reconstruction error (MSE per sample)
        x_flat = x.squeeze(-1)  # (batch, seq_len)
        recon_error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=-1)  # (batch,)
        
        return recon_error


def nt_xent_loss(z1, z2, temperature=CONTRASTIVE_TEMPERATURE):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    Args:
        z1, z2: (batch, embed_dim) - projected embeddings of two augmentations
    Returns: scalar loss
    """
    batch_size = z1.shape[0]
    
    # Normalize
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    
    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2*batch, embed_dim)
    
    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2*batch, 2*batch)
    
    # Mask out self-similarity (use float16-safe value for mixed precision)
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e4)
    
    # Positive pairs: (i, i+batch) and (i+batch, i) — vectorized
    pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device)
    idx = torch.arange(batch_size, device=z.device)
    pos_mask[idx, idx + batch_size] = 1
    pos_mask[idx + batch_size, idx] = 1
    
    # Compute loss
    exp_sim = torch.exp(sim)
    pos_exp = exp_sim * pos_mask
    pos_sum = pos_exp.sum(dim=-1)
    all_sum = exp_sim.sum(dim=-1)
    loss = -torch.log(pos_sum / all_sum).mean()
    
    return loss


def tcr_ad_loss(model, x_full, x_aug1_full, x_aug2_full, y_full=None, 
                normal_mask=None, cls_weight=0.1):
    """Compute combined TCR-AD loss with semi-supervised classification.
    
    Args:
        model: TCRAD model
        x_full: original input (batch, seq_len, 1) - all samples
        x_aug1_full, x_aug2_full: augmented views (batch, seq_len, 1)
        y_full: labels for classification (batch,), optional
        normal_mask: boolean mask for normal samples (y==0)
        cls_weight: weight for classification loss
    
    Returns:
        total_loss, contrastive_loss, recon_loss, cls_loss
    """
    # Forward all samples for classification
    emb_full, recon_full, proj_full, logits_full = model(x_full)
    
    # Classification loss on ALL samples
    cls_loss = torch.tensor(0.0, device=x_full.device)
    if y_full is not None:
        cls_loss = F.binary_cross_entropy_with_logits(logits_full, y_full.float())
    
    # Reconstruction and contrastive loss on NORMAL samples only (or all if no mask)
    recon_loss = torch.tensor(0.0, device=x_full.device)
    contrastive_loss = torch.tensor(0.0, device=x_full.device)
    
    if normal_mask is None:
        # Validation mode: use all samples
        x_norm = x_full
        recon_norm = recon_full
        proj_norm = proj_full
        x_aug1_norm = x_aug1_full
        x_aug2_norm = x_aug2_full
    elif normal_mask.sum() > 1:
        x_norm = x_full[normal_mask]
        recon_norm = recon_full[normal_mask]
        proj_norm = proj_full[normal_mask]
        x_aug1_norm = x_aug1_full[normal_mask]
        x_aug2_norm = x_aug2_full[normal_mask]
    else:
        x_norm = None
    
    if x_norm is not None:
        # Reconstruction loss
        x_flat = x_norm.squeeze(-1)
        recon_loss = F.mse_loss(recon_norm, x_flat)
        
        # Contrastive loss
        _, _, proj_aug1, _ = model(x_aug1_norm)
        _, _, proj_aug2, _ = model(x_aug2_norm)
        
        c_loss1 = nt_xent_loss(proj_norm, proj_aug1, CONTRASTIVE_TEMPERATURE)
        c_loss2 = nt_xent_loss(proj_norm, proj_aug2, CONTRASTIVE_TEMPERATURE)
        c_loss3 = nt_xent_loss(proj_aug1, proj_aug2, CONTRASTIVE_TEMPERATURE)
        contrastive_loss = (c_loss1 + c_loss2 + c_loss3) / 3
    
    # Combined loss
    total_loss = (CONTRASTIVE_WEIGHT * contrastive_loss + 
                  RECON_WEIGHT * recon_loss + 
                  cls_weight * cls_loss)
    
    return total_loss, contrastive_loss, recon_loss, cls_loss


class Autoencoder(nn.Module):
    """Baseline autoencoder for anomaly detection."""
    
    def __init__(self, seq_len=SUB_SEQ_LEN):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len)
        )
    
    def forward(self, x):
        x = x.squeeze(-1)
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
    
    def get_anomaly_score(self, x):
        recon = self.forward(x)
        x_flat = x.squeeze(-1)
        score = F.mse_loss(recon, x_flat, reduction='none').mean(dim=-1)
        return score


class VAE(nn.Module):
    """Baseline Variational Autoencoder."""
    
    def __init__(self, seq_len=SUB_SEQ_LEN, latent_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = x.squeeze(-1)
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def get_anomaly_score(self, x):
        recon, mu, logvar = self.forward(x)
        x_flat = x.squeeze(-1)
        recon_error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=-1)
        # Add KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return recon_error + 0.1 * kl


class DAGMM(nn.Module):
    """Deep Autoencoding Gaussian Mixture Model for anomaly detection."""
    
    def __init__(self, seq_len=SUB_SEQ_LEN, n_gmm=3, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len)
        )
        # Estimation network
        self.estimate = nn.Sequential(
            nn.Linear(latent_dim + 2, 10),
            nn.ReLU(),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=-1)
        )
        self.n_gmm = n_gmm
    
    def forward(self, x):
        x = x.squeeze(-1)
        z = self.encoder(x)
        recon = self.decoder(z)
        recon_error = F.mse_loss(recon, x, reduction='none').mean(dim=-1, keepdim=True)
        z_norm = torch.norm(z, p=2, dim=-1, keepdim=True)
        features = torch.cat([z, recon_error, z_norm], dim=-1)
        gamma = self.estimate(features)
        return recon, z, gamma
    
    def get_anomaly_score(self, x):
        recon, z, gamma = self.forward(x)
        x_flat = x.squeeze(-1)
        recon_error = F.mse_loss(recon, x_flat, reduction='none').mean(dim=-1)
        return recon_error


if __name__ == '__main__':
    # Test TCR-AD model
    model = TCRAD()
    x = torch.randn(8, SUB_SEQ_LEN, 1)
    x_aug1 = torch.randn(8, SUB_SEQ_LEN, 1)
    x_aug2 = torch.randn(8, SUB_SEQ_LEN, 1)
    
    emb, recon, proj = model(x)
    print(f"Input: {x.shape}")
    print(f"Embedding: {emb.shape}")
    print(f"Reconstruction: {recon.shape}")
    print(f"Projection: {proj.shape}")
    
    total_loss, c_loss, r_loss = tcr_ad_loss(model, x, x_aug1, x_aug2)
    print(f"Total loss: {total_loss:.4f}, Contrastive: {c_loss:.4f}, Recon: {r_loss:.4f}")
    
    scores = model.get_anomaly_score(x)
    print(f"Anomaly scores: {scores.shape}")
    print("Model test passed!")