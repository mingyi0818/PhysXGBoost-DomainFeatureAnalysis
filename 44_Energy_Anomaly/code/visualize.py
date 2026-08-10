"""Visualization scripts for TCR-AD experiments."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from data_loader import load_sgcc_data


# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_results(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def fig1_architecture():
    """Figure 1: TCR-AD architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Input
    ax.text(6, 7.5, 'Input Time Series (256 days)', ha='center', va='center', 
            fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Two branches
    ax.annotate('', xy=(3, 6.5), xytext=(6, 7), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(9, 6.5), xytext=(6, 7), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Time encoder
    ax.text(3, 5.5, 'Time-Domain Encoder\n(Multi-Scale 1D-CNN\n+ Self-Attention)', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.annotate('', xy=(3, 4.5), xytext=(3, 5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Freq encoder
    ax.text(9, 5.5, 'Frequency-Domain\nEncoder\n(FFT + MLP)', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.annotate('', xy=(9, 4.5), xytext=(9, 5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Fusion gate
    ax.text(6, 4, 'Adaptive Gated Fusion', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax.annotate('', xy=(6, 3.5), xytext=(6, 3.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Shared embedding
    ax.text(6, 3, 'Shared Embedding\n(128-dim)', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Two heads
    ax.annotate('', xy=(3.5, 2.5), xytext=(6, 2.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(8.5, 2.5), xytext=(6, 2.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Contrastive head
    ax.text(3.5, 1.8, 'Contrastive Head\n(NT-Xent Loss)', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    ax.annotate('', xy=(3.5, 1.2), xytext=(3.5, 1.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Reconstruction head
    ax.text(8.5, 1.8, 'Reconstruction Head\n(MSE Loss)', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='lightskyblue', alpha=0.8))
    ax.annotate('', xy=(8.5, 1.2), xytext=(8.5, 1.5), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Combined score
    ax.text(6, 0.7, 'Combined Anomaly Score\nα × Recon_Error + (1-α) × Contrastive_Score', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.9))
    
    ax.set_title('Figure 1: TCR-AD Architecture Overview', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig1_architecture.png'))
    plt.close()
    print("Figure 1 saved")


def fig2_comparison():
    """Figure 2: Main comparison results bar chart."""
    df = load_results('main_comparison_summary.csv')
    if df is None:
        print("No main_comparison_summary.csv found")
        return
    
    models = df['model'].values
    auc_means = df['auc_roc_mean'].values
    auc_stds = df['auc_roc_std'].values
    f1_means = df['f1_mean'].values
    f1_stds = df['f1_std'].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # AUC-ROC
    colors = ['#2ecc71' if m == 'TCR-AD' else '#3498db' for m in models]
    bars1 = ax1.bar(models, auc_means, yerr=auc_stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('AUC-ROC', fontsize=12)
    ax1.set_title('AUC-ROC Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='x', rotation=45)
    for bar, mean, std in zip(bars1, auc_means, auc_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # F1-Score
    bars2 = ax2.bar(models, f1_means, yerr=f1_stds, capsize=5, color=colors, alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='x', rotation=45)
    for bar, mean, std in zip(bars2, f1_means, f1_stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Figure 2: Performance Comparison Across Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig2_comparison.png'))
    plt.close()
    print("Figure 2 saved")


def fig3_ablation():
    """Figure 3: Ablation study results."""
    df = load_results('ablation_results.csv')
    if df is None:
        print("No ablation_results.csv found")
        return
    
    variants = df['variant'].values
    auc_vals = df['auc_roc'].values
    f1_vals = df['best_f1'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(variants))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, auc_vals, width, label='AUC-ROC', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_vals, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Ablation Variant', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Figure 3: Ablation Study Results', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig3_ablation.png'))
    plt.close()
    print("Figure 3 saved")


def fig4_sensitivity():
    """Figure 4: Parameter sensitivity analysis."""
    df = load_results('sensitivity_all.csv')
    if df is None:
        print("No sensitivity_all.csv found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    params = ['embedding_dim', 'sub_seq_len', 'learning_rate', 'contrastive_weight']
    titles = ['Embedding Dimension', 'Sub-Sequence Length', 'Learning Rate', 'Contrastive Weight']
    
    for i, (param, title) in enumerate(zip(params, titles)):
        param_df = df[df['param'] == param]
        if param_df.empty:
            continue
        
        values = param_df['value'].values
        auc_vals = param_df['auc_roc'].values
        f1_vals = param_df['f1'].values
        
        ax = axes[i]
        ax.plot(range(len(values)), auc_vals, 'o-', label='AUC-ROC', color='#3498db', linewidth=2, markersize=8)
        ax.plot(range(len(values)), f1_vals, 's--', label='F1-Score', color='#e74c3c', linewidth=2, markersize=8)
        
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=45)
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{title} Sensitivity', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 4: Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig4_sensitivity.png'))
    plt.close()
    print("Figure 4 saved")


def fig5_tsne():
    """Figure 5: t-SNE visualization of learned embeddings."""
    from sklearn.manifold import TSNE
    
    print("Generating t-SNE visualization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    from models import TCRAD
    model = TCRAD().to(device)
    ckpt_path = os.path.join(CKPT_DIR, 'tcrad_best.pth')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded trained model")
    else:
        print("No trained model found, using untrained")
    
    model.eval()
    
    # Load a sample of data
    X, y = load_sgcc_data(sample_ratio=0.1)
    
    # Get embeddings
    import torch
    embeddings = []
    labels_small = []
    with torch.no_grad():
        for i in range(min(500, X.shape[0])):
            x = torch.FloatTensor(X[i, :SUB_SEQ_LEN]).unsqueeze(0).unsqueeze(-1).to(device)
            emb = model.encode(x)
            embeddings.append(emb.cpu().numpy())
            labels_small.append(y[i])
    
    embeddings = np.concatenate(embeddings)
    labels_small = np.array(labels_small)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    normal_idx = labels_small == 0
    anomaly_idx = labels_small == 1
    
    ax.scatter(emb_2d[normal_idx, 0], emb_2d[normal_idx, 1], 
              c='#3498db', label='Normal', alpha=0.6, s=20)
    ax.scatter(emb_2d[anomaly_idx, 0], emb_2d[anomaly_idx, 1], 
              c='#e74c3c', label='Anomaly', alpha=0.8, s=30, marker='x')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Figure 5: t-SNE Visualization of Learned Embeddings', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig5_tsne.png'))
    plt.close()
    print("Figure 5 saved")


def fig6_analysis():
    """Figure 6: Additional analysis - ROC curves and score distributions."""
    df = load_results('main_comparison.csv')
    if df is None:
        print("No main_comparison.csv found")
        return
    
    # Compute average scores per model
    models = df['model'].unique()
    model_metrics = {}
    for m in models:
        m_df = df[df['model'] == m]
        model_metrics[m] = {
            'auc_mean': m_df['auc_roc'].mean(),
            'auc_std': m_df['auc_roc'].std(),
            'f1_mean': m_df['best_f1'].mean(),
            'precision_mean': m_df['best_precision'].mean(),
            'recall_mean': m_df['best_recall'].mean()
        }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['auc_roc', 'f1', 'precision', 'recall']
    metric_labels = ['AUC-ROC', 'F1-Score', 'Precision', 'Recall']
    x = np.arange(len(metric_labels))
    width = 0.12
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    for i, (model, color) in enumerate(zip(model_metrics.keys(), colors)):
        values = [model_metrics[model][m] for m in metrics]
        ax.bar(x + i * width - 2 * width, values, width, label=model, color=color, alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Figure 6: Comprehensive Performance Metrics Across Models', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig6_analysis.png'))
    plt.close()
    print("Figure 6 saved")


if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    fig1_architecture()
    fig2_comparison()
    fig3_ablation()
    fig4_sensitivity()
    fig5_tsne()
    fig6_analysis()
    
    print(f"\nAll figures saved to {PLOTS_DIR}")