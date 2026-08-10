#!/usr/bin/env python3
"""
Generate high-resolution figures (>=300 DPI) for all research directions.
Figure 1: Architecture diagram
Figure 2: Model performance comparison (bar chart)
Figure 3: Ablation results OR Feature importance (bar chart)
Figure 4: Sensitivity analysis OR Multi-metric comparison (radar chart)
Figure 5: Feature importance (when available)
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

BASE = Path('D:/ResearchPaperPrepare')

DIRECTIONS = [
    '44_Energy_Anomaly', '46_FlightDelay_PhysXGBoost', '47_OnlineShoppers',
    '48_CreditDefault', '49_Superconductor', '50_BuildingEnergy',
    '51_GasTurbine', '52_CCPP', '53_BikeSharing', '54_NewsPopularity',
    '55_CalHousing', '56_PowerConsumption', '58_CDNOW', '59_NYCProperty',
    '60_StudentPerf', '61_DryBean', '63_HotelBooking', '64_FlightDelay', '65_HR'
]

MODEL_NAMES = ['XGB', 'LGB', 'Cat', 'RF']
MODEL_LABELS = ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest']
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

def load_results(direction):
    results_dir = BASE / direction / 'results'
    data = {}
    for fname in ['summary.json', 'comprehensive_results.json', 'additional_metrics.json',
                  'nox_summary.json', 'per_seed_results.json']:
        path = results_dir / fname
        if path.exists():
            try:
                with open(path) as f:
                    key = fname.replace('.json', '')
                    data[key] = json.load(f)
            except:
                pass
    return data

def get_summary(data):
    """Get summary data from various sources."""
    if 'summary' in data:
        return data['summary']
    if 'nox_summary' in data:
        return data['nox_summary']
    return None

def get_metric_name(data):
    summary = get_summary(data)
    if summary and 'Raw' in summary:
        for v in summary['Raw'].values():
            if 'AUC' in v: return 'AUC'
            if 'R2' in v: return 'R2'
            if 'F1' in v: return 'F1'
    return 'Score'

def fig1_architecture(direction, data, plots_dir):
    """Figure 1: Architecture diagram."""
    n_raw = 'N'
    n_dom = 'N'
    n_total = 'N'
    
    if 'additional_metrics' in data:
        am = data['additional_metrics']
        n_raw = am.get('n_features', 'N')
        if 'feature_importance' in am and 'XGB' in am['feature_importance']:
            n_dom = len(am['feature_importance']['XGB'])
            n_total = int(n_raw) + int(n_dom) if isinstance(n_raw, int) else 'N'
    
    if 'comprehensive_results' in data:
        cr = data['comprehensive_results']
        n_raw = cr.get('n_raw_features', n_raw)
        n_dom = cr.get('n_domain_features', n_dom)
        n_total = cr.get('n_total_features', n_total)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    boxes = [
        (1, 7, f'Raw Features\n({n_raw} features)', '#E3F2FD'),
        (5, 7, f'Domain Feature\nEngineering\n({n_dom} new features)', '#FFF3E0'),
        (9, 7, f'Augmented\nFeatures\n({n_total} total)', '#E8F5E9'),
        (11, 4, 'Tree Models\nXGBoost\nLightGBM\nCatBoost\nRandom Forest', '#F3E5F5'),
        (7, 1, 'Statistical\nAnalysis\n(Wilcoxon, t-test,\n95% CI, Cohen\'s d)', '#FFEBEE'),
        (3, 4, 'Ablation\n& Sensitivity\nAnalysis', '#FFFDE7'),
    ]
    
    for x, y, text, color in boxes:
        box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    arrows = [
        (1.8, 7, 4.2, 7), (5.8, 7, 8.2, 7),
        (9.8, 6.2, 10.2, 4.8), (10.2, 3.2, 7.8, 1.8),
        (8.2, 6.2, 3.8, 4.8),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_title(f'{direction}: Domain Feature Augmentation Framework', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(plots_dir / 'fig1_architecture.png', dpi=300)
    plt.close()
    return True

def fig2_performance_comparison(direction, data, plots_dir):
    """Figure 2: Bar chart comparing Raw vs Domain performance."""
    summary = get_summary(data)
    if not summary:
        return False
    
    metric = get_metric_name(data)
    metric_key = metric.replace('R2', 'R2').replace('²', '2')
    
    raw_vals, dom_vals, std_raw, std_dom = [], [], [], []
    for m in MODEL_NAMES:
        if m in summary.get('Raw', {}):
            val = summary['Raw'][m].get(metric_key, summary['Raw'][m].get(metric, 0))
            raw_vals.append(val)
            std_raw.append(summary['Raw'][m].get('std', 0))
        else:
            raw_vals.append(0)
            std_raw.append(0)
        
        if m in summary.get('Domain', {}):
            val = summary['Domain'][m].get(metric_key, summary['Domain'][m].get(metric, 0))
            dom_vals.append(val)
            std_dom.append(summary['Domain'][m].get('std', 0))
        else:
            dom_vals.append(0)
            std_dom.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODEL_LABELS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, raw_vals, width, label='Raw Features', color='#90CAF9', yerr=std_raw, capsize=3)
    bars2 = ax.bar(x + width/2, dom_vals, width, label='Domain Features', color='#1565C0', yerr=std_dom, capsize=3)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{direction}: Raw vs Domain Feature Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.002, f'{h:.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.002, f'{h:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'fig2_performance_comparison.png', dpi=300)
    plt.close()
    return True

def fig3_ablation_or_feature_importance(direction, data, plots_dir):
    """Figure 3: Ablation results (if available) OR Feature importance."""
    # Try ablation first
    if 'comprehensive_results' in data and 'ablation' in data.get('comprehensive_results', {}):
        ablation = data['comprehensive_results']['ablation']
        if ablation:
            features = list(ablation.keys())[:10]
            means = [ablation[f].get('mean', 0) for f in features]
            stds = [ablation[f].get('std', 0) for f in features]
            
            metric = get_metric_name(data)
            summary = get_summary(data)
            baseline = 0
            if summary and 'Domain' in summary and 'XGB' in summary['Domain']:
                baseline = summary['Domain']['XGB'].get(metric, 0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(features))
            bars = ax.bar(x, means, yerr=stds, capsize=3,
                         color=['#4CAF50' if m >= baseline else '#FF9800' for m in means])
            ax.axhline(y=baseline, color='red', linestyle='--', label=f'Full Domain ({baseline:.4f})')
            ax.set_xlabel('Removed Domain Feature', fontsize=12)
            ax.set_ylabel(f'{metric} (Leave-One-Out)', fontsize=12)
            ax.set_title(f'{direction}: Ablation Analysis', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=8, rotation=45, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'fig3_ablation_results.png', dpi=300)
            plt.close()
            return True
    
    # Fallback: Feature importance from additional_metrics
    if 'additional_metrics' in data and 'feature_importance' in data['additional_metrics']:
        fi_data = data['additional_metrics']['feature_importance']
        for model_key in ['XGB', 'LGB', 'Cat', 'RF']:
            if model_key not in fi_data or not fi_data[model_key]:
                continue
            
            fi = fi_data[model_key]
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            features = [f[0] for f in sorted_fi]
            values = [f[1] for f in sorted_fi]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color='#1565C0', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'{direction}: Top 15 Feature Importance ({model_key})', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            
            for i, v in enumerate(values):
                ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'fig3_feature_importance.png', dpi=300)
            plt.close()
            return True
    
    return False

def fig4_sensitivity_or_radar(direction, data, plots_dir):
    """Figure 4: Sensitivity heatmap OR Multi-metric radar chart."""
    # Try sensitivity first
    if 'comprehensive_results' in data and 'sensitivity' in data.get('comprehensive_results', {}):
        sensitivity = data['comprehensive_results']['sensitivity']
        if sensitivity:
            n_ests = [100, 200, 300, 500]
            depths = [4, 6, 8, 10]
            matrix = np.full((len(depths), len(n_ests)), np.nan)
            
            for i, d in enumerate(depths):
                for j, n in enumerate(n_ests):
                    key = f"n_est={n}_depth={d}"
                    if key in sensitivity:
                        matrix[i, j] = sensitivity[key].get('mean', 0)
            
            if not np.all(np.isnan(matrix)):
                metric = get_metric_name(data)
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(np.arange(len(n_ests)))
                ax.set_yticks(np.arange(len(depths)))
                ax.set_xticklabels([str(n) for n in n_ests])
                ax.set_yticklabels([str(d) for d in depths])
                ax.set_xlabel('n_estimators', fontsize=12)
                ax.set_ylabel('max_depth', fontsize=12)
                ax.set_title(f'{direction}: Hyperparameter Sensitivity', fontsize=14)
                
                for i in range(len(depths)):
                    for j in range(len(n_ests)):
                        if not np.isnan(matrix[i, j]):
                            ax.text(j, i, f'{matrix[i, j]:.4f}', ha='center', va='center',
                                   fontsize=9, color='white' if matrix[i, j] > np.nanmean(matrix) else 'black')
                
                plt.colorbar(im, ax=ax, label=metric)
                plt.tight_layout()
                plt.savefig(plots_dir / 'fig4_sensitivity_analysis.png', dpi=300)
                plt.close()
                return True
    
    # Fallback: Multi-metric comparison from additional_metrics
    if 'additional_metrics' in data and 'models' in data['additional_metrics']:
        models = data['additional_metrics']['models']
        task = data['additional_metrics'].get('task', 'regression')
        
        if task == 'classification':
            metrics = ['AUC', 'Accuracy', 'F1_Macro', 'Precision', 'Recall']
        else:
            metrics = ['R2', 'RMSE', 'MAE', 'Pearson_r']
        
        # Normalize metrics to [0, 1] for radar
        norm_vals = {}
        for mk in metrics:
            vals = []
            for m in MODEL_NAMES:
                if m in models:
                    vals.append(models[m].get(f'{mk}_mean', 0))
                else:
                    vals.append(0)
            
            if not vals or max(vals) == min(vals):
                norm_vals[mk] = [0.5] * len(vals)
            else:
                if mk in ['RMSE', 'MAE']:  # Lower is better
                    norm_vals[mk] = [1 - (v - min(vals)) / (max(vals) - min(vals)) for v in vals]
                else:
                    norm_vals[mk] = [(v - min(vals)) / (max(vals) - min(vals)) for v in vals]
        
        # Create grouped bar chart instead of radar for clarity
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, (m, label, color) in enumerate(zip(MODEL_NAMES, MODEL_LABELS, COLORS)):
            vals = [norm_vals[mk][i] for mk in metrics]
            ax.bar(x + i * width - 1.5 * width, vals, width, label=label, color=color, alpha=0.8)
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Normalized Score', fontsize=12)
        ax.set_title(f'{direction}: Multi-Metric Model Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'fig4_multi_metric_comparison.png', dpi=300)
        plt.close()
        return True
    
    return False

def fig5_training_time(direction, data, plots_dir):
    """Figure 5: Training time comparison."""
    if 'additional_metrics' not in data or 'models' not in data['additional_metrics']:
        return False
    
    models = data['additional_metrics']['models']
    
    times = []
    pred_times = []
    for m in MODEL_NAMES:
        if m in models:
            times.append(models[m].get('train_time_s_mean', 0))
            pred_times.append(models[m].get('pred_time_s_mean', 0))
        else:
            times.append(0)
            pred_times.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODEL_LABELS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, times, width, label='Training Time (s)', color='#FF9800')
    bars2 = ax.bar(x + width/2, pred_times, width, label='Prediction Time (s)', color='#4CAF50')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'{direction}: Training and Prediction Time', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'fig5_training_time.png', dpi=300)
    plt.close()
    return True

def generate_for_direction(direction):
    data = load_results(direction)
    if not data:
        print(f"  {direction}: No results data")
        return 0
    
    plots_dir = BASE / direction / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    count = 0
    if fig1_architecture(direction, data, plots_dir): count += 1
    if fig2_performance_comparison(direction, data, plots_dir): count += 1
    if fig3_ablation_or_feature_importance(direction, data, plots_dir): count += 1
    if fig4_sensitivity_or_radar(direction, data, plots_dir): count += 1
    if fig5_training_time(direction, data, plots_dir): count += 1
    
    print(f"  {direction}: {count}/5 figures generated")
    return count

print(f"\n{'='*60}")
print(f"Generating figures for all directions")
print(f"{'='*60}\n")

total_figs = 0
for d in DIRECTIONS:
    try:
        total_figs += generate_for_direction(d)
    except Exception as e:
        print(f"  {d}: ERROR - {e}")

print(f"\n{'='*60}")
print(f"Total figures generated: {total_figs}")
print(f"{'='*60}")
