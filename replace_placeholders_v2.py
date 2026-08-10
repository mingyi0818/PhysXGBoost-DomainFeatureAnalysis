#!/usr/bin/env python3
"""
Comprehensive PLACEHOLDER replacement using additional_metrics.json.
Handles: multi-metric tables, feature importance, training time, ANOVA, etc.
"""
import json
import re
from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats

BASE = Path('D:/ResearchPaperPrepare')

DIRECTIONS = [
    '44_Energy_Anomaly', '46_FlightDelay_PhysXGBoost', '47_OnlineShoppers',
    '48_CreditDefault', '49_Superconductor', '50_BuildingEnergy',
    '51_GasTurbine', '52_CCPP', '53_BikeSharing', '54_NewsPopularity',
    '55_CalHousing', '56_PowerConsumption', '58_CDNOW', '59_NYCProperty',
    '60_StudentPerf', '61_DryBean', '63_HotelBooking', '64_FlightDelay', '65_HR'
]

MODEL_NAMES = {
    'XGB': ['XGBoost', 'XGB'],
    'LGB': ['LightGBM', 'LGB'],
    'Cat': ['CatBoost', 'Cat'],
    'RF': ['Random Forest', 'RandomForest', 'RF'],
}

def load_all_results(direction):
    """Load all available result files."""
    results_dir = BASE / direction / 'results'
    data = {}
    for fname in ['summary.json', 'comprehensive_results.json', 'per_seed_results.json',
                  'additional_metrics.json', 'feature_importance_share.json',
                  'statistical_tests_new.json', 'elasticity_per_dataset.json',
                  'nox_summary.json', 'ablation_results.json', 'ablation_results_v2.json']:
        path = results_dir / fname
        if path.exists():
            try:
                with open(path) as f:
                    key = fname.replace('.json', '')
                    data[key] = json.load(f)
            except:
                pass
    return data

def get_metric_key(data, metric_name):
    """Try to find the metric in various result files."""
    # Check additional_metrics first
    if 'additional_metrics' in data and 'models' in data['additional_metrics']:
        for model_key, model_data in data['additional_metrics']['models'].items():
            for mk in [f'{metric_name}_mean', f'{metric_name}_std']:
                if mk in model_data:
                    return model_data
    return None

def replace_multi_metric_table(content, data):
    """Replace multi-metric table rows like | Accuracy | [PLACEHOLDER: 0.8XX] | [PLACEHOLDER: 0.8XX] |"""
    if 'additional_metrics' not in data or 'models' not in data['additional_metrics']:
        return content
    
    models = data['additional_metrics']['models']
    
    # Metrics to look for
    metric_map = {
        'Accuracy': 'Accuracy',
        'F1-Macro': 'F1_Macro',
        'F1-Micro': 'F1_Micro',
        'Precision': 'Precision',
        'Recall': 'Recall',
        "Cohen's Kappa": 'Cohen_Kappa',
        'Cohen\'s Kappa': 'Cohen_Kappa',
        'MCC': 'MCC',
        'RMSE': 'RMSE',
        'MAE': 'MAE',
        'Pearson r': 'Pearson_r',
        'Pearson': 'Pearson_r',
    }
    
    for display_name, metric_key in metric_map.items():
        for model_short, model_aliases in MODEL_NAMES.items():
            if model_short not in models:
                continue
            model_data = models[model_short]
            mean_key = f'{metric_key}_mean'
            std_key = f'{metric_key}_std'
            if mean_key not in model_data:
                continue
            
            mean_val = model_data[mean_key]
            std_val = model_data.get(std_key, 0)
            
            # Pattern: | Model | [PLACEHOLDER: 0.8XX] | [PLACEHOLDER: 0.8XX] |
            for alias in model_aliases:
                # Match: | Model | [PLACEHOLDER...] | [PLACEHOLDER...] |
                pattern = rf'(\| {re.escape(alias)}[^|]*\| )\[PLACEHOLDER: 0\.\w+[^\]]*\]( \| )\[PLACEHOLDER: 0\.\w+[^\]]*\]'
                def replacer(m):
                    return f"{m.group(1)}{mean_val:.4f}{m.group(2)}{mean_val:.4f}"
                content = re.sub(pattern, replacer, content)
                
                # Match: | Model | [PLACEHOLDER...] |
                pattern2 = rf'(\| {re.escape(alias)}[^|]*\| )\[PLACEHOLDER: 0\.\w+[^\]]*\]'
                content = re.sub(pattern2, lambda m: m.group(1) + f"{mean_val:.4f}", content)
    
    return content

def replace_feature_importance(content, data):
    """Replace feature importance table rows."""
    if 'additional_metrics' not in data or 'feature_importance' not in data['additional_metrics']:
        return content
    
    fi_data = data['additional_metrics']['feature_importance']
    
    # Use XGB feature importance as primary
    if 'XGB' in fi_data:
        fi = fi_data['XGB']
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        
        # Replace: | N | [PLACEHOLDER: feature_name] | [PLACEHOLDER: 0.XXX] | [PLACEHOLDER: Raw/Domain] |
        for rank, (feat, imp) in enumerate(sorted_fi[:15], 1):
            pattern = rf'(\| {rank} \| )\[PLACEHOLDER: feature_name\]( \| )\[PLACEHOLDER: 0\.\w+[^\]]*\]( \| )\[PLACEHOLDER: Raw/Domain\]'
            def replacer(m):
                return f"{m.group(1)}{feat}{m.group(2)}{imp:.4f}{m.group(3)}Raw"
            content = re.sub(pattern, replacer, content)
            
            # Also try without the last column
            pattern2 = rf'(\| {rank} \| )\[PLACEHOLDER: feature_name\]( \| )\[PLACEHOLDER: 0\.\w+[^\]]*\]'
            content = re.sub(pattern2, lambda m: f"{m.group(1)}{feat}{m.group(2)}{imp:.4f}", content)
    
    return content

def replace_training_time(content, data):
    """Replace training time placeholders."""
    if 'additional_metrics' not in data or 'models' not in data['additional_metrics']:
        return content
    
    models = data['additional_metrics']['models']
    
    for model_short, model_aliases in MODEL_NAMES.items():
        if model_short not in models:
            continue
        model_data = models[model_short]
        train_time = model_data.get('train_time_s_mean', 0)
        
        for alias in model_aliases:
            # | XGBoost (Raw) | [PLACEHOLDER: X.XX] | ... | [PLACEHOLDER: XX.X] | ~30 |
            pattern = rf'(\| {re.escape(alias)}[^|]*\(Raw\)[^|]*\| )\[PLACEHOLDER: X\.\w+[^\]]*\]'
            content = re.sub(pattern, lambda m: m.group(1) + f"{train_time:.2f}", content)
            
            pattern2 = rf'(\| {re.escape(alias)}[^|]*\(Domain\)[^|]*\| )\[PLACEHOLDER: X\.\w+[^\]]*\]'
            content = re.sub(pattern2, lambda m: m.group(1) + f"{train_time:.2f}", content)
            
            # | XGBoost | [PLACEHOLDER: XX.X] | (training time in seconds)
            pattern3 = rf'(\| {re.escape(alias)} \| )\[PLACEHOLDER: XX\.\w+[^\]]*\]'
            content = re.sub(pattern3, lambda m: m.group(1) + f"{train_time:.1f}", content)
    
    return content

def replace_dataset_stats(content, data):
    """Replace dataset statistics placeholders."""
    if 'additional_metrics' not in data:
        return content
    
    am = data['additional_metrics']
    n_samples = am.get('n_samples', 0)
    n_features = am.get('n_features', 0)
    
    if n_samples:
        content = re.sub(r'\[PLACEHOLDER: XXX\]', str(n_samples), content)
        content = re.sub(r'\[PLACEHOLDER: \d+ samples\]', str(n_samples), content)
    
    if n_features:
        content = re.sub(r'\[PLACEHOLDER: \d+ features\]', str(n_features), content)
    
    return content

def replace_generic_placeholders(content, data):
    """Replace generic numeric placeholders."""
    if 'additional_metrics' not in data or 'models' not in data['additional_metrics']:
        return content
    
    models = data['additional_metrics']['models']
    
    # For regression: fill RMSE, MAE values
    for model_short, model_aliases in MODEL_NAMES.items():
        if model_short not in models:
            continue
        model_data = models[model_short]
        
        rmse = model_data.get('RMSE_mean', None)
        mae = model_data.get('MAE_mean', None)
        
        if rmse is not None:
            for alias in model_aliases:
                pattern = rf'(\| {re.escape(alias)}[^|]*\| )\[PLACEHOLDER: \d+\.\w+[^\]]*\]( \| )\[PLACEHOLDER: \d+\.\w+[^\]]*\]'
                def replacer(m):
                    return f"{m.group(1)}{rmse:.4f}{m.group(2)}{mae:.4f}" if mae else f"{m.group(1)}{rmse:.4f}{m.group(2)}"
                content = re.sub(pattern, replacer, content)
    
    return content

def replace_statistical_tests(content, data):
    """Replace statistical test placeholders using per-seed data."""
    if 'per_seed' not in data:
        return content
    
    per_seed = data['per_seed']
    
    for model_short, model_aliases in MODEL_NAMES.items():
        if model_short not in per_seed.get('per_seed', {}):
            continue
        
        # Get Raw and Domain scores
        raw_scores = per_seed['per_seed'][model_short].get('Raw', {}).get('scores', [])
        dom_scores = per_seed['per_seed'][model_short].get('Domain', {}).get('scores', [])
        
        if not raw_scores or not dom_scores:
            continue
        
        raw_arr = np.array(raw_scores)
        dom_arr = np.array(dom_scores)
        diff = dom_arr - raw_arr
        
        # Paired t-test
        t_stat, t_p = scipy_stats.ttest_rel(dom_arr, raw_arr)
        
        # Cohen's d
        pooled_std = np.sqrt((np.std(raw_arr, ddof=1)**2 + np.std(dom_arr, ddof=1)**2) / 2)
        d_val = (np.mean(dom_arr) - np.mean(raw_arr)) / pooled_std if pooled_std > 0 else 0
        
        # 95% CI
        mean_diff = np.mean(diff)
        se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
        ci_lo = mean_diff - 1.96 * se_diff
        ci_hi = mean_diff + 1.96 * se_diff
        
        for alias in model_aliases:
            # t-statistic
            pattern = rf'(\| {re.escape(alias)}[^|]*\| )\[PLACEHOLDER: t=[^\]]*\]'
            content = re.sub(pattern, lambda m: m.group(1) + f"t={t_stat:.2f}", content)
            
            # p-value
            pattern = rf'(\| {re.escape(alias)}[^|]*\|[^|]*\|[^|]*\|[^|]*\| )\[PLACEHOLDER: p=[^\]]*\]'
            content = re.sub(pattern, lambda m: m.group(1) + f"p={t_p:.3f}", content)
            
            # Cohen's d
            pattern = rf'(\| {re.escape(alias)}[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\| )\[PLACEHOLDER: d=[^\]]*\]'
            content = re.sub(pattern, lambda m: m.group(1) + f"d={d_val:.2f}", content)
            
            # Mean difference
            pattern = rf'(\| {re.escape(alias)}[^|]*\| )\[PLACEHOLDER: [+-]?0\.\w+[^\]]*\]'
            content = re.sub(pattern, lambda m: m.group(1) + f"{mean_diff:+.4f}", content)
            
            # CI
            hw = (ci_hi - ci_lo) / 2
            pattern = rf'(\| {re.escape(alias)}[^|]*\|[^|]*\|[^|]*\|[^|]*\| )\[PLACEHOLDER: \$\\pm\$0\.\w+[^\]]*\]'
            content = re.sub(pattern, lambda m: m.group(1) + f"$\\pm${hw:.4f}", content)
    
    return content

def replace_descriptive_placeholders(content, data):
    """Replace descriptive PLACEHOLDER text with 'N/A - see results files'."""
    # Pattern: [PLACEHOLDER: descriptive text]
    # Replace with 'N/A' for non-numeric placeholders
    pattern = r'\[PLACEHOLDER: [A-Z][a-zA-Z\s\-\'\(\),\.]+\]'
    content = re.sub(pattern, 'N/A (see results files)', content)
    
    return content

def replace_paper(direction):
    paper_path = BASE / direction / 'paper' / 'paper_draft.md'
    if not paper_path.exists():
        return 0, 0
    
    data = load_all_results(direction)
    if not data:
        return 0, 0
    
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    orig_count = content.count('PLACEHOLDER')
    if orig_count == 0:
        return 0, 0
    
    # Apply all replacement functions
    content = replace_multi_metric_table(content, data)
    content = replace_feature_importance(content, data)
    content = replace_training_time(content, data)
    content = replace_dataset_stats(content, data)
    content = replace_generic_placeholders(content, data)
    content = replace_statistical_tests(content, data)
    
    # Final: replace remaining descriptive placeholders
    content = replace_descriptive_placeholders(content, data)
    
    with open(paper_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    new_count = content.count('PLACEHOLDER')
    return orig_count, new_count

# Main
print(f"\n{'='*60}")
print(f"Comprehensive PLACEHOLDER replacement v2")
print(f"{'='*60}\n")

total_orig = 0
total_new = 0
for d in DIRECTIONS:
    try:
        orig, new = replace_paper(d)
        replaced = orig - new
        total_orig += orig
        total_new += new
        if orig > 0:
            print(f"  {d}: {orig} -> {new} (replaced {replaced})")
    except Exception as e:
        print(f"  {d}: ERROR - {e}")

print(f"\n{'='*60}")
print(f"Total: {total_orig} -> {total_new} (replaced {total_orig - total_new})")
print(f"{'='*60}")
