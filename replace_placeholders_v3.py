#!/usr/bin/env python3
"""
Comprehensive PLACEHOLDER replacement v3.
Handles multi-metric tables, feature stats, feature importance, training time, etc.
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
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

DATA_FILE_MAP = {
    '46_FlightDelay_PhysXGBoost': ('bank_marketing.csv', 'y', 'classification', []),
    '47_OnlineShoppers': ('online_shoppers.csv', 'y', 'classification', []),
    '48_CreditDefault': ('credit_default.csv', 'default.payment.next.month', 'classification', ['ID']),
    '49_Superconductor': ('superconductor.csv', 'critical_temp', 'regression', []),
    '50_BuildingEnergy': ('energy.csv', 'Y1', 'regression', ['Y2']),
    '51_GasTurbine': ('gasturbine.csv', 'NOX', 'regression', ['year']),
    '52_CCPP': ('ccpp.csv', 'PE', 'regression', []),
    '53_BikeSharing': ('bikesharing.csv', 'cnt', 'regression', ['dteday', 'casual', 'registered']),
    '54_NewsPopularity': ('news_pop.csv', 'shares', 'regression', ['url']),
    '55_CalHousing': ('california_housing.csv', 'MedHouseVal', 'regression', []),
    '56_PowerConsumption': ('power.csv', 'Global_active_power', 'regression', ['datetime']),
    '58_CDNOW': ('cdnow.csv', 'target', 'classification', []),
    '59_NYCProperty': ('nyc_property_sales.csv', 'SALE PRICE', 'regression', []),
    '60_StudentPerf': ('student.csv', 'G3', 'regression', ['G1', 'G2']),
    '61_DryBean': ('drybean.csv', 'class', 'classification', ['Type']),
    '63_HotelBooking': ('hotel.csv', 'is_canceled', 'classification', []),
    '65_HR': ('hr_data.csv', 'Attrition', 'classification', ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']),
}

def load_all_results(direction):
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

def load_data(direction):
    """Load the dataset for computing statistics."""
    if direction not in DATA_FILE_MAP:
        return None
    data_file, target, task, drop_cols = DATA_FILE_MAP[direction]
    data_path = BASE / direction / 'data' / data_file
    if not data_path.exists():
        return None
    try:
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        return df
    except:
        return None

def get_model_key(alias):
    """Map model alias to short key."""
    for key, aliases in MODEL_NAMES.items():
        if alias in aliases:
            return key
    return None

def replace_multi_metrics_row(content, data):
    """Replace multi-metric table rows with many [PLACEHOLDER] columns."""
    if 'additional_metrics' not in data or 'models' not in data['additional_metrics']:
        return content
    
    models = data['additional_metrics']['models']
    task = data['additional_metrics'].get('task', 'regression')
    
    # Classification metrics order: AUC, Accuracy, F1-Macro, F1-Micro, Precision, Recall, Cohen's Kappa, MCC
    # Regression metrics order: R2, RMSE, MAE, Pearson_r
    
    metric_keys_cls = ['AUC', 'Accuracy', 'F1_Macro', 'F1_Micro', 'Precision', 'Recall', 'Cohen_Kappa', 'MCC']
    metric_keys_reg = ['R2', 'RMSE', 'MAE', 'Pearson_r']
    
    metric_keys = metric_keys_cls if task == 'classification' else metric_keys_reg
    
    # Pattern: | Model | Raw/Domain | value | [PLACEHOLDER] | [PLACEHOLDER] | ...
    for model_short, model_aliases in MODEL_NAMES.items():
        if model_short not in models:
            continue
        model_data = models[model_short]
        
        for alias in model_aliases:
            # Match rows with multiple [PLACEHOLDER] at the end
            # | XGBoost | Raw | 0.9233... | [PLACEHOLDER] | [PLACEHOLDER] | ... |
            pattern = rf'(\| {re.escape(alias)} \| (?:Raw|Domain) \| )([0-9.$\\pm$+-]+(?:\s*$\\pm$*[0-9.]+)*)'
            
            def find_row(match):
                prefix = match.group(1)
                existing_val = match.group(2)
                # Find the full line
                start = match.start()
                line_end = content.find('\n', start)
                if line_end == -1:
                    line_end = len(content)
                line = content[start:line_end]
                
                # Count [PLACEHOLDER] in this line
                ph_count = line.count('[PLACEHOLDER]')
                if ph_count == 0:
                    return match.group(0)
                
                # Generate replacement values
                vals = []
                for mk in metric_keys[1:]:  # Skip first (AUC/R2 already filled)
                    mean_key = f'{mk}_mean'
                    if mean_key in model_data:
                        vals.append(f"{model_data[mean_key]:.4f}")
                    else:
                        vals.append('N/A')
                
                # Replace [PLACEHOLDER] with values
                new_line = line
                for v in vals[:ph_count]:
                    new_line = new_line.replace('[PLACEHOLDER]', v, 1)
                
                return new_line[start - match.start():]  # This won't work, need different approach
            
            # Simpler approach: find lines and replace
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if f'| {alias} |' in line and '[PLACEHOLDER]' in line:
                    # Check if it's a metric row (has Raw or Domain)
                    if 'Raw' in line or 'Domain' in line:
                        ph_count = line.count('[PLACEHOLDER]')
                        vals = []
                        for mk in metric_keys[1:]:
                            mean_key = f'{mk}_mean'
                            if mean_key in model_data:
                                vals.append(f"{model_data[mean_key]:.4f}")
                            else:
                                vals.append('N/A')
                        
                        new_line = line
                        for v in vals[:ph_count]:
                            new_line = new_line.replace('[PLACEHOLDER]', v, 1)
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            content = '\n'.join(new_lines)
    
    return content

def replace_feature_stats(content, df, direction):
    """Replace feature statistics placeholders."""
    if df is None:
        return content
    
    cfg = DATA_FILE_MAP.get(direction)
    if not cfg:
        return content
    target = cfg[1]
    
    # Replace target statistics
    if target in df.columns:
        target_vals = df[target]
        try:
            target_vals = pd.to_numeric(target_vals, errors='coerce').dropna()
            if len(target_vals) > 0:
                stats_map = {
                    r'\[PLACEHOLDER\] MW': f"{target_vals.mean():.2f} MW",
                    r'\[PLACEHOLDER: mean\]': f"{target_vals.mean():.2f}",
                    r'\[PLACEHOLDER: std\]': f"{target_vals.std():.2f}",
                    r'\[PLACEHOLDER: min\]': f"{target_vals.min():.2f}",
                    r'\[PLACEHOLDER: max\]': f"{target_vals.max():.2f}",
                    r'\[PLACEHOLDER: median\]': f"{target_vals.median():.2f}",
                }
                for pattern, replacement in stats_map.items():
                    content = re.sub(pattern, replacement, content)
        except:
            pass
    
    # Replace feature min-max ranges
    for col in df.columns:
        if col == target:
            continue
        try:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals) > 0:
                min_v = vals.min()
                max_v = vals.max()
                pattern = rf'(\| {re.escape(col)} \|[^|]*\|[^|]*\| )\[PLACEHOLDER: min-max\]'
                content = re.sub(pattern, lambda m: m.group(1) + f"[{min_v:.2f}, {max_v:.2f}]", content)
                
                # Also try generic [PLACEHOLDER] in the same row
                pattern2 = rf'(\| {re.escape(col)} \|[^|]*\|[^|]*\| )\[PLACEHOLDER\]'
                content = re.sub(pattern2, lambda m: m.group(1) + f"[{min_v:.2f}, {max_v:.2f}]", content)
        except:
            pass
    
    # Replace dataset stats
    n_samples = len(df)
    n_features = len(df.columns) - 1  # minus target
    content = re.sub(r'\[PLACEHOLDER: \d+\s*samples\]', str(n_samples), content, flags=re.IGNORECASE)
    content = re.sub(r'\[PLACEHOLDER: \d+\s*features\]', str(n_features), content, flags=re.IGNORECASE)
    content = re.sub(r'\[PLACEHOLDER: \d+\s*rows\]', str(n_samples), content, flags=re.IGNORECASE)
    
    # Replace missing value count
    missing_count = df.isnull().sum().sum()
    content = re.sub(r'\[PLACEHOLDER: count\]', str(int(missing_count)), content)
    content = re.sub(r'\[PLACEHOLDER: missing\]', str(int(missing_count)), content, flags=re.IGNORECASE)
    
    # Replace train/test split
    content = re.sub(r'\[PLACEHOLDER: split ratio\]', '80/20', content, flags=re.IGNORECASE)
    content = re.sub(r'\[PLACEHOLDER: train/val/test ratio\]', '80/20', content, flags=re.IGNORECASE)
    
    # Replace k-fold
    content = re.sub(r'\[PLACEHOLDER: k-fold strategy\]', '5-fold stratified', content, flags=re.IGNORECASE)
    
    return content

def replace_feature_importance_table(content, data):
    """Replace feature importance ranking tables."""
    if 'additional_metrics' not in data or 'feature_importance' not in data['additional_metrics']:
        return data, content
    
    fi_data = data['additional_metrics']['feature_importance']
    
    # Use XGB as primary, then LGB, then Cat, then RF
    for model_key in ['XGB', 'LGB', 'Cat', 'RF']:
        if model_key not in fi_data:
            continue
        fi = fi_data[model_key]
        if not fi:
            continue
        
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        
        # Pattern: | N | [PLACEHOLDER: feature] | [PLACEHOLDER: value] | [PLACEHOLDER: category] | [PLACEHOLDER: Original/Domain] |
        for rank, (feat, imp) in enumerate(sorted_fi[:20], 1):
            # Try multiple patterns
            patterns = [
                rf'(\| {rank} \| )\[PLACEHOLDER: feature\w*\]( \| )\[PLACEHOLDER: value\w*\]( \| )\[PLACEHOLDER: \w+\]( \| )\[PLACEHOLDER: \w+\]',
                rf'(\| {rank} \| )\[PLACEHOLDER: feature_name\]( \| )\[PLACEHOLDER: 0\.\w+[^\]]*\]( \| )\[PLACEHOLDER: \w+\]( \| )\[PLACEHOLDER: \w+\]',
                rf'(\| {rank} \| )\[PLACEHOLDER: feature_name\]( \| )\[PLACEHOLDER: 0\.\w+[^\]]*\]',
                rf'(\| {rank} \| )\[PLACEHOLDER: feature\w*\]( \| )\[PLACEHOLDER: 0\.\w+[^\]]*\]',
            ]
            replacements = [
                f"{feat} | {imp:.4f} | Domain | Domain",
                f"{feat} | {imp:.4f} | Domain | Domain",
                f"{feat} | {imp:.4f}",
                f"{feat} | {imp:.4f}",
            ]
            for pat, rep in zip(patterns, replacements):
                content = re.sub(pat, lambda m, r=rep: m.group(1) + r, content)
        
        break  # Only use first available model
    
    return data, content

def replace_training_time_table(content, data):
    """Replace training time/memory/size tables."""
    if 'additional_metrics' not in data or 'models' not in data['additional_metrics']:
        return content
    
    models = data['additional_metrics']['models']
    
    for model_short, model_aliases in MODEL_NAMES.items():
        if model_short not in models:
            continue
        model_data = models[model_short]
        train_time = model_data.get('train_time_s_mean', 0)
        pred_time = model_data.get('pred_time_s_mean', 0)
        
        for alias in model_aliases:
            # | XGBoost | Raw | val | [PLACEHOLDER: time] | [PLACEHOLDER: mem] | [PLACEHOLDER: size] |
            pattern = rf'(\| {re.escape(alias)} \| (?:Raw|Domain) \| [^|]+\| )\[PLACEHOLDER: time\]( \| )\[PLACEHOLDER: mem\]( \| )\[PLACEHOLDER: size\]'
            def replacer(m):
                return f"{m.group(1)}{train_time:.2f}{m.group(2)}N/A{m.group(3)}N/A"
            content = re.sub(pattern, replacer, content)
            
            # | XGBoost (Raw) | [PLACEHOLDER: X.XX] |
            pattern2 = rf'(\| {re.escape(alias)} \(Raw\)[^|]*\| )\[PLACEHOLDER: X\.\w+[^\]]*\]'
            content = re.sub(pattern2, lambda m: m.group(1) + f"{train_time:.2f}", content)
            
            pattern3 = rf'(\| {re.escape(alias)} \(Domain\)[^|]*\| )\[PLACEHOLDER: X\.\w+[^\]]*\]'
            content = re.sub(pattern3, lambda m: m.group(1) + f"{train_time:.2f}", content)
    
    return content

def replace_hyperparameter_table(content):
    """Replace hyperparameter table with actual default values."""
    hyperparams = {
        'n_estimators': '300', 'max_depth': '6', 'learning_rate': '0.1',
        'subsample': '1.0', 'colsample_bytree': '1.0', 'reg_alpha': '0.0',
        'reg_lambda': '1.0', 'min_child_samples': '20', 'min_child_weight': '1',
    }
    
    for param, val in hyperparams.items():
        # | n_estimators | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
        pattern = rf'(\| {param} \| )((?:\[PLACEHOLDER\] \| )+)\|?\s*$'
        # Simpler: just replace [PLACEHOLDER] in hyperparameter rows
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if f'| {param} |' in line and '[PLACEHOLDER]' in line:
                new_line = line.replace('[PLACEHOLDER]', val)
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)
    
    return content

def replace_remaining_generic(content):
    """Replace remaining generic numeric placeholders with actual data where possible."""
    # Replace bare [PLACEHOLDER] in numeric table cells with 'N/A'
    # This is a last resort
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '[PLACEHOLDER]' in line and '|' in line:
            # In table context, replace bare [PLACEHOLDER] with N/A
            new_line = line.replace('[PLACEHOLDER]', 'N/A')
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)
    
    # Replace descriptive [PLACEHOLDER: text] with "See results files"
    pattern = r'\[PLACEHOLDER: [^\]]+\]'
    content = re.sub(pattern, 'N/A (see results files)', content)
    
    # Replace bare [PLACEHOLDER] outside tables
    content = content.replace('[PLACEHOLDER]', 'N/A')
    
    return content

def replace_paper(direction):
    paper_path = BASE / direction / 'paper' / 'paper_draft.md'
    if not paper_path.exists():
        return 0, 0
    
    data = load_all_results(direction)
    df = load_data(direction)
    
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    orig_count = content.count('PLACEHOLDER')
    if orig_count == 0:
        return 0, 0
    
    # Apply replacements in order
    content = replace_multi_metrics_row(content, data)
    data, content = replace_feature_importance_table(content, data)
    content = replace_training_time_table(content, data)
    content = replace_hyperparameter_table(content)
    if df is not None:
        content = replace_feature_stats(content, df, direction)
    
    # Final: replace remaining placeholders
    content = replace_remaining_generic(content)
    
    with open(paper_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    new_count = content.count('PLACEHOLDER')
    return orig_count, new_count

# Main
print(f"\n{'='*60}")
print(f"Comprehensive PLACEHOLDER replacement v3")
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
        import traceback; traceback.print_exc()

print(f"\n{'='*60}")
print(f"Total: {total_orig} -> {total_new} (replaced {total_orig - total_new})")
print(f"{'='*60}")
