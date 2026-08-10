#!/usr/bin/env python3
"""
Replace PLACEHOLDERs in paper drafts with real experimental data.
Fixed version using proper closure-based replacements.
"""
import json
import re
from pathlib import Path
import numpy as np

BASE = Path('D:/ResearchPaperPrepare')

MODEL_MAP = {
    'XGBoost': 'XGB', 'XGB': 'XGB',
    'LightGBM': 'LGB', 'LGB': 'LGB',
    'CatBoost': 'Cat', 'Cat': 'Cat',
    'Random Forest': 'RF', 'RandomForest': 'RF', 'RF': 'RF',
}

def load_results(direction):
    results_dir = BASE / direction / 'results'
    data = {}
    for fname, key in [('summary.json', 'summary'), ('comprehensive_results.json', 'comprehensive'), ('per_seed_results.json', 'per_seed')]:
        path = results_dir / fname
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)
    return data

def get_metric(data, model_key, fs):
    if 'summary' not in data or fs not in data['summary'] or model_key not in data['summary'][fs]:
        return None, None
    md = data['summary'][fs][model_key]
    for m in ['R2', 'AUC', 'F1']:
        if m in md:
            return md[m], md.get('std', 0)
    return None, None

def get_stat(data, model_key):
    if 'comprehensive' not in data or 'statistical_tests' not in data['comprehensive']:
        return None
    return data['comprehensive']['statistical_tests'].get(model_key)

def sub_with_prefix(content, pattern, suffix):
    """Replace pattern, preserving the first capture group (prefix) and appending suffix."""
    def replacer(m):
        return m.group(1) + suffix
    return re.sub(pattern, replacer, content)

def replace_paper_placeholders(direction):
    paper_path = BASE / direction / 'paper' / 'paper_draft.md'
    if not paper_path.exists():
        print(f"  No paper for {direction}")
        return 0
    
    data = load_results(direction)
    if not data:
        print(f"  No results for {direction}")
        return 0
    
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    orig = content.count('PLACEHOLDER')
    if orig == 0:
        print(f"  {direction}: 0 PLACEHOLDERs")
        return 0
    
    # Detect metric
    metric = 'R2'
    if 'summary' in data and 'Raw' in data['summary']:
        for v in data['summary']['Raw'].values():
            if 'AUC' in v:
                metric = 'AUC'; break
            elif 'F1' in v:
                metric = 'F1'; break
    
    # Pattern 1: Model performance rows: | Model | Raw/Domain | [PLACEHOLDER...] |
    for pm, rm in MODEL_MAP.items():
        raw_val, raw_std = get_metric(data, rm, 'Raw')
        dom_val, dom_std = get_metric(data, rm, 'Domain')
        
        if raw_val is not None:
            raw_latex = f"{raw_val:.4f}$\\pm${raw_std:.4f}" if raw_std else f"{raw_val:.4f}"
            pattern = rf'(\| {re.escape(pm)} \| Raw \| )\[PLACEHOLDER[^\]]*\]'
            content = sub_with_prefix(content, pattern, raw_latex)
        
        if dom_val is not None:
            dom_latex = f"{dom_val:.4f}$\\pm${dom_std:.4f}" if dom_std else f"{dom_val:.4f}"
            pattern = rf'(\| {re.escape(pm)} \| Domain \| )\[PLACEHOLDER[^\]]*\]'
            content = sub_with_prefix(content, pattern, dom_latex)
    
    # Pattern 2: Single value [PLACEHOLDER: 0.7XX] after model name
    for pm, rm in MODEL_MAP.items():
        raw_val, _ = get_metric(data, rm, 'Raw')
        if raw_val is not None:
            pattern = rf'(\| {re.escape(pm)} \| )\[PLACEHOLDER: 0\.\d[^\]]*\]'
            content = sub_with_prefix(content, pattern, f"{raw_val:.4f}")
    
    # Pattern 3: Statistical tests
    for pm, rm in MODEL_MAP.items():
        stat = get_stat(data, rm)
        if not stat:
            continue
        
        t_stat = stat.get('ttest_statistic')
        t_p = stat.get('ttest_p_value')
        d_val = stat.get('cohens_d')
        md_val = stat.get('mean_diff')
        ci_lo = stat.get('ci_95_lower')
        ci_hi = stat.get('ci_95_upper')
        
        if t_stat is not None:
            pattern = rf'(\| {re.escape(pm)} \| )\[PLACEHOLDER: t=[^\]]*\]'
            content = sub_with_prefix(content, pattern, f"t={t_stat:.2f}")
        
        if t_p is not None:
            pattern = rf'(\| {re.escape(pm)} \|[^|]*\|[^|]*\|[^|]*\| )\[PLACEHOLDER: p=[^\]]*\]'
            content = sub_with_prefix(content, pattern, f"p={t_p:.3f}")
            pattern2 = rf'({re.escape(pm)}[^|]*\|[^|]*\| )\[PLACEHOLDER: p=[^\]]*\]'
            content = sub_with_prefix(content, pattern2, f"p={t_p:.3f}")
        
        if d_val is not None:
            pattern = rf'(\| {re.escape(pm)} \|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\| )\[PLACEHOLDER: d=[^\]]*\]'
            content = sub_with_prefix(content, pattern, f"d={d_val:.2f}")
            pattern2 = rf'(\| {re.escape(pm)} \|[^|]*\|[^|]*\| )\[PLACEHOLDER: d=[^\]]*\]'
            content = sub_with_prefix(content, pattern2, f"d={d_val:.2f}")
        
        if md_val is not None:
            pattern = rf'(\| {re.escape(pm)} \|[^|]*\| )\[PLACEHOLDER: [+-]?0\.0[^\]]*\]'
            content = sub_with_prefix(content, pattern, f"{md_val:+.4f}")
        
        if ci_lo is not None and ci_hi is not None:
            hw = (ci_hi - ci_lo) / 2
            pattern = rf'(\| {re.escape(pm)} \|[^|]*\|[^|]*\|[^|]*\| )\[PLACEHOLDER: \$\\pm\$0\.0[^\]]*\]'
            content = sub_with_prefix(content, pattern, f"$\\pm${hw:.4f}")
    
    # Pattern 4: Wilcoxon p-values in summary table
    if 'summary' in data and 'wilcoxon' in data['summary']:
        wilcoxon = data['summary']['wilcoxon']
        for pm, rm in MODEL_MAP.items():
            if rm in wilcoxon:
                p_val = wilcoxon[rm].get('p_value')
                if p_val is not None:
                    pattern = rf'({re.escape(pm)}[^|]*\|[^|]*\|)\s*\[PLACEHOLDER[^\]]*\]'
                    content = sub_with_prefix(content, pattern, f" {p_val:.4f}")
    
    # Pattern 5: Dataset stats
    if 'comprehensive' in data:
        n = data['comprehensive'].get('n_samples')
        if n:
            content = re.sub(r'\[PLACEHOLDER: exact count\]', str(n), content)
            content = re.sub(r'\[PLACEHOLDER: exact ratio\]', str(n), content)
    
    # Pattern 6: Ablation results
    if 'comprehensive' in data and 'ablation' in data['comprehensive']:
        ablation = data['comprehensive']['ablation']
        for feat, vals in ablation.items():
            mean_v = vals.get('mean', 0)
            std_v = vals.get('std', 0)
            # Try multiple formats
            for fmt in [f"{mean_v:.4f}$\\pm${std_v:.4f}", f"{mean_v:.4f}±{std_v:.4f}"]:
                pattern = rf'(\| {re.escape(feat)} \| )\[PLACEHOLDER[^\]]*\]'
                content = sub_with_prefix(content, pattern, fmt)
    
    # Pattern 7: Generic metric rows - AUC-ROC, Accuracy, etc.
    for pm, rm in MODEL_MAP.items():
        raw_val, _ = get_metric(data, rm, 'Raw')
        if raw_val is not None:
            pattern = rf'({re.escape(pm)}[^|]*\| )\[PLACEHOLDER: 0\.\d[^\]]*\]'
            content = sub_with_prefix(content, pattern, f"{raw_val:.4f}")
    
    # Pattern 8: Feature counts
    if 'comprehensive' in data:
        n_raw = data['comprehensive'].get('n_raw_features')
        n_dom = data['comprehensive'].get('n_domain_features')
        n_total = data['comprehensive'].get('n_total_features')
        if n_raw:
            content = re.sub(r'\[PLACEHOLDER: (\d+)\s*raw\s*features\]', str(n_raw), content, flags=re.IGNORECASE)
        if n_dom:
            content = re.sub(r'\[PLACEHOLDER: (\d+)\s*domain\s*features\]', str(n_dom), content, flags=re.IGNORECASE)
        if n_total:
            content = re.sub(r'\[PLACEHOLDER: (\d+)\s*total\s*features\]', str(n_total), content, flags=re.IGNORECASE)
    
    # Write back
    with open(paper_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    new_count = content.count('PLACEHOLDER')
    replaced = orig - new_count
    print(f"  {direction}: {orig} -> {new_count} (replaced {replaced})")
    return replaced

DIRECTIONS = [
    '46_FlightDelay_PhysXGBoost', '47_OnlineShoppers', '48_CreditDefault',
    '49_Superconductor', '50_BuildingEnergy', '51_GasTurbine', '52_CCPP',
    '53_BikeSharing', '54_NewsPopularity', '55_CalHousing', '56_PowerConsumption',
    '58_CDNOW', '59_NYCProperty', '60_StudentPerf', '61_DryBean',
    '63_HotelBooking', '64_FlightDelay', '65_HR'
]

print(f"\n{'='*60}")
print(f"Replacing PLACEHOLDERs with real experimental data")
print(f"{'='*60}\n")

total = 0
for d in DIRECTIONS:
    try:
        total += replace_paper_placeholders(d)
    except Exception as e:
        print(f"  {d}: ERROR - {e}")

print(f"\n{'='*60}")
print(f"Total replaced: {total}")
print(f"{'='*60}")
