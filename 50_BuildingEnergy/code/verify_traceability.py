"""Verify that every number in paper_draft.md traces back to a result file."""
import json

s = json.load(open('../results/summary.json'))
fi = json.load(open('../results/feature_importance_share.json'))

print('=== Data traceability verification ===\n')

print('--- Table 2 R2 values (3 decimal) ---')
for fs in ['Raw', 'Domain']:
    for m in ['XGB', 'LGB', 'Cat', 'RF']:
        r2 = s[fs][m]['R2']
        std = s[fs][m]['std']
        print(f'  {fs:6} {m:3}: R2={r2:.3f} (raw={r2:.4f})  '
              f'std={std:.3f} (raw={std:.4f})')

print('\n--- Delta R2 (percentage points, paper Table 2 last col) ---')
for m in ['XGB', 'LGB', 'Cat', 'RF']:
    d = (s['Domain'][m]['R2'] - s['Raw'][m]['R2']) * 100
    print(f'  {m:3}: +{d:.1f}pp  (raw={d:.2f})')

print('\n--- CatBoost relative improvement (paper: 11.5%) ---')
rel = ((s['Domain']['Cat']['R2'] - s['Raw']['Cat']['R2'])
       / s['Raw']['Cat']['R2'] * 100)
print(f'  CatBoost: {rel:.1f}%  (raw={rel:.2f})')

print('\n--- Wilcoxon p-values (paper: 0.0078) ---')
for m in ['XGB', 'LGB', 'Cat', 'RF']:
    p = s['wilcoxon'][m]['p_value']
    print(f'  {m:3}: p={p}  -> paper 0.0078')

print('\n--- Feature importance (paper: 52% top-3 share) ---')
share = fi['seed42_actual_top3_share']
print(f'  seed42 top3 share: {share:.4f} -> paper 52%')
print(f'  top3 names: {fi["seed42_actual_top3"]}')

print('\n--- Feature importance percentages ---')
mfi = fi['mean_feature_importance']
total = sum(mfi.values())
checks = [
    ('enthalpy_indoor', '5--6%'),
    ('THI_indoor', '5--6%'),
    ('stack_effect', '3--5%'),
    ('wind_chill', '3--5%'),
]
for k, claim in checks:
    pct = mfi[k] / total * 100
    print(f'  {k:20}: {pct:.1f}%  (paper claim: {claim})')

print('\n--- R2 range (paper: 0.34--0.49) ---')
all_raw = [s['Raw'][m]['R2'] for m in ['XGB', 'LGB', 'Cat', 'RF']]
all_dom = [s['Domain'][m]['R2'] for m in ['XGB', 'LGB', 'Cat', 'RF']]
print(f'  min Raw: {min(all_raw):.3f}  max Domain: {max(all_dom):.3f}')
print(f'  paper: 0.34--0.49')

print('\n=== Verification complete ===')
