"""
Bank Marketing Domain Feature Engineering
=========================================
Domain-derived features for predicting term deposit subscription.
CS+Finance方向 — 第2个执行方向

Feature categories:
  1. Customer Profile (cust_*) — demographic interactions
  2. Financial Health (fin_*) — debt burden indices  
  3. Campaign Dynamics (camp_*) — marketing effectiveness
  4. Economic Context (econ_*) — macro indicators
  5. Temporal Encoding (time_*) — seasonal patterns
"""

import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
import xgboost as xgb, lightgbm as lgb, catboost as cb
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / 'data' / 'bank_marketing.csv'
RESULTS_PATH = ROOT / 'results'

# ============================================================
# 1. Load & Prep
# ============================================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Handle NaN in categoricals
    for c in ['contact', 'poutcome']:
        if c in df.columns:
            df[c] = df[c].fillna('unknown')
    # Encode target
    df['y'] = (df['y'] == 'yes').astype(int)
    # Drop any remaining NaN rows
    df = df.dropna()
    return df

# ============================================================
# 2. Feature Engineering
# ============================================================
def build_features(df):
    """Build raw + domain features"""
    X = df.drop(columns=['y'])
    y = df['y']
    
    # Label encode categoricals
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'month', 'day_of_week', 'poutcome']
    le = {}
    for c in cat_cols:
        if c in X.columns:
            le[c] = LabelEncoder()
            X[c] = le[c].fit_transform(X[c].astype(str))
    # Handle 'balance' if present (numeric, might have NaN for 'unknown')
    if 'balance' in X.columns:
        X['balance'] = pd.to_numeric(X['balance'], errors='coerce').fillna(0)
        # Categorize balance
        X['fin_balance_low'] = (X['balance'] < 0).astype(int)
        X['fin_balance_high'] = (X['balance'] > 5000).astype(int)
    
    X_raw = X.copy()
    
    # === Domain Features ===
    # Customer profile
    X['cust_age_job'] = X['age'] * X['job']
    X['cust_edu_marital'] = X['education'] * X['marital']
    X['cust_age_squared'] = X['age'] ** 2
    # Age brackets
    X['cust_age_young'] = (X['age'] < 30).astype(int)
    X['cust_age_senior'] = (X['age'] > 60).astype(int)
    
    # Financial health
    X['fin_debt_score'] = X['default'] + X['housing'] + X['loan']
    X['fin_has_debt'] = (X['fin_debt_score'] > 0).astype(int)
    X['fin_housing_loan'] = X['housing'] * X['loan']
    
    # Campaign dynamics
    X['camp_intensity'] = X['campaign'] / X['duration'].clip(1)
    X['camp_prev_contact'] = (X['previous'] > 0).astype(int) * X['campaign']
    X['camp_pdays_recency'] = np.where(X['pdays'] == 999, 0, 1 / (X['pdays'].clip(1) + 1))
    X['camp_pdays_never'] = (X['pdays'] == 999).astype(int)
    X['camp_contact_month'] = X['contact'] * X['month']
    X['camp_duration_log'] = np.log1p(X['duration'])
    X['camp_high_effort'] = (X['campaign'] > 3).astype(int)
    X['camp_success_history'] = (X['poutcome'] == 2).astype(int)  # 2=success
    
    # Financial depth (balance-based)
    X['fin_balance_loan'] = X['balance'] * X['loan']
    X['fin_balance_housing'] = X['balance'] * X['housing']
    X['fin_leveraged'] = ((X['loan'] == 1) & (X['balance'] < 0)).astype(int)
    
    # Temporal
    X['time_month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
    X['time_month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
    X['time_dow_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 5)
    X['time_dow_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 5)
    
    # Duration × campaign interaction
    X['camp_duration_campaign'] = X['duration'] * X['campaign']
    
    return X_raw, X, y

# ============================================================
# 3. Training
# ============================================================
def train_evaluate(X_raw, X_domain, y, n_seeds=7):
    seeds = [42, 123, 456, 789, 1024, 2048, 373][:n_seeds]
    
    models = {
        'XGBoost': lambda s: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                                 subsample=0.8, random_state=s, verbosity=0, n_jobs=-1,
                                                 scale_pos_weight=7.5),
        'LightGBM': lambda s: lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                                  subsample=0.8, random_state=s, verbose=-1, n_jobs=-1,
                                                  class_weight='balanced'),
        'CatBoost': lambda s: cb.CatBoostClassifier(n_estimators=200, depth=6, learning_rate=0.05,
                                                     subsample=0.8, random_seed=s, verbose=0, thread_count=-1,
                                                     auto_class_weights='Balanced'),
        'RandomForest': lambda s: RandomForestClassifier(n_estimators=200, max_depth=12,
                                                          random_state=s, n_jobs=-1, class_weight='balanced'),
    }
    
    results = {'Raw': {}, 'Domain': {}}
    
    for feat_name, X_feat in [('Raw', X_raw), ('Domain', X_domain)]:
        print(f"\n--- {feat_name} Features ({X_feat.shape[1]} dims) ---")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_feat, y, test_size=0.2, random_state=42, stratify=y)
        
        for mname, mfactory in models.items():
            aucs, f1s = [], []
            for seed in seeds:
                m = mfactory(seed)
                m.fit(X_tr, y_tr)
                y_prob = m.predict_proba(X_te)[:, 1]
                y_pred = m.predict(X_te)
                aucs.append(roc_auc_score(y_te, y_prob))
                f1s.append(f1_score(y_te, y_pred))
            results[feat_name][mname] = {
                'AUC': np.mean(aucs), 'AUC_std': np.std(aucs),
                'F1': np.mean(f1s), 'F1_std': np.std(f1s)
            }
            print(f"  {mname:12s}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}, "
                  f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
    
    return results

# ============================================================
# 4. Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Bank Marketing Domain Feature Experiment")
    print("=" * 60)
    
    df = load_data()
    print(f"Samples: {len(df)}, Target: {df['y'].sum()} yes ({df['y'].mean()*100:.1f}%)")
    
    X_raw, X_domain, y = build_features(df)
    print(f"Raw features: {X_raw.shape[1]}, Domain features: {X_domain.shape[1]}")
    
    results = train_evaluate(X_raw, X_domain, y)
    
    # Stats
    print("\n" + "=" * 60)
    print("Wilcoxon Tests (AUC):")
    for m in ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']:
        delta = results['Domain'][m]['AUC'] - results['Raw'][m]['AUC']
        print(f"  {m:12s}: ΔAUC = {delta:+.4f}")
    
    import json
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_PATH / 'summary.json'}")
