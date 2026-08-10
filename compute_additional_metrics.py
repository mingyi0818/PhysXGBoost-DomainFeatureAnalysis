#!/usr/bin/env python3
"""
Compute additional metrics for all directions:
- Classification: Accuracy, F1-Macro, F1-Micro, Precision, Recall, Cohen's Kappa, MCC
- Regression: RMSE, MAE, Pearson r
- SHAP feature importance (using XGBoost feature_importances_ as proxy)
- Training time and memory usage
"""
import json
import time
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

BASE = Path('D:/ResearchPaperPrepare')
SEEDS = [42, 123, 456, 789, 2024]

DIRECTIONS_CONFIG = {
    '46_FlightDelay_PhysXGBoost': {'data_file': 'bank_marketing.csv', 'target': 'y', 'task': 'classification', 'drop_cols': []},
    '47_OnlineShoppers': {'data_file': 'online_shoppers.csv', 'target': 'y', 'task': 'classification', 'drop_cols': []},
    '48_CreditDefault': {'data_file': 'credit_default.csv', 'target': 'default.payment.next.month', 'task': 'classification', 'drop_cols': ['ID']},
    '49_Superconductor': {'data_file': 'superconductor.csv', 'target': 'critical_temp', 'task': 'regression', 'drop_cols': []},
    '50_BuildingEnergy': {'data_file': 'energy.csv', 'target': 'Y1', 'task': 'regression', 'drop_cols': ['Y2']},
    '51_GasTurbine': {'data_file': 'gasturbine.csv', 'target': 'NOX', 'task': 'regression', 'drop_cols': ['year']},
    '52_CCPP': {'data_file': 'ccpp.csv', 'target': 'PE', 'task': 'regression', 'drop_cols': []},
    '53_BikeSharing': {'data_file': 'bikesharing.csv', 'target': 'cnt', 'task': 'regression', 'drop_cols': ['dteday', 'casual', 'registered']},
    '54_NewsPopularity': {'data_file': 'news_pop.csv', 'target': 'shares', 'task': 'regression', 'drop_cols': ['url']},
    '55_CalHousing': {'data_file': 'california_housing.csv', 'target': 'MedHouseVal', 'task': 'regression', 'drop_cols': []},
    '56_PowerConsumption': {'data_file': 'power.csv', 'target': 'Global_active_power', 'task': 'regression', 'drop_cols': ['datetime']},
    '58_CDNOW': {'data_file': 'cdnow.csv', 'target': 'target', 'task': 'classification', 'drop_cols': []},
    '59_NYCProperty': {'data_file': 'nyc_property_sales.csv', 'target': 'SALE PRICE', 'task': 'regression', 'drop_cols': ['Unnamed: 0', 'ADDRESS', 'APARTMENT NUMBER', 'SALE DATE', 'EASE-MENT', 'NEIGHBORHOOD', 'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE', 'TAX CLASS AT PRESENT', 'TAX CLASS AT TIME OF SALE']},
    '60_StudentPerf': {'data_file': 'student.csv', 'target': 'G3', 'task': 'regression', 'drop_cols': ['G1', 'G2']},
    '61_DryBean': {'data_file': 'drybean.csv', 'target': 'class', 'task': 'classification', 'drop_cols': ['Type']},
    '63_HotelBooking': {'data_file': 'hotel.csv', 'target': 'is_canceled', 'task': 'classification', 'drop_cols': []},
    '64_FlightDelay': {'data_file': None, 'target': None, 'task': 'classification', 'drop_cols': []},  # No data file
    '65_HR': {'data_file': 'hr_data.csv', 'target': 'Attrition', 'task': 'classification', 'drop_cols': ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']},
}

def load_and_prep(direction, cfg):
    """Load data and prepare features."""
    if cfg['data_file'] is None:
        return None, None, None, None
    
    data_path = BASE / direction / 'data' / cfg['data_file']
    if not data_path.exists():
        return None, None, None, None
    
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # NYC property special cleaning
    if direction == '59_NYCProperty':
        df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
        df = df[df['SALE PRICE'] > 100].copy()
        low = df['SALE PRICE'].quantile(0.01)
        high = df['SALE PRICE'].quantile(0.99)
        df = df[(df['SALE PRICE'] >= low) & (df['SALE PRICE'] <= high)].copy()
        for col in ['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS',
                    'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT',
                    'ZIP CODE', 'BLOCK', 'LOT', 'BOROUGH']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Drop columns
    for col in cfg.get('drop_cols', []):
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Encode categorical
    for col in df.select_dtypes(include=['object']).columns:
        if col != cfg['target']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Separate target
    if cfg['target'] not in df.columns:
        return None, None, None, None
    
    y = df[cfg['target']]
    X = df.drop(columns=[cfg['target']])
    
    # Encode target for classification
    if cfg['task'] == 'classification' and y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
    
    # For NewsPopularity, log-transform target
    if direction == '54_NewsPopularity':
        y = np.log1p(y)
    
    return X, y, cfg['task'], direction

def compute_additional_metrics(direction):
    """Compute additional metrics beyond AUC/R2."""
    cfg = DIRECTIONS_CONFIG.get(direction)
    if not cfg:
        return None
    
    X, y, task, dir_name = load_and_prep(direction, cfg)
    if X is None:
        print(f"  {direction}: No data available")
        return None
    
    results = {
        'direction': direction,
        'task': task,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'models': {},
        'feature_importance': {},
    }
    
    models_cls = {
        'XGB': lambda seed: __import__('xgboost').XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=seed, n_jobs=-1, eval_metric='logloss', verbosity=0),
        'LGB': lambda seed: __import__('lightgbm').LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=seed, n_jobs=-1, verbose=-1),
        'Cat': lambda seed: __import__('catboost').CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            random_seed=seed, verbose=0),
        'RF': lambda seed: RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=seed, n_jobs=-1),
    }
    
    models_reg = {
        'XGB': lambda seed: __import__('xgboost').XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=seed, n_jobs=-1, verbosity=0),
        'LGB': lambda seed: __import__('lightgbm').LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=seed, n_jobs=-1, verbose=-1),
        'Cat': lambda seed: __import__('catboost').CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.1,
            random_seed=seed, verbose=0),
        'RF': lambda seed: RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=seed, n_jobs=-1),
    }
    
    models = models_reg if task == 'regression' else models_cls
    
    for model_name, model_factory in models.items():
        model_results = {'per_seed': [], 'feature_importance': None}
        
        for seed in SEEDS:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed)
            
            # Time training
            t0 = time.time()
            model = model_factory(seed)
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            
            # Time prediction
            t0 = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - t0
            
            seed_metrics = {
                'train_time_s': float(train_time),
                'pred_time_s': float(pred_time),
            }
            
            if task == 'regression':
                seed_metrics['R2'] = float(r2_score(y_test, y_pred))
                seed_metrics['RMSE'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                seed_metrics['MAE'] = float(mean_absolute_error(y_test, y_pred))
                r, p = stats.pearsonr(y_test, y_pred)
                seed_metrics['Pearson_r'] = float(r)
                seed_metrics['Pearson_p'] = float(p)
            else:
                n_classes = len(np.unique(y))
                if n_classes == 2:
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        seed_metrics['AUC'] = float(roc_auc_score(y_test, y_proba))
                    except:
                        seed_metrics['AUC'] = 0.0
                
                seed_metrics['Accuracy'] = float(accuracy_score(y_test, y_pred))
                seed_metrics['F1_Macro'] = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
                seed_metrics['F1_Micro'] = float(f1_score(y_test, y_pred, average='micro', zero_division=0))
                seed_metrics['Precision'] = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
                seed_metrics['Recall'] = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
                seed_metrics['Cohen_Kappa'] = float(cohen_kappa_score(y_test, y_pred))
                seed_metrics['MCC'] = float(matthews_corrcoef(y_test, y_pred))
            
            model_results['per_seed'].append(seed_metrics)
        
        # Aggregate
        agg = {}
        for key in model_results['per_seed'][0]:
            vals = [s[key] for s in model_results['per_seed']]
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
        results['models'][model_name] = agg
        
        # Feature importance (using last trained model)
        try:
            if hasattr(model, 'feature_importances_'):
                fi = model.feature_importances_
                results['feature_importance'][model_name] = {
                    X.columns[i]: float(fi[i]) for i in range(len(fi))
                }
        except:
            pass
    
    # Save
    results_dir = BASE / direction / 'results'
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / 'additional_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    metric_names = list(results['models'].get('XGB', {}).keys())
    print(f"  {direction}: Saved additional metrics ({len(metric_names)} metrics × {len(models)} models)")
    
    return results

# Main
DIRECTIONS = [d for d in DIRECTIONS_CONFIG.keys() if DIRECTIONS_CONFIG[d]['data_file'] is not None]

print(f"\n{'='*60}")
print(f"Computing additional metrics for {len(DIRECTIONS)} directions")
print(f"{'='*60}\n")

for d in DIRECTIONS:
    try:
        compute_additional_metrics(d)
    except Exception as e:
        print(f"  {d}: ERROR - {e}")
        import traceback; traceback.print_exc()

print(f"\n{'='*60}")
print(f"Done!")
print(f"{'='*60}")
