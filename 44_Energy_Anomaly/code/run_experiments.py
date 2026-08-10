"""Main experiment runner for TCR-AD: full pipeline."""

import torch
import numpy as np
import os
import sys
import json
import csv
import time
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from models import TCRAD, Autoencoder, VAE, DAGMM
from data_loader import load_sgcc_data, create_data_loaders
from train import (
    set_seed, train_tcrad, train_ae, train_vae, train_dagmm,
    get_anomaly_scores_tcrad, evaluate_anomaly_detection, evaluate_sklearn_baseline
)


def save_results(results, filename):
    """Save results to CSV."""
    path = os.path.join(RESULTS_DIR, filename)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved to {path}")


def run_main_comparison():
    """Run main comparison between TCR-AD and baselines."""
    print("\n" + "="*60)
    print("MAIN COMPARISON: TCR-AD vs Baselines")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    all_results = []
    
    for seed in RANDOM_SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        
        # Load data
        X, y = load_sgcc_data(sample_ratio=SAMPLE_RATIO, random_state=seed)
        train_loader, val_loader, test_loader, splits = create_data_loaders(
            X, y, random_state=seed
        )
        X_train, y_train, X_val, y_val, X_test, y_test = splits
        
        # Flatten for sklearn models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # ===== TCR-AD =====
        print("\nTraining TCR-AD...")
        model = TCRAD().to(device)
        t0 = time.time()
        model, _ = train_tcrad(model, train_loader, val_loader, device)
        train_time = time.time() - t0
        scores, labels = get_anomaly_scores_tcrad(model, test_loader, device)
        results = evaluate_anomaly_detection(scores, labels)
        results['model'] = 'TCR-AD'
        results['seed'] = seed
        results['train_time'] = round(train_time, 2)
        n_params = sum(p.numel() for p in model.parameters())
        results['params'] = n_params
        all_results.append(results)
        print(f"TCR-AD: AUC={results['auc_roc']:.4f}, F1={results['best_f1']:.4f}")
        
        # ===== OCSVM =====
        print("\nTraining OCSVM...")
        t0 = time.time()
        # Subsample for OCSVM (sklearn SVM doesn't scale well)
        X_train_normal = X_train_flat[y_train == 0]
        if X_train_normal.shape[0] > 5000:
            rng = np.random.RandomState(seed)
            sub_idx = rng.choice(X_train_normal.shape[0], 5000, replace=False)
            X_train_normal_sub = X_train_normal[sub_idx]
        else:
            X_train_normal_sub = X_train_normal
        ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        ocsvm.fit(X_train_normal_sub)  # Only normal data (subsampled)
        train_time = time.time() - t0
        scores = -ocsvm.decision_function(X_test_flat)
        results = evaluate_anomaly_detection(scores, y_test)
        results['model'] = 'OCSVM'
        results['seed'] = seed
        results['train_time'] = round(train_time, 2)
        results['params'] = 0
        all_results.append(results)
        print(f"OCSVM: AUC={results['auc_roc']:.4f}, F1={results['best_f1']:.4f}")
        
        # ===== IForest =====
        print("\nTraining IForest...")
        t0 = time.time()
        iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=seed)
        iforest.fit(X_train_flat)
        train_time = time.time() - t0
        scores = -iforest.score_samples(X_test_flat)
        results = evaluate_anomaly_detection(scores, y_test)
        results['model'] = 'IForest'
        results['seed'] = seed
        results['train_time'] = round(train_time, 2)
        results['params'] = 0
        all_results.append(results)
        print(f"IForest: AUC={results['auc_roc']:.4f}, F1={results['best_f1']:.4f}")
        
        # ===== Autoencoder =====
        print("\nTraining AE...")
        ae = Autoencoder(seq_len=SUB_SEQ_LEN).to(device)
        t0 = time.time()
        ae = train_ae(ae, train_loader, device)
        train_time = time.time() - t0
        ae.eval()
        scores = []
        with torch.no_grad():
            for batch in test_loader:
                x, _ = batch
                x = x.to(device)
                recon = ae(x)
                recon_err = torch.nn.MSELoss(reduction='none')(recon, x.squeeze(-1)).mean(dim=-1)
                scores.append(recon_err.cpu().numpy())
        scores = np.concatenate(scores)
        results = evaluate_anomaly_detection(scores, y_test)
        results['model'] = 'AE'
        results['seed'] = seed
        results['train_time'] = round(train_time, 2)
        results['params'] = sum(p.numel() for p in ae.parameters())
        all_results.append(results)
        print(f"AE: AUC={results['auc_roc']:.4f}, F1={results['best_f1']:.4f}")
        
        # ===== VAE =====
        print("\nTraining VAE...")
        vae = VAE(seq_len=SUB_SEQ_LEN).to(device)
        t0 = time.time()
        vae = train_vae(vae, train_loader, device)
        train_time = time.time() - t0
        vae.eval()
        scores = []
        with torch.no_grad():
            for batch in test_loader:
                x, _ = batch
                x = x.to(device)
                score = vae.get_anomaly_score(x)
                scores.append(score.cpu().numpy())
        scores = np.concatenate(scores)
        results = evaluate_anomaly_detection(scores, y_test)
        results['model'] = 'VAE'
        results['seed'] = seed
        results['train_time'] = round(train_time, 2)
        results['params'] = sum(p.numel() for p in vae.parameters())
        all_results.append(results)
        print(f"VAE: AUC={results['auc_roc']:.4f}, F1={results['best_f1']:.4f}")
        
        # ===== DAGMM =====
        print("\nTraining DAGMM...")
        dagmm = DAGMM(seq_len=SUB_SEQ_LEN).to(device)
        t0 = time.time()
        dagmm = train_dagmm(dagmm, train_loader, device)
        train_time = time.time() - t0
        dagmm.eval()
        scores = []
        with torch.no_grad():
            for batch in test_loader:
                x, _ = batch
                x = x.to(device)
                score = dagmm.get_anomaly_score(x)
                scores.append(score.cpu().numpy())
        scores = np.concatenate(scores)
        results = evaluate_anomaly_detection(scores, y_test)
        results['model'] = 'DAGMM'
        results['seed'] = seed
        results['train_time'] = round(train_time, 2)
        results['params'] = sum(p.numel() for p in dagmm.parameters())
        all_results.append(results)
        print(f"DAGMM: AUC={results['auc_roc']:.4f}, F1={results['best_f1']:.4f}")
    
    # Save all results
    save_results(all_results, 'main_comparison.csv')
    
    # Compute mean/std across seeds
    models = list(set(r['model'] for r in all_results))
    summary = []
    for m in models:
        m_results = [r for r in all_results if r['model'] == m]
        summary.append({
            'model': m,
            'auc_roc_mean': np.mean([r['auc_roc'] for r in m_results]),
            'auc_roc_std': np.std([r['auc_roc'] for r in m_results]),
            'f1_mean': np.mean([r['best_f1'] for r in m_results]),
            'f1_std': np.std([r['best_f1'] for r in m_results]),
            'precision_mean': np.mean([r['best_precision'] for r in m_results]),
            'recall_mean': np.mean([r['best_recall'] for r in m_results]),
            'train_time_mean': np.mean([r['train_time'] for r in m_results]),
            'params_mean': np.mean([r['params'] for r in m_results])
        })
    save_results(summary, 'main_comparison_summary.csv')
    
    print("\n" + "="*60)
    print("MAIN COMPARISON SUMMARY")
    print("="*60)
    for s in summary:
        print(f"{s['model']:10s}: AUC={s['auc_roc_mean']:.4f}±{s['auc_roc_std']:.4f}, F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}")
    
    return all_results, summary


def run_ablation():
    """Run ablation study on TCR-AD components."""
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(RANDOM_SEEDS[0])
    
    X, y = load_sgcc_data(sample_ratio=SAMPLE_RATIO)
    train_loader, val_loader, test_loader, splits = create_data_loaders(X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    
    # Ablation variants
    variants = {
        'Full TCR-AD': {'use_time': True, 'use_freq': True},
        'w/o Time Encoder': {'use_time': False, 'use_freq': True},
        'w/o Freq Encoder': {'use_time': True, 'use_freq': False},
        'w/o Contrastive (Recon only)': {'use_time': True, 'use_freq': True},  # no contrastive
    }
    
    results = []
    import config as cfg_module
    import models as models_module
    for name, cfg in variants.items():
        print(f"\n--- {name} ---", flush=True)
        # Save original values
        orig_cw = cfg_module.CONTRASTIVE_WEIGHT
        orig_rw = cfg_module.RECON_WEIGHT
        
        model = TCRAD().to(device)
        model.use_time_encoder = cfg['use_time']
        model.use_freq_encoder = cfg['use_freq']
        
        # For recon-only variant, set contrastive weight to 0
        if 'Contrastive' in name:
            cfg_module.CONTRASTIVE_WEIGHT = 0.0
            cfg_module.RECON_WEIGHT = 1.0
            models_module.CONTRASTIVE_WEIGHT = 0.0
            models_module.RECON_WEIGHT = 1.0
        
        model, _ = train_tcrad(model, train_loader, val_loader, device, n_epochs=N_EPOCHS)
        scores, labels = get_anomaly_scores_tcrad(model, test_loader, device)
        eval_results = evaluate_anomaly_detection(scores, labels)
        eval_results['variant'] = name
        results.append(eval_results)
        print(f"{name}: AUC={eval_results['auc_roc']:.4f}, F1={eval_results['best_f1']:.4f}", flush=True)
        
        # Restore config
        cfg_module.CONTRASTIVE_WEIGHT = orig_cw
        cfg_module.RECON_WEIGHT = orig_rw
        models_module.CONTRASTIVE_WEIGHT = orig_cw
        models_module.RECON_WEIGHT = orig_rw
    
    save_results(results, 'ablation_results.csv')
    return results


def run_sensitivity():
    """Run parameter sensitivity analysis (reduced for efficiency)."""
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(RANDOM_SEEDS[0])
    
    X, y = load_sgcc_data(sample_ratio=SAMPLE_RATIO)
    train_loader, val_loader, test_loader, splits = create_data_loaders(X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    
    results = []
    
    import config as cfg_module
    import models as models_module
    
    # Save original values
    orig_embed = EMBED_DIM
    orig_cw = CONTRASTIVE_WEIGHT
    orig_rw = RECON_WEIGHT
    
    # 1. Embedding dimension sensitivity
    print("\n--- Embedding Dimension Sensitivity ---", flush=True)
    for dim in [32, 64, 128, 256]:
        print(f"  Embedding Dim = {dim}", flush=True)
        # Override in both config and models module (models.py uses its own copy via 'from config import *')
        cfg_module.EMBED_DIM = dim
        models_module.EMBED_DIM = dim
        model = models_module.TCRAD().to(device)
        model, _ = train_tcrad(model, train_loader, val_loader, device, n_epochs=N_EPOCHS)
        scores, labels = get_anomaly_scores_tcrad(model, test_loader, device)
        eval_results = evaluate_anomaly_detection(scores, labels)
        results.append({
            'param': 'embedding_dim',
            'value': dim,
            'auc_roc': float(eval_results['auc_roc']),
            'f1': float(eval_results['best_f1'])
        })
        print(f"    AUC={eval_results['auc_roc']:.4f}", flush=True)
    
    # 2. Contrastive weight sensitivity
    print("\n--- Contrastive Weight Sensitivity ---", flush=True)
    # Reset EMBED_DIM first
    cfg_module.EMBED_DIM = orig_embed
    models_module.EMBED_DIM = orig_embed
    for cw in [0.0, 0.25, 0.75, 1.0]:
        print(f"  Contrastive Weight = {cw}", flush=True)
        cfg_module.CONTRASTIVE_WEIGHT = cw
        cfg_module.RECON_WEIGHT = 1.0 - cw
        models_module.CONTRASTIVE_WEIGHT = cw
        models_module.RECON_WEIGHT = 1.0 - cw
        model = models_module.TCRAD().to(device)
        model, _ = train_tcrad(model, train_loader, val_loader, device, n_epochs=N_EPOCHS)
        scores, labels = get_anomaly_scores_tcrad(model, test_loader, device)
        eval_results = evaluate_anomaly_detection(scores, labels)
        results.append({
            'param': 'contrastive_weight',
            'value': cw,
            'auc_roc': float(eval_results['auc_roc']),
            'f1': float(eval_results['best_f1'])
        })
        print(f"    AUC={eval_results['auc_roc']:.4f}", flush=True)
    
    # Reset config to original values
    cfg_module.EMBED_DIM = orig_embed
    cfg_module.CONTRASTIVE_WEIGHT = orig_cw
    cfg_module.RECON_WEIGHT = orig_rw
    models_module.EMBED_DIM = orig_embed
    models_module.CONTRASTIVE_WEIGHT = orig_cw
    models_module.RECON_WEIGHT = orig_rw
    
    save_results(results, 'sensitivity_all.csv')
    
    # Also save as JSON
    sens_json_path = os.path.join(RESULTS_DIR, 'sensitivity_results.json')
    with open(sens_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON to {sens_json_path}")
    
    return results


def run_complexity_analysis():
    """Compute model complexity."""
    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS")
    print("="*60)
    
    results = []
    
    models_list = {
        'TCR-AD': TCRAD(),
        'AE': Autoencoder(seq_len=SUB_SEQ_LEN),
        'VAE': VAE(seq_len=SUB_SEQ_LEN),
        'DAGMM': DAGMM(seq_len=SUB_SEQ_LEN),
    }
    
    x = torch.randn(1, SUB_SEQ_LEN, 1)
    
    for name, model in models_list.items():
        n_params = sum(p.numel() for p in model.parameters())
        
        # Measure inference time
        model.eval()
        times = []
        with torch.no_grad():
            for _ in range(100):
                t0 = time.time()
                _ = model(x)
                t1 = time.time()
                times.append((t1 - t0) * 1000)  # ms
        
        avg_time = np.mean(times[10:])  # Skip first 10 warmup
        results.append({
            'model': name,
            'params': n_params,
            'inference_time_ms': round(avg_time, 4)
        })
        print(f"{name:10s}: {n_params:,} params, {avg_time:.4f} ms")
    
    save_results(results, 'complexity_analysis.csv')
    return results


def run_statistical_tests(main_results=None):
    """Run statistical significance tests between TCR-AD and baselines.
    Reuses results from main comparison if available to avoid re-training."""
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    from scipy import stats as scipy_stats
    
    if main_results is not None:
        # Reuse results from main comparison
        all_results = []
        for r in main_results:
            entry = {}
            if r['model'] in ['TCR-AD', 'OCSVM', 'IForest', 'AE', 'VAE', 'DAGMM']:
                entry[r['model']] = r['auc_roc']
            if entry:
                all_results.append(entry)
        
        # Group by seed
        seed_results = {}
        for r in main_results:
            seed = r['seed']
            if seed not in seed_results:
                seed_results[seed] = {}
            seed_results[seed][r['model']] = r['auc_roc']
        
        all_results = list(seed_results.values())
        print(f"Reusing results from main comparison ({len(all_results)} seeds)")
    else:
        # Fall back to re-training (original behavior)
        all_results = []
        for seed in RANDOM_SEEDS:
            set_seed(seed)
            X, y = load_sgcc_data(sample_ratio=SAMPLE_RATIO, random_state=seed)
            train_loader, val_loader, test_loader, splits = create_data_loaders(X, y, random_state=seed)
            X_train, y_train, X_val, y_val, X_test, y_test = splits
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # TCR-AD
            model = TCRAD().to(device)
            model, _ = train_tcrad(model, train_loader, val_loader, device, n_epochs=N_EPOCHS)
            scores_tcrad, _ = get_anomaly_scores_tcrad(model, test_loader, device)
            auc_tcrad = evaluate_anomaly_detection(scores_tcrad, y_test)['auc_roc']
            all_results.append({'TCR-AD': auc_tcrad})
            
            # Baselines
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            
            # OCSVM
            X_train_normal = X_train_flat[y_train == 0]
            if X_train_normal.shape[0] > 5000:
                rng = np.random.RandomState(seed)
                sub_idx = rng.choice(X_train_normal.shape[0], 5000, replace=False)
                X_train_normal = X_train_normal[sub_idx]
            ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
            ocsvm.fit(X_train_normal)
            scores = -ocsvm.decision_function(X_test_flat)
            all_results[-1]['OCSVM'] = evaluate_anomaly_detection(scores, y_test)['auc_roc']
            
            # IForest
            iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=seed)
            iforest.fit(X_train_flat)
            scores = -iforest.score_samples(X_test_flat)
            all_results[-1]['IForest'] = evaluate_anomaly_detection(scores, y_test)['auc_roc']
    
    # Compute paired t-tests against all baselines
    baselines = ['OCSVM', 'IForest', 'AE', 'VAE', 'DAGMM']
    stat_results = []
    for bl in baselines:
        tcrad_aucs = [r['TCR-AD'] for r in all_results if 'TCR-AD' in r and bl in r]
        bl_aucs = [r[bl] for r in all_results if 'TCR-AD' in r and bl in r]
        if len(tcrad_aucs) < 2:
            continue
        t_stat, p_val = scipy_stats.ttest_rel(tcrad_aucs, bl_aucs)
        diff_arr = np.array(tcrad_aucs) - np.array(bl_aucs)
        d = (np.mean(tcrad_aucs) - np.mean(bl_aucs)) / (np.std(diff_arr) + 1e-10)
        # 95% CI
        mean_diff = float(np.mean(diff_arr))
        se_diff = float(np.std(diff_arr, ddof=1) / np.sqrt(len(diff_arr))) if len(diff_arr) > 1 else 0.0
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        stat_results.append({
            'baseline': bl,
            'tcrad_auc_mean': float(np.mean(tcrad_aucs)),
            'baseline_auc_mean': float(np.mean(bl_aucs)),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(abs(d)),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'n_seeds': len(tcrad_aucs)
        })
        print(f"TCR-AD vs {bl}: t={t_stat:.4f}, p={p_val:.4f}, d={abs(d):.4f}")
    
    save_results(stat_results, 'statistical_tests.csv')
    
    # Also save as JSON
    stat_json_path = os.path.join(RESULTS_DIR, 'statistical_tests.json')
    with open(stat_json_path, 'w') as f:
        json.dump(stat_results, f, indent=2)
    print(f"Saved JSON to {stat_json_path}")
    
    return stat_results


if __name__ == '__main__':
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Phase 1: Main comparison
    all_results, summary = run_main_comparison()
    
    # Save main comparison results as JSON
    main_json_path = os.path.join(RESULTS_DIR, 'main_comparison_results.json')
    with open(main_json_path, 'w') as f:
        json.dump({'per_seed': all_results, 'summary': summary}, f, indent=2, default=str)
    print(f"\nSaved main comparison JSON to {main_json_path}")
    
    # Phase 2: Ablation
    ablation_results = run_ablation()
    
    # Save ablation results as JSON
    abl_json_path = os.path.join(RESULTS_DIR, 'ablation_results.json')
    with open(abl_json_path, 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    print(f"Saved ablation JSON to {abl_json_path}")
    
    # Phase 3: Sensitivity (reduced)
    sensitivity_results = run_sensitivity()
    
    # Phase 4: Complexity
    complexity_results = run_complexity_analysis()
    
    # Save complexity results as JSON
    comp_json_path = os.path.join(RESULTS_DIR, 'complexity_analysis.json')
    with open(comp_json_path, 'w') as f:
        json.dump(complexity_results, f, indent=2, default=str)
    print(f"Saved complexity JSON to {comp_json_path}")
    
    # Phase 5: Statistical tests (reuse main comparison results)
    stat_results = run_statistical_tests(main_results=all_results)
    
    # Generate combined summary for paper reference
    combined_summary = {
        'direction': '44_Energy_Anomaly',
        'task': 'anomaly_detection',
        'dataset': 'SGCC Electricity Theft Detection',
        'metric': 'AUC-ROC',
        'seeds': RANDOM_SEEDS,
        'main_comparison': summary,
        'ablation': ablation_results,
        'sensitivity': sensitivity_results,
        'complexity': complexity_results,
        'statistical_tests': stat_results,
    }
    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(combined_summary, f, indent=2, default=str)
    print(f"\nSaved combined summary to {summary_path}", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("ALL EXPERIMENTS COMPLETED", flush=True)
    print("="*60, flush=True)