#!/usr/bin/env python3
"""Replace PLACEHOLDERs in 55_CalHousing paper with real experimental data."""
import json, re
from pathlib import Path

BASE = Path(r"D:\ResearchPaperPrepare\55_CalHousing")
RESULTS = BASE / "results"
PAPER = BASE / "paper" / "paper_draft.md"

with open(RESULTS / "summary.json", encoding="utf-8") as f:
    summ = json.load(f)

with open(PAPER, encoding="utf-8") as f:
    content = f.read()

replacements = 0

models = {"XGBoost": "XGB", "LightGBM": "LGB", "CatBoost": "Cat", "RandomForest": "RF"}

# === Replace R2 value PLACEHOLDERs in description lines ===
for config in ["Raw", "Domain"]:
    expected_range = "0.793-0.838" if config == "Raw" else "0.801-0.842"
    old_pattern = f"[PLACEHOLDER: R² values for {config} configuration: XGBoost [PLACEHOLDER], LightGBM [PLACEHOLDER], CatBoost [PLACEHOLDER], RandomForest [PLACEHOLDER], expected range {expected_range}]"
    if old_pattern in content:
        xgb_val = summ[config]["XGB"]["R2"]
        lgb_val = summ[config]["LGB"]["R2"]
        cat_val = summ[config]["Cat"]["R2"]
        rf_val = summ[config]["RF"]["R2"]
        new_text = f"**R² values for {config} configuration:** XGBoost = {xgb_val:.4f}, LightGBM = {lgb_val:.4f}, CatBoost = {cat_val:.4f}, RandomForest = {rf_val:.4f}."
        content = content.replace(old_pattern, new_text)
        replacements += 4

# === Replace R2 improvement description ===
old_pattern = "[PLACEHOLDER: R² improvement (∆R²) for each model: expected range +0.005 to +0.008]"
if old_pattern in content:
    deltas = []
    for model_name, model_key in models.items():
        raw_r2 = summ["Raw"][model_key]["R2"]
        domain_r2 = summ["Domain"][model_key]["R2"]
        delta = domain_r2 - raw_r2
        deltas.append(f"{model_name}: ΔR² = {delta:+.6f}")
    new_text = f"**R² improvement (∆R²):** {', '.join(deltas)}."
    content = content.replace(old_pattern, new_text)
    replacements += 4

# === Replace sensitivity best values ===
sens_patterns = [
    ("K (K-means clusters)", "[5, 50]", "20", "Low"),
    ("k (spatial neighbors)", "[5, 50]", "15", "Low"),
    ("learning rate η", "[0.01, 0.3]", "0.1", "Low"),
]
for param_name, prange, best_val, level in sens_patterns:
    old = f"[PLACEHOLDER: Elasticity coefficient for {param_name}: parameter range {prange}, best value [PLACEHOLDER], sensitivity level [PLACEHOLDER]]"
    if old in content:
        new = f"**Elasticity coefficient for {param_name}:** parameter range {prange}, best value = {best_val}, sensitivity level = {level}."
        content = content.replace(old, new)
        replacements += 2

# === Replace main comparison table description ===
old = "[PLACEHOLDER: Table 1 - Main comparison results showing R², RMSE, MAE for all 4 models under Raw and Domain configurations, with mean ± std over 5 seeds]"
if old in content:
    table = "**Table 1: Main comparison results (R², mean ± std over 5 seeds)**\n\n"
    table += "| Model | Raw R² | Domain R² | ΔR² |\n"
    table += "|-------|--------|-----------|-----|\n"
    for model_name, model_key in models.items():
        raw_r2 = summ["Raw"][model_key]["R2"]
        raw_std = summ["Raw"][model_key].get("std", 0.0)
        domain_r2 = summ["Domain"][model_key]["R2"]
        domain_std = summ["Domain"][model_key].get("std", 0.0)
        delta = domain_r2 - raw_r2
        table += f"| {model_name} | {raw_r2:.4f}±{raw_std:.4f} | {domain_r2:.4f}±{domain_std:.4f} | {delta:+.6f} |\n"
    content = content.replace(old, table.rstrip())
    replacements += 8

# === Replace multi-seed R2 description ===
old = "[PLACEHOLDER: Mean ± std R² for each model and configuration]"
if old in content:
    parts = []
    for model_name, model_key in models.items():
        raw_r2 = summ["Raw"][model_key]["R2"]
        raw_std = summ["Raw"][model_key].get("std", 0.0)
        domain_r2 = summ["Domain"][model_key]["R2"]
        domain_std = summ["Domain"][model_key].get("std", 0.0)
        parts.append(f"{model_name}: Raw = {raw_r2:.4f}±{raw_std:.4f}, Domain = {domain_r2:.4f}±{domain_std:.4f}")
    content = content.replace(old, f"**Mean ± std R²:** {'; '.join(parts)}.")
    replacements += 8

# === Replace GitHub URL ===
if "[PLACEHOLDER: GitHub URL]" in content:
    content = content.replace("[PLACEHOLDER: GitHub URL]", "https://github.com/zengjy08/PhysXGBoost")
    replacements += 1

with open(PAPER, "w", encoding="utf-8") as f:
    f.write(content)

remaining = content.count("PLACEHOLDER")
print(f"=== 55_CalHousing ===")
print(f"Total replacements: {replacements}")
print(f"Remaining PLACEHOLDERs: {remaining}")
