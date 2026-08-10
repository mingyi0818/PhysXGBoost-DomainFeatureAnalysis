#!/usr/bin/env python3
"""Supplementary replacement script for 54_NewsPopularity and 60_StudentPerf."""
import json, re, math
import numpy as np
from pathlib import Path

# === 54_NewsPopularity: Fill ablation table baseline values ===
BASE_54 = Path(r"D:\ResearchPaperPrepare\54_NewsPopularity")
with open(BASE_54 / "results" / "comprehensive_results.json", encoding="utf-8") as f:
    comp54 = json.load(f)
with open(BASE_54 / "results" / "summary.json", encoding="utf-8") as f:
    summ54 = json.load(f)

PAPER_54 = BASE_54 / "paper" / "paper_draft.md"
with open(PAPER_54, encoding="utf-8") as f:
    content54 = f.read()

replacements_54 = 0

# Fill ablation table "Raw (baseline)" and "Full Domain" rows
# CatBoost is the best model, use its R2 values
cat_raw_r2 = summ54["Raw"]["Cat"]["R2"]
cat_domain_r2 = summ54["Domain"]["Cat"]["R2"]

# Pattern: [PLACEHOLDER: 0.0241] in Raw baseline row
old = "[PLACEHOLDER: 0.0241]"
if old in content54:
    content54 = content54.replace(old, f"${cat_raw_r2:.4f}$")
    replacements_54 += 1

# Pattern: [PLACEHOLDER: 0.0283] in Full Domain row
old = "[PLACEHOLDER: 0.0283]"
if old in content54:
    content54 = content54.replace(old, f"${cat_domain_r2:.4f}$")
    replacements_54 += 1

# Fill n_samples references if any PLACEHOLDER for exact count
n_samples = comp54["n_samples"]
n_raw_features = comp54["n_raw_features"]

# Fix: Replace [PLACEHOLDER: exact count] if exists
if "[PLACEHOLDER: exact count]" in content54:
    content54 = content54.replace("[PLACEHOLDER: exact count]", str(n_samples))
    replacements_54 += 1

# Fill edge deployment model sizes (leave as is - no data)

# Fix log-transform table improvement column
# The table has "1.0000" in Log-transformed column which is Wilcoxon p-value
# Leave as is since no log-transform data

# Fill summary table entries
# Line 848: | Ablation | [PLACEHOLDER] | [PLACEHOLDER] | - no ablation data, leave as is

# Replace remaining GitHub URL if any
if "[PLACEHOLDER: GitHub URL]" in content54:
    content54 = content54.replace("[PLACEHOLDER: GitHub URL]", "https://github.com/zengjy08/PhysXGBoost")
    replacements_54 += 1

with open(PAPER_54, "w", encoding="utf-8") as f:
    f.write(content54)

print(f"=== 54_NewsPopularity (supplementary) ===")
print(f"Additional replacements: {replacements_54}")
remaining_54 = content54.count("PLACEHOLDER")
print(f"Remaining PLACEHOLDERs: {remaining_54}")

# === 60_StudentPerf: Fix formatting in ΔAUC column ===
BASE_60 = Path(r"D:\ResearchPaperPrepare\60_StudentPerf")
PAPER_60 = BASE_60 / "paper" / "paper_draft.md"
with open(PAPER_60, encoding="utf-8") as f:
    content60 = f.read()

replacements_60 = 0

# Fix formatting: "$-0.0110\pm$0.0707$" -> "$-0.0110 \pm 0.0707$"
# Pattern: number\pm$number$ -> number \pm number
pattern = r'\$(-?\d+\.\d+)\\pm\$(-?\d+\.\d+)\$'
matches = re.findall(pattern, content60)
for mean_val, std_val in matches:
    old_str = f"${mean_val}\\pm${std_val}$"
    new_str = f"${mean_val} \\pm {std_val}$"
    content60 = content60.replace(old_str, new_str)
    replacements_60 += 1

# Also fix the ΔAUC column format in 65_HR
BASE_65 = Path(r"D:\ResearchPaperPrepare\65_HR")
PAPER_65 = BASE_65 / "paper" / "paper_draft.md"
with open(PAPER_65, encoding="utf-8") as f:
    content65 = f.read()

replacements_65 = 0
matches65 = re.findall(pattern, content65)
for mean_val, std_val in matches65:
    old_str = f"${mean_val}\\pm${std_val}$"
    new_str = f"${mean_val} \\pm {std_val}$"
    content65 = content65.replace(old_str, new_str)
    replacements_65 += 1

with open(PAPER_60, "w", encoding="utf-8") as f:
    f.write(content60)
with open(PAPER_65, "w", encoding="utf-8") as f:
    f.write(content65)

print(f"\n=== 60_StudentPerf (formatting fix) ===")
print(f"Formatting fixes: {replacements_60}")
remaining_60 = content60.count("PLACEHOLDER")
print(f"Remaining PLACEHOLDERs: {remaining_60}")

print(f"\n=== 65_HR (formatting fix) ===")
print(f"Formatting fixes: {replacements_65}")
remaining_65 = content65.count("PLACEHOLDER")
print(f"Remaining PLACEHOLDERs: {remaining_65}")
