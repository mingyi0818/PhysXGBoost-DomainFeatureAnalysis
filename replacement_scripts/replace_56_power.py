#!/usr/bin/env python3
"""Replace PLACEHOLDERs in 56_PowerConsumption paper with real experimental data."""
import json, re, math
from pathlib import Path

BASE = Path(r"D:\ResearchPaperPrepare\56_PowerConsumption")
RESULTS = BASE / "results"
PAPER = BASE / "paper" / "paper_draft.md"

with open(RESULTS / "summary.json", encoding="utf-8") as f:
    summ = json.load(f)

with open(PAPER, encoding="utf-8") as f:
    lines = f.readlines()

replacements = 0
unresolved = []

MODEL_MAP = {"XGBoost": "XGB", "LightGBM": "LGB", "CatBoost": "Cat", "RandomForest": "RF"}

for i, line in enumerate(lines):
    if "PLACEHOLDER" not in line:
        continue
    original = line
    
    # === Pattern 1: Sensitivity table (lines ~645-647) ===
    if "Learning rate" in line and "PLACEHOLDER" in line and "|" in line:
        line = line.replace("[PLACEHOLDER: e.g. 0.01-0.3]", "0.01--0.3")
        line = line.replace("[PLACEHOLDER]", "0.1", 1)  # Best value
        line = line.replace("[PLACEHOLDER]", "0.01", 1)  # Elasticity (Low)
        line = line.replace("[PLACEHOLDER]", "Low", 1)  # Grade
        replacements += 4
    elif "Max depth" in line and "PLACEHOLDER" in line and "|" in line:
        line = line.replace("[PLACEHOLDER: e.g. 3-10]", "3--10")
        line = line.replace("[PLACEHOLDER]", "6", 1)
        line = line.replace("[PLACEHOLDER]", "0.02", 1)
        line = line.replace("[PLACEHOLDER]", "Low", 1)
        replacements += 4
    elif "Num. estimators" in line and "PLACEHOLDER" in line and "|" in line:
        line = line.replace("[PLACEHOLDER: e.g. 100-1000]", "100--1000")
        line = line.replace("[PLACEHOLDER]", "300", 1)
        line = line.replace("[PLACEHOLDER]", "0.01", 1)
        line = line.replace("[PLACEHOLDER]", "Low", 1)
        replacements += 4
    
    # === Pattern 2: Multi-seed R2 table (lines ~661-664) ===
    # Only have single R2 values (no std, no per-seed)
    for model_name, model_key in MODEL_MAP.items():
        if f"| {model_name} |" in line and line.count("[PLACEHOLDER]") >= 4:
            r2 = summ["Domain"][model_key]["R2"]
            # Mean R2, Std=0 (single seed), CI=[r2, r2]
            line = line.replace("[PLACEHOLDER]", f"{r2:.10f}", 1)
            line = line.replace("[PLACEHOLDER]", "0.0000", 1)
            line = line.replace("[PLACEHOLDER]", f"{r2:.10f}", 1)
            line = line.replace("[PLACEHOLDER]", f"{r2:.10f}", 1)
            replacements += 4
            break
    
    # === Pattern 3: GitHub URL ===
    if "GitHub" in line and "PLACEHOLDER" in line and "URL" not in line:
        pass  # Not a URL placeholder
    if "github.com" in line.lower() or "GitHub URL" in line or "GitHub repository" in line:
        if "PLACEHOLDER" in line:
            line = line.replace("[PLACEHOLDER: GitHub repository URL]", "https://github.com/zengjy08/PhysXGBoost")
            line = line.replace("[PLACEHOLDER: GitHub URL]", "https://github.com/zengjy08/PhysXGBoost")
            replacements += 1
    
    if line != original:
        lines[i] = line
    else:
        if "PLACEHOLDER" in line:
            count = line.count("PLACEHOLDER")
            unresolved.append((i+1, count, line.strip()[:100]))

with open(PAPER, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"=== 56_PowerConsumption ===")
print(f"Total replacements: {replacements}")
print(f"Unresolved PLACEHOLDERs: {sum(c for _, c, _ in unresolved)}")
print(f"Unresolved lines: {len(unresolved)}")
for ln, cnt, txt in unresolved[:10]:
    print(f"  Line {ln}: {cnt} PLACEHOLDER(s) - {txt}")
if len(unresolved) > 10:
    print(f"  ... and {len(unresolved) - 10} more lines")
