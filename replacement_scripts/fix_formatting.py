#!/usr/bin/env python3
"""Fix LaTeX formatting in 65_HR paper."""
import re
from pathlib import Path

PAPER = Path(r"D:\ResearchPaperPrepare\65_HR\paper\paper_draft.md")
with open(PAPER, encoding="utf-8") as f:
    content = f.read()

# Fix pattern: $0.0084$\pm$0.0101$ -> $0.0084 \pm 0.0101$
# General pattern: $number$\pm$number$
pattern = r'\$(-?\d+\.\d+)\$\\pm\$(-?\d+\.\d+)\$'
matches = re.findall(pattern, content)
fixes = 0
for mean_val, std_val in matches:
    old_str = f"${mean_val}$\\pm${std_val}$"
    new_str = f"${mean_val} \\pm {std_val}$"
    content = content.replace(old_str, new_str)
    fixes += 1

with open(PAPER, "w", encoding="utf-8") as f:
    f.write(content)

print(f"=== 65_HR formatting fix ===")
print(f"Fixes applied: {fixes}")
