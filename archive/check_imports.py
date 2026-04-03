#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查所有脚本的导入和未解析引用"""
import subprocess
import sys
import os
from pathlib import Path

base = Path(__file__).parent
os.chdir(base)

scripts = [
    'train_final_robust.py',
    'interpretability_analysis.py',
    'generate_individual_reports.py',
    'validate_overfitting.py',
    'run_all.py',
    'utils/model_utils.py',
    'utils/data_utils.py',
    'utils/logging_utils.py',
]

print('=== 导入检查（py_compile）===')
for s in scripts:
    p = base / s
    if not p.exists():
        print(f'  MISS: {s}')
        continue
    result = subprocess.run(
        [sys.executable, '-m', 'py_compile', str(p)],
        capture_output=True, text=True, timeout=30,
        encoding='utf-8', cwd=str(base)
    )
    if result.returncode == 0:
        print(f'  OK: {s}')
    else:
        print(f'  ERROR: {s}')
        print(f'    {result.stderr[:300]}')

print()
print('=== 检查 generate_individual_reports.py 中的 Path 导入 ===')
content = (base / 'generate_individual_reports.py').read_text(encoding='utf-8')
lines = content.split('\n')
for i, l in enumerate(lines[:30]):
    if 'import' in l or 'from' in l:
        print(f'  Line {i+1}: {l}')

print()
print('=== 检查 interpretability_analysis.py 中的路径引用 ===')
content2 = (base / 'interpretability_analysis.py').read_text(encoding='utf-8')
for i, l in enumerate(content2.split('\n')):
    if 'apex' in l or '4.1.9' in l:
        print(f'  Line {i+1}: {l}')

print()
print('=== 检查 run_all.py 中的路径引用 ===')
content3 = (base / 'run_all.py').read_text(encoding='utf-8')
for i, l in enumerate(content3.split('\n')):
    if 'apex' in l or '4.1.9' in l:
        print(f'  Line {i+1}: {l}')
