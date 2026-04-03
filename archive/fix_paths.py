#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复所有脚本中的硬编码路径"""
from pathlib import Path

scripts = [
    'generate_individual_reports.py',
    'train_v7_ultimate.py',
    'check_v7_overfitting.py',
    'train_optimized_v3.py',
    'train_final_robust.py',
    'validate_overfitting.py',
    'generate_interaction_preserved_dataset.py',
]

base = Path(__file__).parent

for script in scripts:
    path = base / script
    if not path.exists():
        print(f'NOT FOUND: {script}')
        continue
    content = path.read_text(encoding='utf-8')
    original = content

    # 替换所有 4.1.9.apex 路径
    content = content.replace("'4.1.9.apex/", "str(Path(__file__).parent / '")
    content = content.replace('"4.1.9.apex/', 'str(Path(__file__).parent / "')

    # 更精确的替换
    content = content.replace(
        "pd.read_excel('4.1.9.apex/data/processed/real_data_103.xlsx')",
        "pd.read_excel(Path(__file__).parent / 'data/processed/real_data_103.xlsx')"
    )
    content = content.replace(
        'pd.read_excel("4.1.9.apex/data/processed/real_data_103.xlsx")',
        "pd.read_excel(Path(__file__).parent / 'data/processed/real_data_103.xlsx')"
    )
    content = content.replace(
        "out_dir = Path('4.1.9.apex/results/individual_reports')",
        "out_dir = Path(__file__).parent / 'results/individual_reports'"
    )
    content = content.replace(
        'out_dir = Path("4.1.9.apex/results/individual_reports")',
        "out_dir = Path(__file__).parent / 'results/individual_reports'"
    )

    if content != original:
        path.write_text(content, encoding='utf-8')
        print(f'Fixed: {script}')
    else:
        print(f'No changes: {script}')

print('Done.')
