#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix generate_individual_reports.py path issues"""
from pathlib import Path

path = Path('F:/Project/4.1.9.final/generate_individual_reports.py')
content = path.read_text(encoding='utf-8')

# Fix line 457: str(Path(...)) -> Path(...)
old1 = "df = pd.read_excel(str(Path(__file__).parent / 'data/processed/real_data_103.xlsx')"
new1 = "df = pd.read_excel(Path(__file__).parent / 'data/processed/real_data_103.xlsx'"
content = content.replace(old1, new1)

# Fix line 497: Path(str(Path(...))) -> Path(...)
old2 = "out_dir = Path(str(Path(__file__).parent / 'results/individual_reports')"
new2 = "out_dir = Path(__file__).parent / 'results/individual_reports'"
content = content.replace(old2, new2)

path.write_text(content, encoding='utf-8')

# Verify
lines = content.split('\n')
for i, l in enumerate(lines):
    if 'apex' in l or 'read_excel' in l or 'out_dir = ' in l:
        print(f'Line {i+1}: {repr(l)}')
print('Done.')
