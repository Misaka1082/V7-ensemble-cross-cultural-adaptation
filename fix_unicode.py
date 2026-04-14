#!/usr/bin/env python3
"""Fix R² Unicode issue in training scripts - replace R² with R2"""
import os

files_to_fix = [
    r'f:\Project\4_1_9_final\train_v7_complete_with_cv.py',
    r'f:\Project\4_1_9_final\train_france_v7_complete_with_cv.py',
]

for fpath in files_to_fix:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    count = content.count('R\u00b2')
    new_content = content.replace('R\u00b2', 'R2')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Fixed {os.path.basename(fpath)}: replaced {count} occurrences of R2 (superscript) with R2")

print("Done.")
