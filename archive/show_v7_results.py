#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
from pathlib import Path

v7 = json.loads(Path(__file__).parent.joinpath('results/v7_ultimate/v7_results.json').read_text(encoding='utf-8'))
folds = v7.get('fold_results', [])

# Compute overall from fold results
all_r2 = [f['ensemble_r2'] for f in folds]
avg_r2 = np.mean(all_r2)
std_r2 = np.std(all_r2)

print('=== V7 Ultimate 汇总 ===')
print('平均R2: {:.4f} +- {:.4f}'.format(avg_r2, std_r2))
print()
print('各折R2:')
for fold in folds:
    print('  第{}折: {:.4f}  RMSE={:.4f}  MAE={:.4f}'.format(
        fold['fold'], fold['ensemble_r2'], fold['ensemble_rmse'], fold['ensemble_mae']))

# Check if summary exists with data
summary = v7.get('summary', {})
if summary.get('overall_r2', 0) > 0:
    print()
    print('整体R2:', round(summary.get('overall_r2', 0), 4))
    print('整体RMSE:', round(summary.get('overall_rmse', 0), 4))
    print('整体MAE:', round(summary.get('overall_mae', 0), 4))
else:
    print()
    print('注：summary字段为空，以上为各折平均值')
