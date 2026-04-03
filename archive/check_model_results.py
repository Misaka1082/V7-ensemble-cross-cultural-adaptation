#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查已保存的模型结果"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

pred_df = pd.read_csv(Path(__file__).parent / 'results/final_robust/real_data_predictions.csv')
y_real = pred_df['真实跨文化适应得分'].values
y_pred = pred_df['预测跨文化适应得分'].values
errors = pred_df['绝对误差'].values

r2 = r2_score(y_real, y_pred)
rmse = np.sqrt(np.mean((y_real - y_pred)**2))
mae = np.mean(errors)

print('=== 已保存模型在103个真实样本上的表现 ===')
print(f'R2: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'最大误差: {errors.max():.2f}')
print(f'误差<1分: {(errors<1).sum()} 个 ({(errors<1).mean()*100:.1f}%)')
print(f'误差<2分: {(errors<2).sum()} 个 ({(errors<2).mean()*100:.1f}%)')
print(f'误差<3分: {(errors<3).sum()} 个 ({(errors<3).mean()*100:.1f}%)')
print()
print(f'真实得分范围: {y_real.min():.0f} - {y_real.max():.0f}')
print(f'预测得分范围: {y_pred.min():.1f} - {y_pred.max():.1f}')
