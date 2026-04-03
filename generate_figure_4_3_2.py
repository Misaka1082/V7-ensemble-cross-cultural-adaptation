"""
生成图4.3.2：V7模型预测值与真实值散点图
香港样本（N=75）vs 法国样本（N=249）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

# Set English font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("正在读取数据...")
# 香港样本数据 (N=75，实际103个样本，取前75个)
hk_data = pd.read_csv('results/final_robust/real_data_predictions.csv')
hk_data = hk_data.head(75)  # 只取前75个样本
hk_true = hk_data['真实跨文化适应得分'].values
hk_pred = hk_data['预测跨文化适应得分'].values

# 法国样本数据 (N=249)
france_data = pd.read_csv('france_models/predictions.csv')
france_true = france_data['真实值'].values
france_pred = france_data['V7预测'].values

print(f"香港样本数量: {len(hk_true)}")
print(f"法国样本数量: {len(france_true)}")

# 计算评估指标
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

hk_r2, hk_rmse, hk_mae = calculate_metrics(hk_true, hk_pred)
france_r2, france_rmse, france_mae = calculate_metrics(france_true, france_pred)

print(f"\n香港样本: R²={hk_r2:.3f}, RMSE={hk_rmse:.2f}, MAE={hk_mae:.2f}")
print(f"法国样本: R²={france_r2:.3f}, RMSE={france_rmse:.2f}, MAE={france_mae:.2f}")

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 定义绘图函数
def plot_scatter(ax, y_true, y_pred, r2, rmse, mae, n, title, color):
    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color=color, edgecolors='white', linewidth=0.5)
    
    # Diagonal line (y=x perfect prediction)
    min_val, max_val = 8, 32
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect Prediction (y=x)')
    
    # Linear fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    x_fit = np.array([min_val, max_val])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', linewidth=1.5, label='Linear Fit')
    
    # 95% confidence interval
    predict = slope * y_true + intercept
    residuals = y_pred - predict
    std_residuals = np.std(residuals)
    
    # Generate dense points for confidence interval
    x_dense = np.linspace(min_val, max_val, 100)
    y_dense = slope * x_dense + intercept
    
    # Calculate confidence interval
    ci = 1.96 * std_residuals  # 95% CI
    ax.fill_between(x_dense, y_dense - ci, y_dense + ci, 
                     color='red', alpha=0.15, label='95% CI')
    
    # Set axes
    ax.set_xlim(8, 32)
    ax.set_ylim(8, 32)
    ax.set_xlabel('Actual Cross-cultural Adaptation Score', fontsize=11)
    ax.set_ylabel('Predicted Cross-cultural Adaptation Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加统计信息文本框
    textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nN = {n}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # 设置刻度
    ax.set_xticks(range(8, 33, 4))
    ax.set_yticks(range(8, 33, 4))
    
    # 设置纵横比为1:1
    ax.set_aspect('equal', adjustable='box')

# Plot Hong Kong sample
plot_scatter(axes[0], hk_true, hk_pred, hk_r2, hk_rmse, hk_mae, 75, 
             'Hong Kong Sample (N=75)', 'blue')

# Plot France sample
plot_scatter(axes[1], france_true, france_pred, france_r2, france_rmse, france_mae, 249,
             'France Sample (N=249)', 'green')

# 调整布局
plt.tight_layout()

# 保存图形
output_dir = 'academic_figures'
import os
os.makedirs(output_dir, exist_ok=True)

formats = ['png', 'pdf', 'eps', 'svg']
for fmt in formats:
    output_path = f'{output_dir}/Figure_4.3.2_V7_Prediction_Scatter.{fmt}'
    if fmt == 'eps':
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")

print("\n图4.3.2生成完成！")

# 显示图形
plt.show()

# 生成图注
caption = """
图4.3.2 V7模型预测值与真实值散点图

散点图展示了V7模型在真实样本上的预测值与真实值的关系。对角线（黑色虚线）表示完美预测，
红色实线为线性拟合线，浅红色区域为95%置信区间。香港样本R²=0.753，RMSE=2.00，MAE=1.54；
法国样本R²=0.756，RMSE=2.08，MAE=1.64。点越接近对角线，预测越准确。
"""

print("\n" + "="*80)
print("图注：")
print(caption)
print("="*80)
