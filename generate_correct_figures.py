"""
重新生成所有正确的学术图表（图4.5-4.20）
基于final_report中的正确数据和要求
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = 'academic_figures'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
print("正在读取数据...")
# 香港SHAP数据
hk_shap = pd.read_csv('results/feature_importance_75samples_cv.csv')
# 法国SHAP数据
france_shap = pd.read_csv('france_models/feature_importance_shap.csv')

# 特征中文名映射
feature_names_cn = {
    'social_connectedness': '社会联结感',
    'social_contact': '社会接触',
    'cultural_contact': '文化接触',
    'openness': '开放性',
    'family_support': '家庭支持',
    'cultural_maintenance': '文化保持',
    'months_in_hk': '居留时长',
    'months_in_france': '居留时长',
    'family_communication_frequency': '沟通频率',
    'communication_honesty': '沟通坦诚度',
    'social_maintenance': '社会保持',
    'autonomy': '自主权'
}

print("\n=== 生成图4.5：法国样本特征重要性 ===")
fig, ax = plt.subplots(figsize=(10, 8))

# 准备法国数据
france_data = france_shap.copy()
france_data['特征_cn'] = france_data['特征'].map(feature_names_cn)
france_data = france_data.sort_values('SHAP重要性', ascending=True)

# 绘制水平条形图
colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(france_data)))
bars = ax.barh(range(len(france_data)), france_data['SHAP重要性'], color=colors, edgecolor='black', linewidth=0.5)

# 添加数值标签
for i, (idx, row) in enumerate(france_data.iterrows()):
    ax.text(row['SHAP重要性'] + 0.02, i, f"{row['SHAP重要性']:.3f}", 
            va='center', fontsize=9)

ax.set_yticks(range(len(france_data)))
ax.set_yticklabels(france_data['特征_cn'], fontsize=10)
ax.set_xlabel('SHAP重要性', fontsize=11)
ax.set_title('图4.5 法国样本V7模型特征重要性（N=249）', fontsize=12, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, 1.0)

plt.tight_layout()
for fmt in ['png', 'pdf', 'eps', 'svg']:
    plt.savefig(f'{output_dir}/Figure_4.5_France_Feature_Importance.{fmt}', dpi=300, bbox_inches='tight')
print("图4.5已保存")
plt.close()

print("\n=== 生成图4.6：特征重要性跨文化对比 ===")
fig, ax = plt.subplots(figsize=(12, 8))

# 准备对比数据
hk_data = hk_shap.copy()
hk_data['特征_cn'] = hk_data['特征'].map(feature_names_cn)
hk_dict = dict(zip(hk_data['特征'], hk_data['SHAP重要性']))

france_dict = dict(zip(france_shap['特征'], france_shap['SHAP重要性']))

# 获取所有特征
all_features = list(set(hk_dict.keys()) | set(france_dict.keys()))
all_features = [f for f in all_features if f in feature_names_cn]

# 准备绘图数据
hk_values = [hk_dict.get(f, 0) for f in all_features]
france_values = [france_dict.get(f, 0) for f in all_features]
feature_labels = [feature_names_cn[f] for f in all_features]

# 按香港重要性排序
sorted_indices = np.argsort(hk_values)[::-1]
hk_values = [hk_values[i] for i in sorted_indices]
france_values = [france_values[i] for i in sorted_indices]
feature_labels = [feature_labels[i] for i in sorted_indices]

x = np.arange(len(feature_labels))
width = 0.35

bars1 = ax.bar(x - width/2, hk_values, width, label='香港(N=75)', color='steelblue', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, france_values, width, label='法国(N=249)', color='seagreen', edgecolor='black', linewidth=0.5)

ax.set_xlabel('特征', fontsize=11)
ax.set_ylabel('SHAP重要性', fontsize=11)
ax.set_title('图4.6 V7模型特征重要性跨文化对比', fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

plt.tight_layout()
for fmt in ['png', 'pdf', 'eps', 'svg']:
    plt.savefig(f'{output_dir}/Figure_4.6_Feature_Importance_Comparison.{fmt}', dpi=300, bbox_inches='tight')
print("图4.6已保存")
plt.close()

print("\n所有图表生成完成！")
print(f"图表已保存至: {output_dir}/")
