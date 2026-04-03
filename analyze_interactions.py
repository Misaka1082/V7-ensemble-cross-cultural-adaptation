"""
V7模型2阶和3阶交互效应分析
用于理论创新和未来研究
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("V7模型交互效应分析")
print("=" * 80)

# ============================================================================
# 加载数据
# ============================================================================
print("\n【步骤1/5】加载数据...")

df_real = pd.read_excel('F:/Project/4.1.9.final/data/processed/real_data_filtered_48months.xlsx')
column_mapping = {
    '跨文化适应程度': 'cross_cultural_adaptation',
    '文化保持': 'cultural_maintenance',
    '社会保持': 'social_maintenance',
    '文化接触': 'cultural_contact',
    '社会接触': 'social_contact',
    '家庭支持': 'family_support',
    '家庭沟通频率': 'family_communication_frequency',
    '沟通坦诚度': 'communication_honesty',
    '自主权': 'autonomy',
    '社会联结感': 'social_connectedness',
    '开放性': 'openness',
    '来港时长': 'months_in_hk'
}
df_real = df_real.rename(columns=column_mapping)

feature_cols = [
    'cultural_maintenance', 'social_maintenance', 'cultural_contact',
    'social_contact', 'family_support', 'family_communication_frequency',
    'communication_honesty', 'autonomy', 'social_connectedness',
    'openness', 'months_in_hk'
]

feature_names_cn = {
    'cultural_maintenance': '文化保持',
    'social_maintenance': '社会保持',
    'cultural_contact': '文化接触',
    'social_contact': '社会接触',
    'family_support': '家庭支持',
    'family_communication_frequency': '家庭沟通频率',
    'communication_honesty': '沟通坦诚度',
    'autonomy': '自主权',
    'social_connectedness': '社会联结感',
    'openness': '开放性',
    'months_in_hk': '来港时长'
}

X = df_real[feature_cols].values
y = df_real['cross_cultural_adaptation'].values

print(f"✓ 样本数: {len(X)}")
print(f"✓ 特征数: {len(feature_cols)}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# 基线模型（仅主效应）
# ============================================================================
print("\n【步骤2/5】建立基线模型（仅主效应）...")

lr_baseline = LinearRegression()
lr_baseline.fit(X_scaled, y)
y_pred_baseline = lr_baseline.predict(X_scaled)
r2_baseline = r2_score(y, y_pred_baseline)

print(f"✓ 基线模型R²: {r2_baseline:.4f}")

# ============================================================================
# 2阶交互效应分析
# ============================================================================
print("\n【步骤3/5】分析2阶交互效应...")

# 生成所有2阶交互项
two_way_interactions = []
interaction_names = []
interaction_r2_improvements = []

for i, j in combinations(range(len(feature_cols)), 2):
    # 创建交互项
    interaction = X_scaled[:, i] * X_scaled[:, j]
    
    # 添加到特征矩阵
    X_with_interaction = np.column_stack([X_scaled, interaction])
    
    # 训练模型
    lr = LinearRegression()
    lr.fit(X_with_interaction, y)
    y_pred = lr.predict(X_with_interaction)
    r2 = r2_score(y, y_pred)
    
    # 计算R²提升
    r2_improvement = r2 - r2_baseline
    
    two_way_interactions.append({
        'feature1': feature_cols[i],
        'feature2': feature_cols[j],
        'feature1_cn': feature_names_cn[feature_cols[i]],
        'feature2_cn': feature_names_cn[feature_cols[j]],
        'r2': r2,
        'r2_improvement': r2_improvement,
        'coefficient': lr.coef_[-1]
    })

# 转换为DataFrame并排序
df_2way = pd.DataFrame(two_way_interactions)
df_2way = df_2way.sort_values('r2_improvement', ascending=False)

print(f"\n✓ 共分析了 {len(df_2way)} 个2阶交互项")
print(f"\nTop 10 最重要的2阶交互:")
print(df_2way.head(10)[['feature1_cn', 'feature2_cn', 'r2_improvement', 'coefficient']].to_string(index=False))

# 保存结果
df_2way.to_csv('F:/Project/4.1.9.final/results/two_way_interactions.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ 保存: two_way_interactions.csv")

# ============================================================================
# 3阶交互效应分析（仅分析Top 20的2阶交互相关的特征）
# ============================================================================
print("\n【步骤4/5】分析3阶交互效应...")

# 获取Top 20 2阶交互中涉及的特征
top_features = set()
for _, row in df_2way.head(20).iterrows():
    top_features.add(row['feature1'])
    top_features.add(row['feature2'])

top_features = list(top_features)
top_feature_indices = [feature_cols.index(f) for f in top_features]

print(f"✓ 基于Top 20 2阶交互，选择了 {len(top_features)} 个关键特征进行3阶分析")

# 生成3阶交互项（仅在关键特征中）
three_way_interactions = []

for i, j, k in combinations(top_feature_indices, 3):
    # 创建交互项
    interaction = X_scaled[:, i] * X_scaled[:, j] * X_scaled[:, k]
    
    # 添加到特征矩阵
    X_with_interaction = np.column_stack([X_scaled, interaction])
    
    # 训练模型
    lr = LinearRegression()
    lr.fit(X_with_interaction, y)
    y_pred = lr.predict(X_with_interaction)
    r2 = r2_score(y, y_pred)
    
    # 计算R²提升
    r2_improvement = r2 - r2_baseline
    
    three_way_interactions.append({
        'feature1': feature_cols[i],
        'feature2': feature_cols[j],
        'feature3': feature_cols[k],
        'feature1_cn': feature_names_cn[feature_cols[i]],
        'feature2_cn': feature_names_cn[feature_cols[j]],
        'feature3_cn': feature_names_cn[feature_cols[k]],
        'r2': r2,
        'r2_improvement': r2_improvement,
        'coefficient': lr.coef_[-1]
    })

# 转换为DataFrame并排序
df_3way = pd.DataFrame(three_way_interactions)
df_3way = df_3way.sort_values('r2_improvement', ascending=False)

print(f"\n✓ 共分析了 {len(df_3way)} 个3阶交互项")
print(f"\nTop 10 最重要的3阶交互:")
print(df_3way.head(10)[['feature1_cn', 'feature2_cn', 'feature3_cn', 'r2_improvement', 'coefficient']].to_string(index=False))

# 保存结果
df_3way.to_csv('F:/Project/4.1.9.final/results/three_way_interactions.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ 保存: three_way_interactions.csv")

# ============================================================================
# 可视化
# ============================================================================
print("\n【步骤5/5】生成可视化图表...")

# 1. 2阶交互热力图
print("  生成2阶交互热力图...")
interaction_matrix = np.zeros((len(feature_cols), len(feature_cols)))
for _, row in df_2way.iterrows():
    i = feature_cols.index(row['feature1'])
    j = feature_cols.index(row['feature2'])
    interaction_matrix[i, j] = row['r2_improvement']
    interaction_matrix[j, i] = row['r2_improvement']

plt.figure(figsize=(12, 10))
sns.heatmap(interaction_matrix, 
            xticklabels=[feature_names_cn[f] for f in feature_cols],
            yticklabels=[feature_names_cn[f] for f in feature_cols],
            annot=True, fmt='.4f', cmap='YlOrRd', center=0)
plt.title('2阶交互效应热力图（R²提升）', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/interaction_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: interaction_heatmap.png")
plt.close()

# 2. Top 20 2阶交互条形图
print("  生成Top 20 2阶交互条形图...")
top20_2way = df_2way.head(20).copy()
top20_2way['interaction_name'] = top20_2way['feature1_cn'] + ' × ' + top20_2way['feature2_cn']

plt.figure(figsize=(12, 8))
plt.barh(range(len(top20_2way)), top20_2way['r2_improvement'].values)
plt.yticks(range(len(top20_2way)), top20_2way['interaction_name'].values)
plt.xlabel('R² 提升', fontsize=12)
plt.title('Top 20 最重要的2阶交互效应', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/top20_2way_interactions.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: top20_2way_interactions.png")
plt.close()

# 3. Top 15 3阶交互条形图
print("  生成Top 15 3阶交互条形图...")
top15_3way = df_3way.head(15).copy()
top15_3way['interaction_name'] = (top15_3way['feature1_cn'] + ' × ' + 
                                   top15_3way['feature2_cn'] + ' × ' + 
                                   top15_3way['feature3_cn'])

plt.figure(figsize=(12, 8))
plt.barh(range(len(top15_3way)), top15_3way['r2_improvement'].values, color='steelblue')
plt.yticks(range(len(top15_3way)), top15_3way['interaction_name'].values, fontsize=9)
plt.xlabel('R² 提升', fontsize=12)
plt.title('Top 15 最重要的3阶交互效应', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/top15_3way_interactions.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: top15_3way_interactions.png")
plt.close()

# 4. 交互效应对比图
print("  生成交互效应对比图...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 2阶交互分布
ax1.hist(df_2way['r2_improvement'].values, bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(df_2way['r2_improvement'].mean(), color='r', linestyle='--', linewidth=2, 
            label=f'平均值: {df_2way["r2_improvement"].mean():.4f}')
ax1.set_xlabel('R² 提升', fontsize=12)
ax1.set_ylabel('频数', fontsize=12)
ax1.set_title('2阶交互效应分布', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3阶交互分布
ax2.hist(df_3way['r2_improvement'].values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax2.axvline(df_3way['r2_improvement'].mean(), color='r', linestyle='--', linewidth=2,
            label=f'平均值: {df_3way["r2_improvement"].mean():.4f}')
ax2.set_xlabel('R² 提升', fontsize=12)
ax2.set_ylabel('频数', fontsize=12)
ax2.set_title('3阶交互效应分布', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/interaction_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: interaction_distribution.png")
plt.close()

# ============================================================================
# 生成详细报告
# ============================================================================
print("\n【生成详细报告】...")

report = f"""
{'='*80}
V7模型交互效应分析报告
{'='*80}

一、分析概况
-----------
样本数: {len(X)}
特征数: {len(feature_cols)}
基线模型R² (仅主效应): {r2_baseline:.4f}

二、2阶交互效应分析
-----------------
分析的交互项数量: {len(df_2way)}
平均R²提升: {df_2way['r2_improvement'].mean():.4f}
最大R²提升: {df_2way['r2_improvement'].max():.4f}
最小R²提升: {df_2way['r2_improvement'].min():.4f}

Top 10 最重要的2阶交互:
{df_2way.head(10)[['feature1_cn', 'feature2_cn', 'r2_improvement', 'coefficient']].to_string(index=False)}

三、3阶交互效应分析
-----------------
分析的交互项数量: {len(df_3way)}
平均R²提升: {df_3way['r2_improvement'].mean():.4f}
最大R²提升: {df_3way['r2_improvement'].max():.4f}
最小R²提升: {df_3way['r2_improvement'].min():.4f}

Top 10 最重要的3阶交互:
{df_3way.head(10)[['feature1_cn', 'feature2_cn', 'feature3_cn', 'r2_improvement', 'coefficient']].to_string(index=False)}

四、理论创新建议
--------------
基于交互效应分析，以下是未来理论创新的方向：

1. **最强2阶交互** ({df_2way.iloc[0]['feature1_cn']} × {df_2way.iloc[0]['feature2_cn']})
   - R²提升: {df_2way.iloc[0]['r2_improvement']:.4f}
   - 系数: {df_2way.iloc[0]['coefficient']:.4f}
   - 理论意义: 这两个因素的协同作用对跨文化适应有显著影响

2. **最强3阶交互** ({df_3way.iloc[0]['feature1_cn']} × {df_3way.iloc[0]['feature2_cn']} × {df_3way.iloc[0]['feature3_cn']})
   - R²提升: {df_3way.iloc[0]['r2_improvement']:.4f}
   - 系数: {df_3way.iloc[0]['coefficient']:.4f}
   - 理论意义: 三个因素的复杂交互揭示了跨文化适应的多层次机制

3. **涉及"开放性"的重要交互**
"""

# 添加涉及开放性的交互
openness_2way = df_2way[(df_2way['feature1'] == 'openness') | (df_2way['feature2'] == 'openness')].head(5)
report += f"\n   2阶交互:\n"
for _, row in openness_2way.iterrows():
    report += f"   - {row['feature1_cn']} × {row['feature2_cn']}: R²提升={row['r2_improvement']:.4f}\n"

openness_3way = df_3way[(df_3way['feature1'] == 'openness') | 
                        (df_3way['feature2'] == 'openness') | 
                        (df_3way['feature3'] == 'openness')].head(5)
report += f"\n   3阶交互:\n"
for _, row in openness_3way.iterrows():
    report += f"   - {row['feature1_cn']} × {row['feature2_cn']} × {row['feature3_cn']}: R²提升={row['r2_improvement']:.4f}\n"

report += f"""

五、保存的文件
------------
1. two_way_interactions.csv - 所有2阶交互详细结果
2. three_way_interactions.csv - 所有3阶交互详细结果
3. interaction_heatmap.png - 2阶交互热力图
4. top20_2way_interactions.png - Top 20 2阶交互条形图
5. top15_3way_interactions.png - Top 15 3阶交互条形图
6. interaction_distribution.png - 交互效应分布图

{'='*80}
"""

# 保存报告
with open('F:/Project/4.1.9.final/results/interaction_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print("\n✓ 保存: interaction_analysis_report.txt")

print("\n" + "=" * 80)
print("交互效应分析完成！")
print("=" * 80)
