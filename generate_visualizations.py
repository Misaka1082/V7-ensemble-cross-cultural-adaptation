"""
生成V7模型的可视化图表
1. 5折交叉验证箱线图
2. SHAP summary plot
3. 预测值vs真实值散点图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import torch
import torch.nn as nn

# DeepFM模型定义
class DeepFM(nn.Module):
    def __init__(self, n_features, embedding_dim=16):
        super(DeepFM, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        self.fm_linear = nn.Linear(n_features, 1)
        self.fm_embeddings = nn.Parameter(torch.randn(n_features, embedding_dim))
        
        self.dnn = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        linear_part = self.fm_linear(x)
        
        x_expanded = x.unsqueeze(2)
        embeddings_expanded = self.fm_embeddings.unsqueeze(0).expand(x.size(0), -1, -1)
        weighted_embeddings = x_expanded * embeddings_expanded
        
        sum_square = torch.sum(weighted_embeddings, dim=1) ** 2
        square_sum = torch.sum(weighted_embeddings ** 2, dim=1)
        fm_part = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        dnn_part = self.dnn(x)
        
        return linear_part + fm_part + dnn_part

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("生成V7模型可视化图表")
print("=" * 80)

# ============================================================================
# 1. 5折交叉验证箱线图
# ============================================================================
print("\n【1/3】生成5折交叉验证箱线图...")

cv_results = pd.read_csv('F:/Project/4.1.9.final/results/cv_results_75samples.csv')

fig, ax = plt.subplots(figsize=(12, 6))
cv_results.boxplot(ax=ax)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('5折交叉验证性能对比', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticklabels(['DeepFM', 'XGBoost', 'LightGBM', 'CatBoost', 'GBM', 'RandomForest', 'V7模型'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/cv_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ 保存: cv_boxplot.png")
plt.close()

# ============================================================================
# 2. SHAP summary plot
# ============================================================================
print("\n【2/3】生成SHAP summary plot...")

# 加载SHAP值
shap_values = np.load('F:/Project/4.1.9.final/results/shap_values_75samples_cv.npy')

# 加载测试数据
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

# 中文特征名
feature_names_cn = [
    '文化保持', '社会保持', '文化接触', '社会接触', '家庭支持',
    '家庭沟通频率', '沟通坦诚度', '自主权', '社会联结感', '开放性', '来港时长'
]

X_test = df_real[feature_cols].values

# SHAP summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names_cn, show=False)
plt.title('SHAP特征重要性总结图', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/shap_summary_plot.png', dpi=300, bbox_inches='tight')
print("✓ 保存: shap_summary_plot.png")
plt.close()

# SHAP bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names_cn, plot_type='bar', show=False)
plt.title('SHAP特征重要性条形图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/shap_bar_plot.png', dpi=300, bbox_inches='tight')
print("✓ 保存: shap_bar_plot.png")
plt.close()

# ============================================================================
# 3. 预测值vs真实值散点图
# ============================================================================
print("\n【3/3】生成预测值vs真实值散点图...")

# 加载模型
fold_models = joblib.load('F:/Project/4.1.9.final/results/v7_models_75samples_5fold.pkl')

# 标准化测试数据并预测
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_predictions = []
for fold_model in fold_models:
    scaler = fold_model['scaler']
    X_test_scaled = scaler.transform(X_test)
    
    # 6个基础学习器预测
    with torch.no_grad():
        deepfm_pred = fold_model['deepfm'](torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy().flatten()
    xgb_pred = fold_model['xgb'].predict(X_test_scaled)
    lgb_pred = fold_model['lgb'].predict(X_test_scaled)
    cb_pred = fold_model['cb'].predict(X_test_scaled)
    gbm_pred = fold_model['gbm'].predict(X_test_scaled)
    rf_pred = fold_model['rf'].predict(X_test_scaled)
    
    meta_features = np.column_stack([deepfm_pred, xgb_pred, lgb_pred, cb_pred, gbm_pred, rf_pred])
    v7_pred = fold_model['meta'].predict(meta_features)
    test_predictions.append(v7_pred)

# 平均预测
final_predictions = np.mean(test_predictions, axis=0)
y_test = df_real['cross_cultural_adaptation'].values

# 计算指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
r2 = r2_score(y_test, final_predictions)
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
mae = mean_absolute_error(y_test, final_predictions)

# 绘制散点图
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, final_predictions, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

# 添加对角线
min_val = min(y_test.min(), final_predictions.min())
max_val = max(y_test.max(), final_predictions.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')

# 添加回归线
z = np.polyfit(y_test, final_predictions, 1)
p = np.poly1d(z)
ax.plot(y_test, p(y_test), 'b-', lw=2, alpha=0.8, label=f'拟合线 (y={z[0]:.2f}x+{z[1]:.2f})')

ax.set_xlabel('真实值', fontsize=12)
ax.set_ylabel('预测值', fontsize=12)
ax.set_title(f'V7模型预测性能\nR²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/prediction_scatter.png', dpi=300, bbox_inches='tight')
print("✓ 保存: prediction_scatter.png")
plt.close()

# 残差图
residuals = y_test - final_predictions
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(final_predictions, residuals, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('预测值', fontsize=12)
ax.set_ylabel('残差 (真实值 - 预测值)', fontsize=12)
ax.set_title('残差分布图', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/residual_plot.png', dpi=300, bbox_inches='tight')
print("✓ 保存: residual_plot.png")
plt.close()

print("\n" + "=" * 80)
print("所有可视化图表已生成！")
print("保存位置: F:/Project/4.1.9.final/results/")
print("=" * 80)
