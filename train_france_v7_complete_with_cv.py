"""
法国样本完整V7模型训练流程（包含DeepFM + 5折交叉验证 + SHAP分析）
基于249个法国样本生成的10万数据（来法国时长1-48个月）
补充缺失的DeepFM模型和完整验证流程
"""
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("=" * 80)
print("法国样本V7模型完整训练流程（DeepFM + 5折交叉验证 + SHAP）")
print("=" * 80)

start_time = time.time()

# ============================================================================
# 第1步：加载数据
# ============================================================================
print("\n【步骤1/12】加载训练数据")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 加载法国生成数据
df_train = pd.read_csv('F:/Project/4_1_9_final/france_data/france_100k_48months.csv')
# 加载法国真实数据
df_real = pd.read_excel('F:/Project/4_1_9_final/france_data/france_data_filtered_48months.xlsx')

# 列名映射
column_mapping = {
    '来法国生活时长': 'months_in_france',
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
    '开放性': 'openness'
}
df_real = df_real.rename(columns=column_mapping)

feature_cols = [
    'cultural_maintenance', 'social_maintenance', 'cultural_contact',
    'social_contact', 'family_support', 'family_communication_frequency',
    'communication_honesty', 'autonomy', 'social_connectedness',
    'openness', 'months_in_france'
]

print(f"✓ 训练数据: {len(df_train):,} 样本")
print(f"✓ 真实数据: {len(df_real)} 样本")
print(f"✓ 特征数量: {len(feature_cols)}")

# ============================================================================
# 第2步：DeepFM模型定义
# ============================================================================
print("\n【步骤2/12】定义DeepFM模型架构")

class DeepFM(nn.Module):
    def __init__(self, n_features, embedding_dim=16):
        super(DeepFM, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        # FM部分：线性层
        self.fm_linear = nn.Linear(n_features, 1)
        # FM部分：嵌入层
        self.fm_embeddings = nn.Parameter(torch.randn(n_features, embedding_dim))
        
        # Deep部分：深度神经网络
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
        # FM线性部分
        linear_part = self.fm_linear(x)
        
        # FM交互部分
        x_expanded = x.unsqueeze(2)
        embeddings_expanded = self.fm_embeddings.unsqueeze(0).expand(x.size(0), -1, -1)
        weighted_embeddings = x_expanded * embeddings_expanded
        
        sum_square = torch.sum(weighted_embeddings, dim=1) ** 2
        square_sum = torch.sum(weighted_embeddings ** 2, dim=1)
        fm_part = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        # Deep部分
        dnn_part = self.dnn(x)
        
        return linear_part + fm_part + dnn_part

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ DeepFM模型已定义")
print(f"✓ 使用设备: {device}")

# ============================================================================
# 第3步：准备5折交叉验证
# ============================================================================
print("\n【步骤3/12】准备5折交叉验证")

X_full = df_train[feature_cols].values
y_full = df_train['cross_cultural_adaptation'].values
X_test = df_real[feature_cols].values
y_test = df_real['cross_cultural_adaptation'].values

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {
    'deepfm': [], 'xgb': [], 'lgb': [], 
    'cb': [], 'gbm': [], 'rf': [], 'v7': []
}

print(f"✓ 5折交叉验证设置完成")
print(f"✓ 每折训练集约: {int(len(X_full) * 0.8):,} 样本")
print(f"✓ 每折验证集约: {int(len(X_full) * 0.2):,} 样本")

# ============================================================================
# 第4-10步：5折交叉验证训练
# ============================================================================
print("\n【步骤4-10/12】5折交叉验证训练（预计需要1-2小时）")

fold_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/5 - 开始时间: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y_full[train_idx], y_full[val_idx]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. DeepFM（新增）
    print(f"\n  [{fold}/5] 训练DeepFM...")
    deepfm_model = DeepFM(len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(deepfm_model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(100):
        deepfm_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = deepfm_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        deepfm_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = deepfm_model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_deepfm_state = deepfm_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    deepfm_model.load_state_dict(best_deepfm_state)
    deepfm_model.eval()
    with torch.no_grad():
        deepfm_val_pred = deepfm_model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy().flatten()
        deepfm_test_pred = deepfm_model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy().flatten()
    
    deepfm_r2 = r2_score(y_val, deepfm_val_pred)
    cv_results['deepfm'].append(deepfm_r2)
    print(f"    DeepFM R²: {deepfm_r2:.4f}")
    
    # 2. XGBoost
    print(f"  [{fold}/5] 训练XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    xgb_val_pred = xgb_model.predict(X_val_scaled)
    xgb_test_pred = xgb_model.predict(X_test_scaled)
    xgb_r2 = r2_score(y_val, xgb_val_pred)
    cv_results['xgb'].append(xgb_r2)
    print(f"    XGBoost R²: {xgb_r2:.4f}")
    
    # 3. LightGBM
    print(f"  [{fold}/5] 训练LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.1, num_leaves=31,
        feature_fraction=0.7, bagging_fraction=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    lgb_val_pred = lgb_model.predict(X_val_scaled)
    lgb_test_pred = lgb_model.predict(X_test_scaled)
    lgb_r2 = r2_score(y_val, lgb_val_pred)
    cv_results['lgb'].append(lgb_r2)
    print(f"    LightGBM R²: {lgb_r2:.4f}")
    
    # 4. CatBoost
    print(f"  [{fold}/5] 训练CatBoost...")
    cb_model = cb.CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, random_state=42, verbose=False
    )
    cb_model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val), early_stopping_rounds=50)
    cb_val_pred = cb_model.predict(X_val_scaled)
    cb_test_pred = cb_model.predict(X_test_scaled)
    cb_r2 = r2_score(y_val, cb_val_pred)
    cv_results['cb'].append(cb_r2)
    print(f"    CatBoost R²: {cb_r2:.4f}")
    
    # 5. GBM
    print(f"  [{fold}/5] 训练GBM...")
    gbm_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=42
    )
    gbm_model.fit(X_train_scaled, y_train)
    gbm_val_pred = gbm_model.predict(X_val_scaled)
    gbm_test_pred = gbm_model.predict(X_test_scaled)
    gbm_r2 = r2_score(y_val, gbm_val_pred)
    cv_results['gbm'].append(gbm_r2)
    print(f"    GBM R²: {gbm_r2:.4f}")
    
    # 6. RandomForest
    print(f"  [{fold}/5] 训练RandomForest...")
    rf_model = RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_split=5,
        max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_val_pred = rf_model.predict(X_val_scaled)
    rf_test_pred = rf_model.predict(X_test_scaled)
    rf_r2 = r2_score(y_val, rf_val_pred)
    cv_results['rf'].append(rf_r2)
    print(f"    RandomForest R²: {rf_r2:.4f}")
    
    # 7. Stacking集成（V7模型）
    print(f"  [{fold}/5] Stacking集成...")
    meta_train = np.column_stack([
        deepfm_val_pred, xgb_val_pred, lgb_val_pred,
        cb_val_pred, gbm_val_pred, rf_val_pred
    ])
    meta_test = np.column_stack([
        deepfm_test_pred, xgb_test_pred, lgb_test_pred,
        cb_test_pred, gbm_test_pred, rf_test_pred
    ])
    
    meta_model = LinearRegression()
    meta_model.fit(meta_train, y_val)
    v7_val_pred = meta_model.predict(meta_train)
    v7_test_pred = meta_model.predict(meta_test)
    
    v7_r2 = r2_score(y_val, v7_val_pred)
    cv_results['v7'].append(v7_r2)
    print(f"    V7模型 R²: {v7_r2:.4f}")
    
    # 保存模型
    fold_models.append({
        'scaler': scaler,
        'deepfm': deepfm_model,
        'xgb': xgb_model,
        'lgb': lgb_model,
        'cb': cb_model,
        'gbm': gbm_model,
        'rf': rf_model,
        'meta': meta_model
    })
    
    print(f"\nFold {fold}/5 完成 - 累计耗时: {(time.time() - start_time)/60:.1f}分钟")

# ============================================================================
# 第11步：交叉验证结果汇总
# ============================================================================
print("\n" + "=" * 80)
print("【步骤11/12】5折交叉验证结果汇总")
print("=" * 80)

cv_summary = pd.DataFrame(cv_results)
print("\n各模型5折交叉验证R²:")
print(cv_summary.to_string(index=False))

print("\n平均R²和标准差:")
for model_name in cv_results.keys():
    mean_r2 = np.mean(cv_results[model_name])
    std_r2 = np.std(cv_results[model_name])
    print(f"  {model_name:12s}: {mean_r2:.4f} ± {std_r2:.4f}")

# 保存交叉验证结果
cv_summary.to_csv('F:/Project/4_1_9_final/france_models/cv_results_france.csv', index=False)
print("\n✓ 交叉验证结果已保存")

# ============================================================================
# 第12步：SHAP可解释性分析
# ============================================================================
print("\n【步骤12/12】SHAP可解释性分析")
print("计算SHAP值（使用第1折模型）...")

# 使用第1折的模型进行SHAP分析
model_for_shap = fold_models[0]
X_test_scaled_shap = model_for_shap['scaler'].transform(X_test)

# 对每个树模型计算SHAP值
shap_values_dict = {}
print("  计算各模型SHAP值...")

for model_name, model in [('lgb', model_for_shap['lgb']),
                          ('cb', model_for_shap['cb']),
                          ('gbm', model_for_shap['gbm']),
                          ('rf', model_for_shap['rf'])]:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled_shap)
        shap_values_dict[model_name] = shap_values
        print(f"    {model_name} SHAP值已计算")
    except Exception as e:
        print(f"    {model_name} SHAP计算失败: {str(e)[:50]}")

# 加权平均SHAP值（使用成功计算的模型）
if len(shap_values_dict) > 0:
    weights = np.ones(len(shap_values_dict)) / len(shap_values_dict)
    shap_values_list = list(shap_values_dict.values())
    shap_values_ensemble = sum(w * sv for w, sv in zip(weights, shap_values_list))
else:
    print("  警告：无法计算SHAP值，跳过SHAP分析")
    shap_values_ensemble = None

# 计算特征重要性
feature_importance = np.abs(shap_values_ensemble).mean(axis=0)
importance_df = pd.DataFrame({
    '特征': feature_cols,
    'SHAP重要性': feature_importance
}).sort_values('SHAP重要性', ascending=False)

print("\n特征重要性排名（基于SHAP值）:")
print(importance_df.to_string(index=False))

# 保存SHAP值和特征重要性
np.save('F:/Project/4_1_9_final/france_models/shap_values_france.npy', shap_values_ensemble)
importance_df.to_csv('F:/Project/4_1_9_final/france_models/feature_importance_shap.csv', index=False)

# 生成SHAP可视化
print("\n生成SHAP可视化图表...")

# SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_ensemble, X_test_scaled_shap, feature_names=feature_cols, show=False)
plt.title('法国样本SHAP Summary Plot', fontsize=14, fontproperties='SimHei')
plt.tight_layout()
plt.savefig('F:/Project/4_1_9_final/france_models/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# SHAP Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_ensemble, X_test_scaled_shap, feature_names=feature_cols, 
                  plot_type='bar', show=False)
plt.title('法国样本特征重要性（SHAP）', fontsize=14, fontproperties='SimHei')
plt.tight_layout()
plt.savefig('F:/Project/4_1_9_final/france_models/shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ SHAP可视化已保存")

# ============================================================================
# 第13步：在测试集上评估最终性能
# ============================================================================
print("\n【步骤13/13】在测试集上评估最终性能")

# 使用5个fold模型的平均预测
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

# 平均5个fold的预测
final_test_pred = np.mean(test_predictions, axis=0)

final_r2 = r2_score(y_test, final_test_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, final_test_pred))
final_mae = mean_absolute_error(y_test, final_test_pred)

print(f"\n最终测试集性能（249个法国样本，5折平均）:")
print(f"  R² = {final_r2:.4f}")
print(f"  RMSE = {final_rmse:.4f}")
print(f"  MAE = {final_mae:.4f}")

# 保存预测结果
predictions_df = pd.DataFrame({
    '真实值': y_test,
    '预测值': final_test_pred,
    '残差': y_test - final_test_pred
})
predictions_df.to_csv('F:/Project/4_1_9_final/france_models/predictions_france_cv.csv', index=False)

# ============================================================================
# 第14步：生成最终报告
# ============================================================================
print("\n【步骤14/14】生成最终报告")

elapsed_time = time.time() - start_time

report = f"""
{'='*80}
法国样本V7模型训练完成报告（DeepFM + 5折交叉验证 + SHAP）
{'='*80}

训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
总耗时: {elapsed_time/60:.1f} 分钟

【数据规模】
  训练数据: {len(df_train):,} 样本
  测试数据: {len(df_real)} 样本（真实法国数据）
  特征数量: {len(feature_cols)}

【5折交叉验证结果】
平均R² ± 标准差:
  DeepFM:      {np.mean(cv_results['deepfm']):.4f} ± {np.std(cv_results['deepfm']):.4f}
  XGBoost:     {np.mean(cv_results['xgb']):.4f} ± {np.std(cv_results['xgb']):.4f}
  LightGBM:    {np.mean(cv_results['lgb']):.4f} ± {np.std(cv_results['lgb']):.4f}
  CatBoost:    {np.mean(cv_results['cb']):.4f} ± {np.std(cv_results['cb']):.4f}
  GBM:         {np.mean(cv_results['gbm']):.4f} ± {np.std(cv_results['gbm']):.4f}
  RandomForest:{np.mean(cv_results['rf']):.4f} ± {np.std(cv_results['rf']):.4f}
  V7模型:      {np.mean(cv_results['v7']):.4f} ± {np.std(cv_results['v7']):.4f}

【最终测试集性能】
  R² = {final_r2:.4f}
  RMSE = {final_rmse:.4f}
  MAE = {final_mae:.4f}

【Top 5 重要特征（基于SHAP值）】
{importance_df.head(5).to_string(index=False)}

【补充的功能】
  ✓ DeepFM模型（之前缺失）
  ✓ 5折交叉验证流程（之前缺失）
  ✓ SHAP可解释性分析（之前缺失）

【保存的文件】
  1. france_v7_models_5fold.pkl - 5个fold的完整模型
  2. cv_results_france.csv - 交叉验证结果
  3. shap_values_france.npy - SHAP值
  4. feature_importance_shap.csv - 特征重要性（SHAP）
  5. shap_summary_plot.png - SHAP摘要图
  6. shap_bar_plot.png - SHAP条形图
  7. predictions_france_cv.csv - 预测结果

【方法论完整性】
现在法国样本具备与香港样本相同的完整V7模型流程：
  ✓ DeepFM深度因子分解机
  ✓ XGBoost、LightGBM、CatBoost、GBM、RandomForest
  ✓ Stacking集成学习
  ✓ 5折交叉验证
  ✓ SHAP可解释性分析

这确保了跨文化研究的方法论一致性，消除了验证缺陷。

{'='*80}
"""

print(report)

# 保存报告
with open('F:/Project/4_1_9_final/france_models/france_v7_final_report_complete.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# 保存模型
joblib.dump(fold_models, 'F:/Project/4_1_9_final/france_models/france_v7_models_5fold.pkl')

print("\n✓ 所有文件已保存到 F:/Project/4_1_9_final/france_models/")
print("=" * 80)
print("训练完成！法国样本现已具备完整的V7模型流程。")
print("=" * 80)
