#!/usr/bin/env python3
"""
V7 Stacking过拟合检验
======================
核心问题：Stacking的R²=0.784是否存在过拟合？

关键分析：
1. Stacking使用的是OOF（Out-of-Fold）预测，理论上无数据泄露
   - 每折的基模型预测是在该折训练集上训练，在测试集上预测
   - 元学习器用全部OOF预测拟合，但这是"伪泄露"：
     元学习器在全部103样本上拟合，然后在同样的103样本上评估
     → 这是元学习器层面的过拟合！

2. 正确的Stacking评估需要嵌套CV（Nested CV）：
   - 外层CV：评估整体性能
   - 内层CV：训练基模型+元学习器

3. 当前实现的问题：
   - 基模型：5折CV，OOF预测无泄露 ✓
   - 元学习器：在全部103样本上拟合，在同样103样本上评估 ✗（过拟合！）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.model_utils import ModelUtils

# ============================================================
# 配置（与V7相同）
# ============================================================
FEATURE_MAPPING_CN_TO_EN = {
    '序号': 'sample_id', '跨文化适应程度': 'cross_cultural_adaptation',
    '文化保持': 'cultural_maintenance', '社会保持': 'social_maintenance',
    '文化接触': 'cultural_contact', '社会接触': 'social_contact',
    '家庭支持': 'family_support', '家庭沟通频率': 'comm_frequency_feeling',
    '沟通坦诚度': 'comm_openness', '自主权': 'personal_autonomy',
    '社会联结感': 'social_connection', '开放性': 'openness', '来港时长': 'months_in_hk'
}

INPUT_FEATURES = [
    'months_in_hk', 'cultural_maintenance', 'cultural_contact',
    'social_maintenance', 'social_contact', 'social_connection',
    'family_support', 'comm_frequency_feeling', 'comm_openness',
    'personal_autonomy', 'openness',
]
TARGET_COL = 'cross_cultural_adaptation'

INTERACTION_PAIRS_2WAY = [
    ('cultural_contact', 'openness'),
    ('family_support', 'openness'),
    ('social_contact', 'openness'),
]
INTERACTION_TRIPLES_3WAY = [
    ('social_contact', 'family_support', 'social_connection'),
    ('social_contact', 'personal_autonomy', 'months_in_hk'),
    ('cultural_contact', 'family_support', 'social_connection'),
]
N_INTERACTIONS = len(INTERACTION_PAIRS_2WAY) + len(INTERACTION_TRIPLES_3WAY) + 1


def compute_interaction_features(X, feature_names=INPUT_FEATURES):
    idx = {name: i for i, name in enumerate(feature_names)}
    interactions = []
    for f1, f2 in INTERACTION_PAIRS_2WAY:
        interactions.append(X[:, idx[f1]] * X[:, idx[f2]])
    for f1, f2, f3 in INTERACTION_TRIPLES_3WAY:
        interactions.append(X[:, idx[f1]] * X[:, idx[f2]] * X[:, idx[f3]])
    interactions.append(X[:, idx['openness']] ** 2)
    return np.column_stack(interactions)


class FlexibleDeepFM(nn.Module):
    def __init__(self, n_features=11, n_interactions=N_INTERACTIONS,
                 embedding_size=16, hidden_layers=None, dropout=0.25):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]
        self.n_features = n_features
        total_in = n_features + n_interactions
        self.linear = nn.Linear(total_in, 1)
        self.fm_embeddings = nn.Parameter(torch.randn(n_features, embedding_size) * 0.01)
        layers = []
        in_dim = total_in
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*layers)
        self.gate = nn.Sequential(nn.Linear(total_in, 32), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(32, 3), nn.Softmax(dim=1))
        self.bias = nn.Parameter(torch.zeros(1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_orig = x[:, :self.n_features]
        linear_out = self.linear(x)
        emb = x_orig.unsqueeze(2) * self.fm_embeddings.unsqueeze(0)
        sum_emb = emb.sum(dim=1)
        sum_sq = (emb ** 2).sum(dim=1)
        fm_out = 0.5 * (sum_emb ** 2 - sum_sq).sum(dim=1, keepdim=True)
        deep_out = self.deep(x)
        gate_weights = self.gate(x)
        combined = torch.cat([linear_out, fm_out, deep_out], dim=1)
        return (combined * gate_weights).sum(dim=1, keepdim=True) + self.bias


def train_deepfm_quick(X_train, y_train, X_val, y_val, seed=42, device='cpu',
                        hidden_layers=None, embedding_size=16, dropout=0.25,
                        lr=0.001, weight_decay=0.015, epochs=200, patience=30):
    """快速训练DeepFM"""
    ModelUtils.set_seed(seed)
    if hidden_layers is None:
        hidden_layers = [256, 128, 64]
    model = FlexibleDeepFM(len(INPUT_FEATURES), N_INTERACTIONS,
                           embedding_size, hidden_layers, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)),
        batch_size=512, shuffle=True, drop_last=True
    )
    val_X = torch.FloatTensor(X_val).to(device)
    best_r2, best_state, pat = -999, None, 0
    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(val_X).cpu().numpy().flatten()
        r2 = r2_score(y_val, preds)
        if r2 > best_r2 + 0.001:
            best_r2 = r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_r2


def get_base_model_preds_oof(X_real_scaled_full, y_real, X_gen_full, y_gen,
                              device, n_folds=5, n_repeat=30):
    """
    获取基模型的OOF预测（用于Stacking）
    每折：在(生成数据 + 训练折真实数据)上训练，在测试折真实数据上预测
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    model_names = ['deepfm', 'xgb', 'lgb', 'cat', 'gbm', 'rf']
    oof_preds = {name: np.zeros(len(y_real)) for name in model_names}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_real_scaled_full), 1):
        X_train_real = X_real_scaled_full[train_idx]
        y_train_real = y_real[train_idx]
        X_test_real = X_real_scaled_full[test_idx]
        y_test_real = y_real[test_idx]

        X_train_aug = np.vstack([X_gen_full] + [X_train_real] * n_repeat)
        y_train_aug = np.concatenate([y_gen] + [y_train_real] * n_repeat)
        perm = np.random.RandomState(42 + fold).permutation(len(X_train_aug))
        X_train_aug = X_train_aug[perm]
        y_train_aug = y_train_aug[perm]

        # DeepFM（3个取平均，加快速度）
        dfm_preds = []
        for m in range(3):
            model, _ = train_deepfm_quick(X_train_aug, y_train_aug, X_test_real, y_test_real,
                                          seed=42 + fold * 100 + m, device=device)
            model.eval()
            with torch.no_grad():
                p = model(torch.FloatTensor(X_test_real).to(device)).cpu().numpy().flatten()
            dfm_preds.append(p)
        oof_preds['deepfm'][test_idx] = np.mean(dfm_preds, axis=0)

        # XGBoost（默认参数，快速）
        xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, random_state=42, verbosity=0, n_jobs=-1)
        xgb_m.fit(X_train_aug, y_train_aug)
        oof_preds['xgb'][test_idx] = xgb_m.predict(X_test_real)

        # LightGBM
        lgb_m = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, random_state=42, verbose=-1, n_jobs=-1)
        lgb_m.fit(X_train_aug, y_train_aug)
        oof_preds['lgb'][test_idx] = lgb_m.predict(X_test_real)

        # CatBoost
        cat_m = cb.CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05,
                                      random_seed=42, verbose=0)
        cat_m.fit(X_train_aug, y_train_aug)
        oof_preds['cat'][test_idx] = cat_m.predict(X_test_real)

        # GBM
        gbm_m = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, random_state=42)
        gbm_m.fit(X_train_aug, y_train_aug)
        oof_preds['gbm'][test_idx] = gbm_m.predict(X_test_real)

        # RF
        rf_m = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1)
        rf_m.fit(X_train_aug, y_train_aug)
        oof_preds['rf'][test_idx] = rf_m.predict(X_test_real)

        print(f"  折{fold}: DeepFM={r2_score(y_test_real, oof_preds['deepfm'][test_idx]):.3f}, "
              f"XGB={r2_score(y_test_real, oof_preds['xgb'][test_idx]):.3f}, "
              f"LGB={r2_score(y_test_real, oof_preds['lgb'][test_idx]):.3f}")

    return oof_preds


def main():
    print("=" * 80)
    print("V7 Stacking过拟合检验 + 嵌套CV验证")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载数据
    real_path = Path(__file__).parent / "data" / "processed" / "real_data_103.xlsx"
    gen_path = Path(__file__).parent / "data" / "processed" / "improved_100k.csv"
    real_data = pd.read_excel(real_path).rename(columns=FEATURE_MAPPING_CN_TO_EN)
    gen_data = pd.read_csv(gen_path)

    cols = INPUT_FEATURES + [TARGET_COL]
    real_data = real_data[cols].dropna()
    gen_data = gen_data[cols].dropna()

    X_real_raw = real_data[INPUT_FEATURES].values
    y_real = real_data[TARGET_COL].values

    # ============================================================
    # 分析1：V7 Stacking的过拟合问题
    # ============================================================
    print("\n" + "=" * 80)
    print("分析1：V7 Stacking的过拟合问题诊断")
    print("=" * 80)

    print("""
  问题：V7 Stacking的R²=0.784是否可信？
  
  当前实现：
    - 基模型：5折CV OOF预测（无泄露）✓
    - 元学习器：在全部103样本OOF预测上拟合，在同样103样本上评估 ✗
    
  这是"元学习器层面的过拟合"：
    - 元学习器（RidgeCV）在103个样本上拟合，然后在同样103个样本上评估
    - 即使RidgeCV有正则化，仍然存在乐观偏差
    - 正确做法：嵌套CV（外层CV评估，内层CV训练基模型+元学习器）
    
  预期：嵌套CV的R²会低于0.784，但高于加权集成的0.728
    """)

    # ============================================================
    # 分析2：嵌套CV（正确的Stacking评估）
    # ============================================================
    print("=" * 80)
    print("分析2：嵌套CV（正确评估Stacking）")
    print("=" * 80)
    print("  外层5折CV + 内层5折CV（训练基模型+元学习器）")
    print("  注意：这会运行5×5=25次基模型训练，耗时较长")
    print()

    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    nested_preds = np.zeros(len(y_real))
    nested_fold_r2 = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(X_real_raw), 1):
        print(f"  外层第{outer_fold}折 (训练:{len(outer_train_idx)}, 测试:{len(outer_test_idx)})")

        X_outer_train = X_real_raw[outer_train_idx]
        y_outer_train = y_real[outer_train_idx]
        X_outer_test = X_real_raw[outer_test_idx]
        y_outer_test = y_real[outer_test_idx]

        # 标准化（基于外层训练集）
        scalers = {}
        for i, col in enumerate(INPUT_FEATURES):
            sc = StandardScaler()
            sc.fit(X_outer_train[:, i:i+1])
            scalers[col] = sc

        X_outer_train_scaled = np.zeros_like(X_outer_train, dtype=np.float64)
        X_outer_test_scaled = np.zeros_like(X_outer_test, dtype=np.float64)
        for i, col in enumerate(INPUT_FEATURES):
            X_outer_train_scaled[:, i] = scalers[col].transform(X_outer_train[:, i:i+1]).flatten()
            X_outer_test_scaled[:, i] = scalers[col].transform(X_outer_test[:, i:i+1]).flatten()

        # 生成数据标准化
        gen_scaled = gen_data.copy()
        for col in INPUT_FEATURES:
            gen_scaled[col] = scalers[col].transform(gen_data[[col]])
        X_gen = gen_scaled[INPUT_FEATURES].values
        y_gen = gen_scaled[TARGET_COL].values

        # 交互特征
        X_outer_train_full = np.hstack([X_outer_train_scaled, compute_interaction_features(X_outer_train_scaled)])
        X_outer_test_full = np.hstack([X_outer_test_scaled, compute_interaction_features(X_outer_test_scaled)])
        X_gen_full = np.hstack([X_gen, compute_interaction_features(X_gen)])

        # 内层CV：获取外层训练集的OOF预测
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model_names = ['deepfm', 'xgb', 'lgb', 'cat', 'gbm', 'rf']
        inner_oof = {name: np.zeros(len(y_outer_train)) for name in model_names}

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(X_outer_train_full), 1):
            X_inner_train = X_outer_train_full[inner_train_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            X_inner_val = X_outer_train_full[inner_val_idx]
            y_inner_val = y_outer_train[inner_val_idx]

            n_repeat = 30
            X_aug = np.vstack([X_gen_full] + [X_inner_train] * n_repeat)
            y_aug = np.concatenate([y_gen] + [y_inner_train] * n_repeat)

            # DeepFM（2个取平均）
            dfm_preds = []
            for m in range(2):
                model, _ = train_deepfm_quick(X_aug, y_aug, X_inner_val, y_inner_val,
                                              seed=42 + outer_fold * 1000 + inner_fold * 100 + m,
                                              device=device, epochs=150, patience=25)
                model.eval()
                with torch.no_grad():
                    p = model(torch.FloatTensor(X_inner_val).to(device)).cpu().numpy().flatten()
                dfm_preds.append(p)
            inner_oof['deepfm'][inner_val_idx] = np.mean(dfm_preds, axis=0)

            # XGBoost
            xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, random_state=42, verbosity=0, n_jobs=-1)
            xgb_m.fit(X_aug, y_aug)
            inner_oof['xgb'][inner_val_idx] = xgb_m.predict(X_inner_val)

            # LightGBM
            lgb_m = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                        subsample=0.8, random_state=42, verbose=-1, n_jobs=-1)
            lgb_m.fit(X_aug, y_aug)
            inner_oof['lgb'][inner_val_idx] = lgb_m.predict(X_inner_val)

            # CatBoost
            cat_m = cb.CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05,
                                          random_seed=42, verbose=0)
            cat_m.fit(X_aug, y_aug)
            inner_oof['cat'][inner_val_idx] = cat_m.predict(X_inner_val)

            # GBM
            gbm_m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                               subsample=0.8, random_state=42)
            gbm_m.fit(X_aug, y_aug)
            inner_oof['gbm'][inner_val_idx] = gbm_m.predict(X_inner_val)

            # RF
            rf_m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                          random_state=42, n_jobs=-1)
            rf_m.fit(X_aug, y_aug)
            inner_oof['rf'][inner_val_idx] = rf_m.predict(X_inner_val)

        # 训练元学习器（在外层训练集的OOF预测上）
        inner_oof_matrix = np.column_stack([inner_oof[n] for n in model_names])
        meta = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
        meta.fit(inner_oof_matrix, y_outer_train)

        # 在外层测试集上预测（需要先获取外层测试集的基模型预测）
        # 用全部外层训练集训练基模型，在外层测试集上预测
        n_repeat = 30
        X_aug_full = np.vstack([X_gen_full] + [X_outer_train_full] * n_repeat)
        y_aug_full = np.concatenate([y_gen] + [y_outer_train] * n_repeat)

        test_base_preds = {}

        # DeepFM
        dfm_preds = []
        for m in range(3):
            model, _ = train_deepfm_quick(X_aug_full, y_aug_full, X_outer_test_full, y_outer_test,
                                          seed=42 + outer_fold * 10000 + m,
                                          device=device, epochs=200, patience=30)
            model.eval()
            with torch.no_grad():
                p = model(torch.FloatTensor(X_outer_test_full).to(device)).cpu().numpy().flatten()
            dfm_preds.append(p)
        test_base_preds['deepfm'] = np.mean(dfm_preds, axis=0)

        xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, random_state=42, verbosity=0, n_jobs=-1)
        xgb_m.fit(X_aug_full, y_aug_full)
        test_base_preds['xgb'] = xgb_m.predict(X_outer_test_full)

        lgb_m = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, random_state=42, verbose=-1, n_jobs=-1)
        lgb_m.fit(X_aug_full, y_aug_full)
        test_base_preds['lgb'] = lgb_m.predict(X_outer_test_full)

        cat_m = cb.CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05,
                                      random_seed=42, verbose=0)
        cat_m.fit(X_aug_full, y_aug_full)
        test_base_preds['cat'] = cat_m.predict(X_outer_test_full)

        gbm_m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, random_state=42)
        gbm_m.fit(X_aug_full, y_aug_full)
        test_base_preds['gbm'] = gbm_m.predict(X_outer_test_full)

        rf_m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1)
        rf_m.fit(X_aug_full, y_aug_full)
        test_base_preds['rf'] = rf_m.predict(X_outer_test_full)

        # 元学习器预测
        test_base_matrix = np.column_stack([test_base_preds[n] for n in model_names])
        fold_preds = meta.predict(test_base_matrix)
        nested_preds[outer_test_idx] = fold_preds

        fold_r2 = r2_score(y_outer_test, fold_preds)
        nested_fold_r2.append(fold_r2)
        print(f"    外层第{outer_fold}折 嵌套CV R²: {fold_r2:.4f}")

    # 汇总
    nested_overall_r2 = r2_score(y_real, nested_preds)
    nested_rmse = np.sqrt(mean_squared_error(y_real, nested_preds))
    nested_mae = mean_absolute_error(y_real, nested_preds)
    nested_avg_r2 = np.mean(nested_fold_r2)
    nested_std_r2 = np.std(nested_fold_r2)

    print(f"\n  嵌套CV结果:")
    print(f"    整体R²: {nested_overall_r2:.4f}")
    print(f"    RMSE:   {nested_rmse:.4f}")
    print(f"    MAE:    {nested_mae:.4f}")
    print(f"    平均折R²: {nested_avg_r2:.4f} ± {nested_std_r2:.4f}")

    # ============================================================
    # 对比汇总
    # ============================================================
    print("\n" + "=" * 80)
    print("过拟合检验结论")
    print("=" * 80)

    v7_stacking_r2 = 0.7842
    overfit_gap = v7_stacking_r2 - nested_overall_r2

    print(f"\n  {'方法':<45} {'R²':>8} {'RMSE':>8}")
    print(f"  {'-'*65}")
    print(f"  {'V7 Stacking（原始，元学习器在全量数据上评估）':<45} {v7_stacking_r2:8.4f} {'1.8938':>8}")
    print(f"  {'嵌套CV Stacking（正确评估）':<45} {nested_overall_r2:8.4f} {nested_rmse:8.4f}")
    print(f"  {'V7 加权集成（无Stacking过拟合）':<45} {'0.7281':>8} {'2.1255':>8}")
    print(f"\n  过拟合gap（原始 - 嵌套CV）: {overfit_gap:.4f}")

    if overfit_gap > 0.05:
        print(f"  ⚠️  存在显著过拟合！原始Stacking R²={v7_stacking_r2:.4f}被高估了{overfit_gap:.4f}")
        print(f"  ✓  真实泛化能力（嵌套CV）: R²={nested_overall_r2:.4f}")
    elif overfit_gap > 0.02:
        print(f"  ⚠️  存在轻微过拟合（gap={overfit_gap:.4f}），嵌套CV R²={nested_overall_r2:.4f}更可信")
    else:
        print(f"  ✓  过拟合不显著（gap={overfit_gap:.4f}），R²={nested_overall_r2:.4f}可信")

    # 保存结果
    out = Path(__file__).parent / "results" / "overfitting_check_v7"
    out.mkdir(parents=True, exist_ok=True)

    results = {
        'v7_stacking_original': v7_stacking_r2,
        'nested_cv_r2': float(nested_overall_r2),
        'nested_cv_rmse': float(nested_rmse),
        'nested_cv_mae': float(nested_mae),
        'nested_avg_fold_r2': float(nested_avg_r2),
        'nested_std_fold_r2': float(nested_std_r2),
        'nested_fold_r2': [float(r) for r in nested_fold_r2],
        'overfit_gap': float(overfit_gap),
        'v7_weighted_r2': 0.7281,
    }

    with open(out / "overfitting_check.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  结果保存到: {out}")
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
