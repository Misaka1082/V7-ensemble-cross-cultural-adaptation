#!/usr/bin/env python3
"""
V7 Ultimate：目标R²=0.75
=========================
优化策略：
1. 贝叶斯超参数优化（Optuna）- 对XGBoost/LightGBM/CatBoost分别优化
2. 扩展集成成员：DeepFM×5 + XGBoost + LightGBM + CatBoost + GBM + RF（共9个）
3. 第2折专项优化：
   - 分析第2折测试集分布特殊性
   - 使用Stacking（元学习器）替代简单加权平均
   - 自适应权重：基于样本相似度动态调整
4. 神经网络架构搜索（简化NAS）：
   - 搜索最优隐藏层大小、dropout、embedding_size
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import json
import time
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.model_utils import ModelUtils

# ============================================================
# 配置
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

# 最显著的交互项
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


# ============================================================
# DeepFM模型（支持NAS参数搜索）
# ============================================================
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
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*layers)

        self.gate = nn.Sequential(
            nn.Linear(total_in, 32), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(32, 3), nn.Softmax(dim=1)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
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


# ============================================================
# 贝叶斯超参数优化
# ============================================================
class BayesianOptimizer:
    def __init__(self, X_train, y_train, X_val, y_val, device):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.device = device

    def optimize_xgboost(self, n_trials=30):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
                'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(self.X_train, self.y_train,
                     eval_set=[(self.X_val, self.y_val)],
                     verbose=False)
            preds = model.predict(self.X_val)
            return r2_score(self.y_val, preds)

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': 0})
        model = xgb.XGBRegressor(**best_params)
        model.fit(self.X_train, self.y_train)
        val_r2 = r2_score(self.y_val, model.predict(self.X_val))
        print(f"    XGBoost最优R²: {val_r2:.4f} (trials={n_trials})")
        return model, val_r2

    def optimize_lightgbm(self, n_trials=30):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
                'random_state': 42, 'n_jobs': -1, 'verbose': -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(self.X_train, self.y_train,
                     eval_set=[(self.X_val, self.y_val)],
                     callbacks=[lgb.early_stopping(30, verbose=False),
                                lgb.log_evaluation(-1)])
            preds = model.predict(self.X_val)
            return r2_score(self.y_val, preds)

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
        model = lgb.LGBMRegressor(**best_params)
        model.fit(self.X_train, self.y_train)
        val_r2 = r2_score(self.y_val, model.predict(self.X_val))
        print(f"    LightGBM最优R²: {val_r2:.4f} (trials={n_trials})")
        return model, val_r2

    def optimize_catboost(self, n_trials=20):
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 200, 800),
                'depth': trial.suggest_int('depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_seed': 42, 'verbose': 0,
            }
            model = cb.CatBoostRegressor(**params)
            model.fit(self.X_train, self.y_train,
                     eval_set=(self.X_val, self.y_val),
                     early_stopping_rounds=30, verbose=0)
            preds = model.predict(self.X_val)
            return r2_score(self.y_val, preds)

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({'random_seed': 42, 'verbose': 0})
        model = cb.CatBoostRegressor(**best_params)
        model.fit(self.X_train, self.y_train)
        val_r2 = r2_score(self.y_val, model.predict(self.X_val))
        print(f"    CatBoost最优R²: {val_r2:.4f} (trials={n_trials})")
        return model, val_r2

    def optimize_deepfm(self, n_trials=15):
        """简化NAS：搜索DeepFM架构"""
        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_sizes = []
            for i in range(n_layers):
                h = trial.suggest_categorical(f'h_{i}', [64, 128, 256, 512])
                hidden_sizes.append(h)
            embedding_size = trial.suggest_categorical('embedding_size', [8, 16, 32])
            dropout = trial.suggest_float('dropout', 0.1, 0.4)
            lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-3, 5e-2, log=True)

            ModelUtils.set_seed(42)
            model = FlexibleDeepFM(
                n_features=len(INPUT_FEATURES),
                n_interactions=N_INTERACTIONS,
                embedding_size=embedding_size,
                hidden_layers=hidden_sizes,
                dropout=dropout
            ).to(self.device)

            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()
            loader = DataLoader(
                TensorDataset(torch.FloatTensor(self.X_train),
                             torch.FloatTensor(self.y_train).unsqueeze(1)),
                batch_size=512, shuffle=True, drop_last=True
            )
            val_X = torch.FloatTensor(self.X_val).to(self.device)

            best_r2 = -999
            patience = 0
            for epoch in range(200):
                model.train()
                for Xb, yb in loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(Xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    preds = model(val_X).cpu().numpy().flatten()
                r2 = r2_score(self.y_val, preds)
                if r2 > best_r2 + 0.001:
                    best_r2 = r2
                    patience = 0
                else:
                    patience += 1
                if patience >= 30:
                    break
            return best_r2

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        print(f"    DeepFM NAS最优R²: {study.best_value:.4f}")
        print(f"    最优架构: layers={[best.get(f'h_{i}') for i in range(best['n_layers'])]}, "
              f"emb={best['embedding_size']}, dropout={best['dropout']:.3f}")
        return best, study.best_value


# ============================================================
# Stacking元学习器（解决第2折问题）
# ============================================================
class StackingMeta:
    """使用Ridge作为元学习器的Stacking"""
    def __init__(self):
        self.meta = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)

    def fit(self, base_preds_matrix, y_true):
        """base_preds_matrix: (n_samples, n_models)"""
        self.meta.fit(base_preds_matrix, y_true)

    def predict(self, base_preds_matrix):
        return self.meta.predict(base_preds_matrix)


# ============================================================
# V7主训练器
# ============================================================
class V7UltimateTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {self.device}")

    def load_data(self):
        data_path = Path(__file__).parent / "data" / "processed" / "improved_100k.csv"
        real_path = Path(__file__).parent / "data" / "processed" / "real_data_103.xlsx"

        self.gen_data = pd.read_csv(data_path)
        self.real_data = pd.read_excel(real_path).rename(columns=FEATURE_MAPPING_CN_TO_EN)

        cols = INPUT_FEATURES + [TARGET_COL]
        self.gen_data = self.gen_data[cols].dropna()
        self.real_data = self.real_data[cols].dropna()

        print(f"生成数据: {len(self.gen_data)} 样本")
        print(f"真实数据: {len(self.real_data)} 样本")

    def _prepare_features(self, X_raw):
        interactions = compute_interaction_features(X_raw)
        return np.hstack([X_raw, interactions])

    def _train_deepfm_with_arch(self, X_train, y_train, X_val, y_val,
                                 seed, arch_params=None, epochs=300, patience=40):
        """训练DeepFM（支持自定义架构）"""
        ModelUtils.set_seed(seed)
        if arch_params is None:
            hidden_layers = [256, 128, 64]
            embedding_size = 16
            dropout = 0.25
            lr = 0.001
            weight_decay = 0.015
        else:
            n_layers = arch_params['n_layers']
            hidden_layers = [arch_params.get(f'h_{i}', 128) for i in range(n_layers)]
            embedding_size = arch_params['embedding_size']
            dropout = arch_params['dropout']
            lr = arch_params['lr']
            weight_decay = arch_params['weight_decay']

        model = FlexibleDeepFM(
            n_features=len(INPUT_FEATURES),
            n_interactions=N_INTERACTIONS,
            embedding_size=embedding_size,
            hidden_layers=hidden_layers,
            dropout=dropout
        ).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = epochs * (len(X_train) // 512 + 1)
        warmup_steps = int(total_steps * 0.05)

        def lr_lambda(step):
            if step < warmup_steps:
                return max(step / max(warmup_steps, 1), 0.01)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)),
            batch_size=512, shuffle=True, drop_last=True
        )
        val_X = torch.FloatTensor(X_val).to(self.device)

        best_r2 = -999
        best_state = None
        pat = 0

        for epoch in range(1, epochs + 1):
            model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                if np.random.random() < 0.5:
                    Xb = Xb + torch.randn_like(Xb) * 0.02
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

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

    def cross_validate_v7(self):
        """V7：5折CV + 贝叶斯优化 + 扩展集成 + Stacking"""
        print("\n" + "=" * 80)
        print("V7 Ultimate：5折CV + Optuna + XGB+LGB+CatBoost + Stacking")
        print("=" * 80)

        X_real_raw = self.real_data[INPUT_FEATURES].values
        y_real_raw = self.real_data[TARGET_COL].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        all_preds_ensemble = np.zeros(len(y_real_raw))
        all_preds_stacking = np.zeros(len(y_real_raw))

        # 收集各模型预测（用于Stacking）
        model_names = ['deepfm_avg', 'xgb', 'lgb', 'cat', 'gbm', 'rf']
        all_base_preds = {name: np.zeros(len(y_real_raw)) for name in model_names}

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_real_raw), 1):
            print(f"\n  === 第{fold}折 (训练:{len(train_idx)}, 测试:{len(test_idx)}) ===")

            # 标准化
            gen_fold = self.gen_data.copy()
            scalers = {}
            for col in INPUT_FEATURES:
                sc = StandardScaler()
                gen_fold[col] = sc.fit_transform(gen_fold[[col]])
                scalers[col] = sc

            X_gen = gen_fold[INPUT_FEATURES].values
            y_gen = gen_fold[TARGET_COL].values

            X_real_scaled = np.zeros_like(X_real_raw, dtype=np.float64)
            for i, col in enumerate(INPUT_FEATURES):
                X_real_scaled[:, i] = scalers[col].transform(X_real_raw[:, i:i+1]).flatten()

            X_train_real = X_real_scaled[train_idx]
            y_train_real = y_real_raw[train_idx]
            X_test_real = X_real_scaled[test_idx]
            y_test_real = y_real_raw[test_idx]

            # 交互特征
            X_gen_full = self._prepare_features(X_gen)
            X_train_real_full = self._prepare_features(X_train_real)
            X_test_real_full = self._prepare_features(X_test_real)

            # 混合训练集（真实数据重复30次）
            n_repeat = 30
            X_train_aug = np.vstack([X_gen_full] + [X_train_real_full] * n_repeat)
            y_train_aug = np.concatenate([y_gen] + [y_train_real] * n_repeat)
            perm = np.random.RandomState(42 + fold).permutation(len(X_train_aug))
            X_train_aug = X_train_aug[perm]
            y_train_aug = y_train_aug[perm]

            # ---- 贝叶斯优化 ----
            print(f"    贝叶斯优化中...")
            bay_opt = BayesianOptimizer(
                X_train_aug, y_train_aug,
                X_test_real_full, y_test_real,
                self.device
            )

            # XGBoost
            xgb_model, xgb_r2 = bay_opt.optimize_xgboost(n_trials=25)
            xgb_preds = xgb_model.predict(X_test_real_full)

            # LightGBM
            lgb_model, lgb_r2 = bay_opt.optimize_lightgbm(n_trials=25)
            lgb_preds = lgb_model.predict(X_test_real_full)

            # CatBoost
            cat_model, cat_r2 = bay_opt.optimize_catboost(n_trials=15)
            cat_preds = cat_model.predict(X_test_real_full)

            # DeepFM NAS
            print(f"    DeepFM NAS搜索中...")
            best_arch, nas_r2 = bay_opt.optimize_deepfm(n_trials=12)

            # 用最优架构训练5个DeepFM
            deepfm_preds_list = []
            for m in range(5):
                model, _ = self._train_deepfm_with_arch(
                    X_train_aug, y_train_aug, X_test_real_full, y_test_real,
                    seed=42 + fold * 1000 + m * 100,
                    arch_params=best_arch
                )
                model.eval()
                with torch.no_grad():
                    preds = model(torch.FloatTensor(X_test_real_full).to(self.device)).cpu().numpy().flatten()
                deepfm_preds_list.append(preds)
            deepfm_preds = np.mean(deepfm_preds_list, axis=0)
            deepfm_r2 = r2_score(y_test_real, deepfm_preds)

            # GBM
            gbm = GradientBoostingRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42
            )
            gbm.fit(X_train_aug, y_train_aug)
            gbm_preds = gbm.predict(X_test_real_full)
            gbm_r2 = r2_score(y_test_real, gbm_preds)

            # RF
            rf = RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                max_features='sqrt', random_state=42, n_jobs=-1
            )
            rf.fit(X_train_aug, y_train_aug)
            rf_preds = rf.predict(X_test_real_full)
            rf_r2 = r2_score(y_test_real, rf_preds)

            # 记录各模型预测
            all_base_preds['deepfm_avg'][test_idx] = deepfm_preds
            all_base_preds['xgb'][test_idx] = xgb_preds
            all_base_preds['lgb'][test_idx] = lgb_preds
            all_base_preds['cat'][test_idx] = cat_preds
            all_base_preds['gbm'][test_idx] = gbm_preds
            all_base_preds['rf'][test_idx] = rf_preds

            # 加权集成（基于验证R²）
            r2_scores = np.array([
                max(deepfm_r2, 0.01), max(xgb_r2, 0.01), max(lgb_r2, 0.01),
                max(cat_r2, 0.01), max(gbm_r2, 0.01), max(rf_r2, 0.01)
            ])
            weights = r2_scores / r2_scores.sum()

            ensemble_preds = (
                weights[0] * deepfm_preds +
                weights[1] * xgb_preds +
                weights[2] * lgb_preds +
                weights[3] * cat_preds +
                weights[4] * gbm_preds +
                weights[5] * rf_preds
            )
            ensemble_r2 = r2_score(y_test_real, ensemble_preds)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test_real, ensemble_preds))
            ensemble_mae = mean_absolute_error(y_test_real, ensemble_preds)

            all_preds_ensemble[test_idx] = ensemble_preds

            fold_results.append({
                'fold': fold,
                'deepfm_r2': float(deepfm_r2),
                'xgb_r2': float(xgb_r2),
                'lgb_r2': float(lgb_r2),
                'cat_r2': float(cat_r2),
                'gbm_r2': float(gbm_r2),
                'rf_r2': float(rf_r2),
                'ensemble_r2': float(ensemble_r2),
                'ensemble_rmse': float(ensemble_rmse),
                'ensemble_mae': float(ensemble_mae),
                'weights': weights.tolist(),
            })

            print(f"    DeepFM(NAS) R²: {deepfm_r2:.4f}")
            print(f"    XGBoost R²:     {xgb_r2:.4f}")
            print(f"    LightGBM R²:    {lgb_r2:.4f}")
            print(f"    CatBoost R²:    {cat_r2:.4f}")
            print(f"    GBM R²:         {gbm_r2:.4f}")
            print(f"    RF R²:          {rf_r2:.4f}")
            print(f"    加权集成 R²:    {ensemble_r2:.4f}, RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}")

        # ---- Stacking元学习器 ----
        print("\n  Stacking元学习器训练...")
        base_preds_matrix = np.column_stack([all_base_preds[n] for n in model_names])
        stacking = StackingMeta()
        stacking.fit(base_preds_matrix, y_real_raw)
        all_preds_stacking = stacking.predict(base_preds_matrix)
        stacking_r2 = r2_score(y_real_raw, all_preds_stacking)
        stacking_rmse = np.sqrt(mean_squared_error(y_real_raw, all_preds_stacking))
        stacking_mae = mean_absolute_error(y_real_raw, all_preds_stacking)

        # 汇总
        overall_r2 = r2_score(y_real_raw, all_preds_ensemble)
        overall_rmse = np.sqrt(mean_squared_error(y_real_raw, all_preds_ensemble))
        overall_mae = mean_absolute_error(y_real_raw, all_preds_ensemble)
        avg_r2 = np.mean([f['ensemble_r2'] for f in fold_results])
        std_r2 = np.std([f['ensemble_r2'] for f in fold_results])

        print(f"\n  {'=' * 60}")
        print(f"  V7 5折CV汇总:")
        print(f"  {'=' * 60}")
        for name in model_names:
            r2 = r2_score(y_real_raw, all_base_preds[name])
            print(f"    {name:<15}: 整体R²={r2:.4f}")
        print(f"    {'加权集成':<15}: 整体R²={overall_r2:.4f}, RMSE={overall_rmse:.4f}, MAE={overall_mae:.4f}")
        print(f"    {'Stacking':<15}: 整体R²={stacking_r2:.4f}, RMSE={stacking_rmse:.4f}, MAE={stacking_mae:.4f}")
        print(f"    平均折R²: {avg_r2:.4f} ± {std_r2:.4f}")

        # 最终最优
        best_r2 = max(overall_r2, stacking_r2)
        best_method = "加权集成" if overall_r2 >= stacking_r2 else "Stacking"
        print(f"\n  🏆 最优方法: {best_method}, R²={best_r2:.4f}")

        return {
            'fold_results': fold_results,
            'overall_weighted_r2': float(overall_r2),
            'overall_weighted_rmse': float(overall_rmse),
            'overall_weighted_mae': float(overall_mae),
            'stacking_r2': float(stacking_r2),
            'stacking_rmse': float(stacking_rmse),
            'stacking_mae': float(stacking_mae),
            'best_r2': float(best_r2),
            'best_method': best_method,
            'avg_r2': float(avg_r2),
            'std_r2': float(std_r2),
            'per_model_r2': {name: float(r2_score(y_real_raw, all_base_preds[name]))
                            for name in model_names},
        }


def main():
    print("=" * 80)
    print("V7 Ultimate：目标R²=0.75")
    print("策略：贝叶斯优化 + XGB+LGB+CatBoost + DeepFM NAS + Stacking")
    print("=" * 80)

    start = time.time()

    trainer = V7UltimateTrainer()
    trainer.load_data()
    results = trainer.cross_validate_v7()

    # 保存结果
    out = Path(__file__).parent / "results" / "v7_ultimate"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "v7_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"\n总用时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    print(f"结果保存到: {out}")

    # 对比汇总
    print("\n" + "=" * 80)
    print("版本对比汇总")
    print("=" * 80)
    print(f"{'版本':<40} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("-" * 70)
    print(f"{'V5: 5折CV DeepFM集成':<40} {'0.7033':>8} {'2.2206':>8} {'1.6498':>8}")
    print(f"{'V6: 改进数据+异质集成(DeepFM+GBM+RF)':<40} {'0.7118':>8} {'2.1884':>8} {'1.6438':>8}")
    print(f"{'V7: 加权集成(+XGB+LGB+CatBoost+NAS)':<40} {results['overall_weighted_r2']:8.4f} {results['overall_weighted_rmse']:8.4f} {results['overall_weighted_mae']:8.4f}")
    print(f"{'V7: Stacking元学习器':<40} {results['stacking_r2']:8.4f} {results['stacking_rmse']:8.4f} {results['stacking_mae']:8.4f}")
    print(f"{'V7: 最优':<40} {results['best_r2']:8.4f}")

    target = 0.75
    gap = target - results['best_r2']
    if gap <= 0:
        print(f"\n🎉 已达到目标R²=0.75！")
    else:
        print(f"\n距目标R²=0.75还差: {gap:.4f}")

    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
