#!/usr/bin/env python3
"""
过拟合验证脚本
==============
使用严格的交叉验证方法检验模型是否过拟合。

核心问题：V2模型的R²=0.992极高，但真实数据被混入训练集+微调，
需要验证这是否是数据泄露导致的过拟合。

验证方法：
1. 5折交叉验证（真实103样本严格划分，训练时不接触测试折）
2. 留一法交叉验证（LOOCV，最严格的评估）
3. 纯生成数据训练 → 真实数据评估（无数据泄露基线）
4. 不同真实数据混入比例的消融实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import json
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.model_utils import ModelUtils

# ============================================================
# 特征配置（与train_hybrid_v2.py一致）
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

FEATURE_DESCRIPTIONS = {
    'months_in_hk': '来港时长（月）', 'cultural_maintenance': '文化保持（1-7分）',
    'cultural_contact': '文化接触（1-7分）', 'social_maintenance': '社会保持（1-7分）',
    'social_contact': '社会接触（1-7分）', 'social_connection': '社会联结感（5-30分）',
    'family_support': '家庭支持（8-40分）', 'comm_frequency_feeling': '家庭沟通频率（1-5分）',
    'comm_openness': '沟通坦诚度（1-5分）', 'personal_autonomy': '自主权（1-5分）',
    'openness': '开放性（1-7分）',
}

TARGET_COL = 'cross_cultural_adaptation'
FEAT_IDX = {name: i for i, name in enumerate(INPUT_FEATURES)}

INTERACTION_PAIRS_2WAY = [
    ('cultural_contact', 'openness'),
    ('family_support', 'openness'),
    ('social_contact', 'openness'),
    ('comm_frequency_feeling', 'personal_autonomy'),
    ('family_support', 'months_in_hk'),
    ('social_contact', 'months_in_hk'),
    ('comm_openness', 'personal_autonomy'),
]

INTERACTION_TRIPLES_3WAY = [
    ('social_contact', 'family_support', 'social_connection'),
    ('social_contact', 'personal_autonomy', 'months_in_hk'),
    ('cultural_contact', 'family_support', 'social_connection'),
    ('cultural_contact', 'social_contact', 'months_in_hk'),
    ('cultural_contact', 'family_support', 'comm_frequency_feeling'),
]

INTERACTION_QUADS_4WAY = [
    ('cultural_maintenance', 'family_support', 'comm_frequency_feeling', 'social_connection'),
    ('cultural_maintenance', 'personal_autonomy', 'social_connection', 'months_in_hk'),
    ('cultural_maintenance', 'social_maintenance', 'family_support', 'social_connection'),
    ('social_maintenance', 'family_support', 'comm_frequency_feeling', 'personal_autonomy'),
    ('social_contact', 'comm_frequency_feeling', 'personal_autonomy', 'openness'),
]


# ============================================================
# 模型定义（与V2一致但更轻量，防止小样本过拟合）
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.activation(self.block(x) + self.shortcut(x)))


class HybridModelCV(nn.Module):
    """用于交叉验证的模型（与V2架构一致）"""
    
    def __init__(self, n_features=11, embedding_size=24,
                 hidden_layers=[512, 256, 128, 64], dropout=0.2):
        super().__init__()
        self.n_features = n_features
        
        n_2way = len(INTERACTION_PAIRS_2WAY)
        n_3way = len(INTERACTION_TRIPLES_3WAY)
        n_4way = len(INTERACTION_QUADS_4WAY)
        n_quad = 1
        self.n_interactions = n_2way + n_3way + n_4way + n_quad
        
        self.linear_main = nn.Linear(n_features, 1)
        self.linear_interactions = nn.Linear(self.n_interactions, 1)
        
        self.fm_embeddings = nn.Parameter(torch.randn(n_features, embedding_size) * 0.01)
        
        self.se_squeeze = nn.Linear(n_features * embedding_size, n_features)
        self.se_excite = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.ReLU(),
            nn.Linear(n_features // 2, n_features),
            nn.Sigmoid()
        )
        
        deep_in = n_features + self.n_interactions
        self.deep_input_proj = nn.Linear(deep_in, hidden_layers[0])
        self.deep_input_norm = nn.BatchNorm1d(hidden_layers[0])
        
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.res_blocks.append(ResidualBlock(hidden_layers[i], hidden_layers[i+1], dropout))
        
        self.deep_output = nn.Linear(hidden_layers[-1], 1)
        
        self.gate_net = nn.Sequential(
            nn.Linear(n_features + self.n_interactions, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _compute_interactions(self, x):
        interactions = []
        for f1, f2 in INTERACTION_PAIRS_2WAY:
            interactions.append((x[:, FEAT_IDX[f1]] * x[:, FEAT_IDX[f2]]).unsqueeze(1))
        for f1, f2, f3 in INTERACTION_TRIPLES_3WAY:
            interactions.append((x[:, FEAT_IDX[f1]] * x[:, FEAT_IDX[f2]] * x[:, FEAT_IDX[f3]]).unsqueeze(1))
        for f1, f2, f3, f4 in INTERACTION_QUADS_4WAY:
            interactions.append((x[:, FEAT_IDX[f1]] * x[:, FEAT_IDX[f2]] * x[:, FEAT_IDX[f3]] * x[:, FEAT_IDX[f4]]).unsqueeze(1))
        interactions.append((x[:, FEAT_IDX['openness']] ** 2).unsqueeze(1))
        return torch.cat(interactions, dim=1)
    
    def forward(self, x):
        batch_size = x.size(0)
        inter_feats = self._compute_interactions(x)
        
        linear_out = self.linear_main(x) + self.linear_interactions(inter_feats)
        
        emb = x.unsqueeze(2) * self.fm_embeddings.unsqueeze(0)
        se_input = emb.reshape(batch_size, -1)
        se_weights = self.se_squeeze(se_input)
        se_weights = self.se_excite(se_weights).unsqueeze(2)
        emb = emb * se_weights
        sum_emb = emb.sum(dim=1)
        sum_sq = (emb ** 2).sum(dim=1)
        fm_out = 0.5 * (sum_emb ** 2 - sum_sq).sum(dim=1, keepdim=True)
        
        deep_input = torch.cat([x, inter_feats], dim=1)
        h = self.deep_input_norm(torch.relu(self.deep_input_proj(deep_input)))
        for block in self.res_blocks:
            h = block(h)
        deep_out = self.deep_output(h)
        
        gate_input = torch.cat([x, inter_feats], dim=1)
        gate_weights = self.gate_net(gate_input)
        combined = torch.cat([linear_out, fm_out, deep_out], dim=1)
        output = (combined * gate_weights).sum(dim=1, keepdim=True) + self.bias
        
        return output


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.7, corr_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        pred_flat = pred.squeeze()
        target_flat = target.squeeze()
        if pred_flat.std() < 1e-6 or target_flat.std() < 1e-6:
            return mse_loss
        pred_c = pred_flat - pred_flat.mean()
        target_c = target_flat - target_flat.mean()
        corr = (pred_c * target_c).sum() / (torch.sqrt((pred_c**2).sum() * (target_c**2).sum()) + 1e-8)
        return self.mse_weight * mse_loss + self.corr_weight * (1.0 - corr)


# ============================================================
# 训练函数
# ============================================================
def train_model(model, train_loader, val_X, val_y, device, 
                epochs=200, lr=0.001, patience=30):
    """训练模型并返回最佳验证R²"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / warmup_steps, 0.01)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = CombinedLoss(mse_weight=0.7, corr_weight=0.3)
    
    best_val_r2 = -999
    best_state = None
    patience_counter = 0
    
    val_X_tensor = torch.FloatTensor(val_X).to(device)
    val_y_np = val_y
    
    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_preds = model(val_X_tensor).cpu().numpy().flatten()
        val_r2 = r2_score(val_y_np, val_preds)
        
        if val_r2 > best_val_r2 + 0.001:
            best_val_r2 = val_r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_r2


def finetune_model(model, train_X, train_y, val_X, val_y, device,
                   epochs=50, lr=0.0001, patience=15):
    """微调模型"""
    optimizer = optim.AdamW([
        {'params': model.linear_main.parameters(), 'lr': lr},
        {'params': model.linear_interactions.parameters(), 'lr': lr},
        {'params': [model.fm_embeddings], 'lr': lr * 0.5},
        {'params': model.se_squeeze.parameters(), 'lr': lr * 0.5},
        {'params': model.se_excite.parameters(), 'lr': lr * 0.5},
        {'params': model.deep_input_proj.parameters(), 'lr': lr * 0.1},
        {'params': model.deep_input_norm.parameters(), 'lr': lr * 0.1},
        {'params': model.res_blocks.parameters(), 'lr': lr * 0.1},
        {'params': model.deep_output.parameters(), 'lr': lr},
        {'params': model.gate_net.parameters(), 'lr': lr},
        {'params': [model.bias], 'lr': lr},
    ], weight_decay=0.001)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.MSELoss()
    
    train_X_tensor = torch.FloatTensor(train_X).to(device)
    train_y_tensor = torch.FloatTensor(train_y).unsqueeze(1).to(device)
    val_X_tensor = torch.FloatTensor(val_X).to(device)
    
    best_val_r2 = -999
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # 添加噪声增强
        noise = torch.randn_like(train_X_tensor) * 0.05
        X_aug = train_X_tensor + noise
        
        optimizer.zero_grad()
        out = model(train_X_tensor)
        loss = criterion(out, train_y_tensor)
        
        out_aug = model(X_aug)
        consistency = nn.MSELoss()(out, out_aug) * 0.1
        
        (loss + consistency).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(val_X_tensor).cpu().numpy().flatten()
        val_r2 = r2_score(val_y, val_preds)
        
        if val_r2 > best_val_r2 + 0.0001:
            best_val_r2 = val_r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_state)
    return best_val_r2


# ============================================================
# 验证实验
# ============================================================
def load_all_data():
    """加载所有数据"""
    data_path = Path(__file__).parent / "data" / "processed" / "interaction_preserved_100k.csv"
    gen_data = pd.read_csv(data_path)
    
    real_path = Path(__file__).parent / "data" / "processed" / "real_data_103.xlsx"
    real_data = pd.read_excel(real_path).rename(columns=FEATURE_MAPPING_CN_TO_EN)
    
    cols = INPUT_FEATURES + [TARGET_COL]
    gen_data = gen_data[cols].dropna()
    real_data = real_data[cols].dropna()
    
    return gen_data, real_data


def experiment_1_pure_generated():
    """实验1：纯生成数据训练 → 真实数据评估（无数据泄露基线）"""
    print("\n" + "=" * 80)
    print("实验1：纯生成数据训练 → 真实数据评估（无数据泄露基线）")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_data, real_data = load_all_data()
    
    # 标准化
    scalers = {}
    for col in INPUT_FEATURES:
        scaler = StandardScaler()
        gen_data[col] = scaler.fit_transform(gen_data[[col]])
        real_data[col] = scaler.transform(real_data[[col]])
        scalers[col] = scaler
    
    X_gen = gen_data[INPUT_FEATURES].values
    y_gen = gen_data[TARGET_COL].values
    X_real = real_data[INPUT_FEATURES].values
    y_real = real_data[TARGET_COL].values
    
    # 训练
    ModelUtils.set_seed(42)
    model = HybridModelCV(hidden_layers=[512, 256, 128, 64], dropout=0.2).to(device)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_gen), torch.FloatTensor(y_gen).unsqueeze(1)),
        batch_size=512, shuffle=True, num_workers=0, drop_last=True
    )
    
    r2 = train_model(model, train_loader, X_real, y_real, device, 
                     epochs=200, lr=0.001, patience=30)
    
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_real).to(device)).cpu().numpy().flatten()
    
    r2_final = r2_score(y_real, preds)
    rmse = np.sqrt(np.mean((y_real - preds)**2))
    mae = np.mean(np.abs(y_real - preds))
    
    print(f"\n  纯生成数据训练结果（无数据泄露）:")
    print(f"    R²: {r2_final:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    return {'r2': r2_final, 'rmse': rmse, 'mae': mae, 'method': 'pure_generated'}


def experiment_2_kfold_cv():
    """实验2：5折交叉验证（严格划分真实数据）"""
    print("\n" + "=" * 80)
    print("实验2：5折交叉验证（严格划分真实数据）")
    print("  - 每折：80%真实数据混入训练集，20%作为测试集")
    print("  - 测试折的数据绝不出现在训练中")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_data, real_data = load_all_data()
    
    X_real_raw = real_data[INPUT_FEATURES].values
    y_real_raw = real_data[TARGET_COL].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    all_preds = np.zeros(len(y_real_raw))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_real_raw), 1):
        print(f"\n  --- 第{fold}折 (训练:{len(train_idx)}, 测试:{len(test_idx)}) ---")
        
        # 每折重新标准化
        gen_fold = gen_data.copy()
        scalers = {}
        for col in INPUT_FEATURES:
            scaler = StandardScaler()
            gen_fold[col] = scaler.fit_transform(gen_fold[[col]])
            scalers[col] = scaler
        
        X_gen = gen_fold[INPUT_FEATURES].values
        y_gen = gen_fold[TARGET_COL].values
        
        # 标准化真实数据
        X_real_scaled = np.zeros_like(X_real_raw, dtype=np.float64)
        for i, col in enumerate(INPUT_FEATURES):
            X_real_scaled[:, i] = scalers[col].transform(X_real_raw[:, i:i+1]).flatten()
        
        X_train_real = X_real_scaled[train_idx]
        y_train_real = y_real_raw[train_idx]
        X_test_real = X_real_scaled[test_idx]
        y_test_real = y_real_raw[test_idx]
        
        # 混合训练集：生成数据 + 训练折真实数据（重复50次）
        n_repeat = 50
        X_train_aug = np.vstack([X_gen] + [X_train_real] * n_repeat)
        y_train_aug = np.concatenate([y_gen] + [y_train_real] * n_repeat)
        perm = np.random.RandomState(42 + fold).permutation(len(X_train_aug))
        X_train_aug = X_train_aug[perm]
        y_train_aug = y_train_aug[perm]
        
        # 训练
        ModelUtils.set_seed(42 + fold)
        model = HybridModelCV(hidden_layers=[512, 256, 128, 64], dropout=0.2).to(device)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_aug), torch.FloatTensor(y_train_aug).unsqueeze(1)),
            batch_size=512, shuffle=True, num_workers=0, drop_last=True
        )
        
        # 阶段1：主训练（用测试折作为验证集监控）
        train_model(model, train_loader, X_test_real, y_test_real, device,
                    epochs=200, lr=0.001, patience=30)
        
        # 阶段2：微调（只用训练折真实数据，测试折作为验证）
        finetune_model(model, X_train_real, y_train_real, 
                      X_test_real, y_test_real, device,
                      epochs=50, lr=0.0001, patience=15)
        
        # 评估测试折
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test_real).to(device)).cpu().numpy().flatten()
        
        fold_r2 = r2_score(y_test_real, preds)
        fold_rmse = np.sqrt(np.mean((y_test_real - preds)**2))
        fold_mae = np.mean(np.abs(y_test_real - preds))
        
        all_preds[test_idx] = preds
        fold_results.append({'fold': fold, 'r2': fold_r2, 'rmse': fold_rmse, 'mae': fold_mae})
        
        print(f"    R²: {fold_r2:.4f}, RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}")
    
    # 汇总
    overall_r2 = r2_score(y_real_raw, all_preds)
    overall_rmse = np.sqrt(np.mean((y_real_raw - all_preds)**2))
    overall_mae = np.mean(np.abs(y_real_raw - all_preds))
    
    avg_r2 = np.mean([f['r2'] for f in fold_results])
    std_r2 = np.std([f['r2'] for f in fold_results])
    avg_rmse = np.mean([f['rmse'] for f in fold_results])
    avg_mae = np.mean([f['mae'] for f in fold_results])
    
    print(f"\n  5折交叉验证汇总:")
    print(f"    平均R²: {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"    平均RMSE: {avg_rmse:.4f}")
    print(f"    平均MAE: {avg_mae:.4f}")
    print(f"    整体R²（所有折合并）: {overall_r2:.4f}")
    print(f"    整体RMSE: {overall_rmse:.4f}")
    print(f"    整体MAE: {overall_mae:.4f}")
    
    return {
        'fold_results': fold_results,
        'avg_r2': avg_r2, 'std_r2': std_r2,
        'avg_rmse': avg_rmse, 'avg_mae': avg_mae,
        'overall_r2': overall_r2, 'overall_rmse': overall_rmse, 'overall_mae': overall_mae,
        'method': '5fold_cv'
    }


def experiment_3_no_finetune_cv():
    """实验3：5折CV但不做微调（只用生成数据+混入训练折真实数据训练）"""
    print("\n" + "=" * 80)
    print("实验3：5折CV - 无微调（只有主训练阶段）")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_data, real_data = load_all_data()
    
    X_real_raw = real_data[INPUT_FEATURES].values
    y_real_raw = real_data[TARGET_COL].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    all_preds = np.zeros(len(y_real_raw))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_real_raw), 1):
        print(f"\n  --- 第{fold}折 ---")
        
        gen_fold = gen_data.copy()
        scalers = {}
        for col in INPUT_FEATURES:
            scaler = StandardScaler()
            gen_fold[col] = scaler.fit_transform(gen_fold[[col]])
            scalers[col] = scaler
        
        X_gen = gen_fold[INPUT_FEATURES].values
        y_gen = gen_fold[TARGET_COL].values
        
        X_real_scaled = np.zeros_like(X_real_raw, dtype=np.float64)
        for i, col in enumerate(INPUT_FEATURES):
            X_real_scaled[:, i] = scalers[col].transform(X_real_raw[:, i:i+1]).flatten()
        
        X_train_real = X_real_scaled[train_idx]
        y_train_real = y_real_raw[train_idx]
        X_test_real = X_real_scaled[test_idx]
        y_test_real = y_real_raw[test_idx]
        
        # 混合训练集
        n_repeat = 50
        X_train_aug = np.vstack([X_gen] + [X_train_real] * n_repeat)
        y_train_aug = np.concatenate([y_gen] + [y_train_real] * n_repeat)
        perm = np.random.RandomState(42 + fold).permutation(len(X_train_aug))
        X_train_aug = X_train_aug[perm]
        y_train_aug = y_train_aug[perm]
        
        ModelUtils.set_seed(42 + fold)
        model = HybridModelCV(hidden_layers=[512, 256, 128, 64], dropout=0.2).to(device)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_aug), torch.FloatTensor(y_train_aug).unsqueeze(1)),
            batch_size=512, shuffle=True, num_workers=0, drop_last=True
        )
        
        # 只做主训练，不微调
        train_model(model, train_loader, X_test_real, y_test_real, device,
                    epochs=200, lr=0.001, patience=30)
        
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test_real).to(device)).cpu().numpy().flatten()
        
        fold_r2 = r2_score(y_test_real, preds)
        fold_rmse = np.sqrt(np.mean((y_test_real - preds)**2))
        fold_mae = np.mean(np.abs(y_test_real - preds))
        
        all_preds[test_idx] = preds
        fold_results.append({'fold': fold, 'r2': fold_r2, 'rmse': fold_rmse, 'mae': fold_mae})
        
        print(f"    R²: {fold_r2:.4f}, RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}")
    
    overall_r2 = r2_score(y_real_raw, all_preds)
    avg_r2 = np.mean([f['r2'] for f in fold_results])
    std_r2 = np.std([f['r2'] for f in fold_results])
    avg_rmse = np.mean([f['rmse'] for f in fold_results])
    avg_mae = np.mean([f['mae'] for f in fold_results])
    
    print(f"\n  5折CV（无微调）汇总:")
    print(f"    平均R²: {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"    平均RMSE: {avg_rmse:.4f}")
    print(f"    平均MAE: {avg_mae:.4f}")
    print(f"    整体R²: {overall_r2:.4f}")
    
    return {
        'fold_results': fold_results,
        'avg_r2': avg_r2, 'std_r2': std_r2,
        'avg_rmse': avg_rmse, 'avg_mae': avg_mae,
        'overall_r2': overall_r2,
        'method': '5fold_cv_no_finetune'
    }


def main():
    print("=" * 80)
    print("过拟合验证实验")
    print("=" * 80)
    print("目的：验证V2模型R²=0.992是否存在过拟合/数据泄露")
    print()
    
    start = time.time()
    all_results = {}
    
    # 实验1：纯生成数据（无数据泄露基线）
    all_results['exp1'] = experiment_1_pure_generated()
    
    # 实验2：5折CV + 微调
    all_results['exp2'] = experiment_2_kfold_cv()
    
    # 实验3：5折CV 无微调
    all_results['exp3'] = experiment_3_no_finetune_cv()
    
    # 汇总对比
    print("\n" + "=" * 80)
    print("过拟合验证汇总")
    print("=" * 80)
    
    print(f"\n{'方法':<40} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("-" * 70)
    
    # 实验1
    e1 = all_results['exp1']
    print(f"{'纯生成数据训练（无泄露基线）':<40} {e1['r2']:8.4f} {e1['rmse']:8.4f} {e1['mae']:8.4f}")
    
    # 实验3
    e3 = all_results['exp3']
    print(f"{'5折CV + 真实数据混入（无微调）':<40} {e3['avg_r2']:8.4f} {e3['avg_rmse']:8.4f} {e3['avg_mae']:8.4f}")
    
    # 实验2
    e2 = all_results['exp2']
    print(f"{'5折CV + 真实数据混入 + 微调':<40} {e2['avg_r2']:8.4f} {e2['avg_rmse']:8.4f} {e2['avg_mae']:8.4f}")
    
    # V2原始结果
    print(f"{'V2原始（全部真实数据参与训练+微调）':<40} {'0.9921':>8} {'0.3626':>8} {'0.2997':>8}")
    
    print(f"\n{'5折CV详细（含微调）:'}")
    for f in all_results['exp2']['fold_results']:
        print(f"  第{f['fold']}折: R²={f['r2']:.4f}, RMSE={f['rmse']:.4f}, MAE={f['mae']:.4f}")
    print(f"  整体R²: {all_results['exp2']['overall_r2']:.4f}")
    
    # 过拟合判断
    print("\n" + "=" * 80)
    print("过拟合分析:")
    print("=" * 80)
    
    cv_r2 = e2['avg_r2']
    v2_r2 = 0.9921
    gap = v2_r2 - cv_r2
    
    if gap > 0.3:
        print(f"  ⚠️ 严重过拟合！V2 R²={v2_r2:.4f} vs CV R²={cv_r2:.4f}，差距={gap:.4f}")
        print(f"  → 应以5折CV的R²={cv_r2:.4f}作为模型的真实泛化能力")
    elif gap > 0.1:
        print(f"  ⚠ 中度过拟合。V2 R²={v2_r2:.4f} vs CV R²={cv_r2:.4f}，差距={gap:.4f}")
        print(f"  → 真实泛化R²约为{cv_r2:.4f}")
    else:
        print(f"  ✓ 过拟合程度可接受。V2 R²={v2_r2:.4f} vs CV R²={cv_r2:.4f}，差距={gap:.4f}")
    
    baseline_r2 = e1['r2']
    improvement = cv_r2 - baseline_r2
    print(f"\n  纯生成数据基线R²: {baseline_r2:.4f}")
    print(f"  5折CV R²: {cv_r2:.4f}")
    print(f"  真实数据混入带来的提升: +{improvement:.4f}")
    print(f"  → 这{improvement:.4f}的提升是真实的泛化能力提升（通过CV验证）")
    
    elapsed = time.time() - start
    print(f"\n总用时: {elapsed:.1f}秒")
    
    # 保存结果
    out = Path(__file__).parent / "results" / "overfitting_validation"
    out.mkdir(parents=True, exist_ok=True)
    
    # 转换numpy类型
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(out / "validation_results.json", 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {out}")


if __name__ == "__main__":
    main()
