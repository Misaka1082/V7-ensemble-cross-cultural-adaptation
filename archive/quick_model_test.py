#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速模型完整性验证脚本
验证 train_final_robust.py 的模型架构、数据加载、训练流程是否正常
使用小规模数据快速验证（约1-2分钟）
"""
import sys
import os
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from train_final_robust import (
    EnsembleTrainer, compute_interaction_features,
    INPUT_FEATURES, TARGET_COL, RobustHybridModel, N_INTERACTIONS
)

print("=" * 70)
print("快速模型完整性验证")
print("=" * 70)

# 1. 测试模型架构
print("\n[1] 测试模型架构...")
model = RobustHybridModel(
    n_features=len(INPUT_FEATURES),
    n_interactions=N_INTERACTIONS,
    embedding_size=16,
    hidden_layers=[128, 64, 32],
    dropout=0.3
)
total_params = sum(p.numel() for p in model.parameters())
print(f"  模型参数量: {total_params:,}")
# 测试前向传播
dummy_input = torch.randn(8, len(INPUT_FEATURES) + N_INTERACTIONS)
output = model(dummy_input)
assert output.shape == (8, 1), f"输出形状错误: {output.shape}"
print(f"  前向传播: OK (输入{dummy_input.shape} -> 输出{output.shape})")

# 2. 测试数据加载
print("\n[2] 测试数据加载...")
trainer = EnsembleTrainer(n_models=2, epochs=20, batch_size=256, lr=0.001, patience=5)
trainer.load_data()
print(f"  生成数据: {len(trainer.gen_data)} 样本, {len(INPUT_FEATURES)} 特征")
print(f"  真实数据: {len(trainer.real_data)} 样本")

# 3. 测试交互特征计算
print("\n[3] 测试交互特征计算...")
X_test = trainer.gen_data[INPUT_FEATURES].values[:100]
interactions = compute_interaction_features(X_test)
print(f"  原始特征: {X_test.shape[1]}, 交互特征: {interactions.shape[1]}, 总计: {X_test.shape[1] + interactions.shape[1]}")
assert interactions.shape[1] == N_INTERACTIONS, f"交互特征数量错误: {interactions.shape[1]} != {N_INTERACTIONS}"
print(f"  交互特征计算: OK")

# 4. 快速训练测试（小数据集）
print("\n[4] 快速训练测试（2000样本，20轮）...")
gen = trainer.gen_data.copy()
real = trainer.real_data.copy()
scalers = {}
for col in INPUT_FEATURES:
    scaler = StandardScaler()
    gen[col] = scaler.fit_transform(gen[[col]])
    real[col] = scaler.transform(real[[col]])
    scalers[col] = scaler
trainer.scalers = scalers

X_gen = gen[INPUT_FEATURES].values[:2000]
y_gen = gen[TARGET_COL].values[:2000]
X_real = real[INPUT_FEATURES].values
y_real = real[TARGET_COL].values

X_gen_full = trainer._prepare_features(X_gen)
X_real_full = trainer._prepare_features(X_real)

model_trained, val_r2 = trainer._train_single_model(
    X_gen_full, y_gen, X_real_full, y_real, seed=42
)
trainer.models = [model_trained]

model_trained.eval()
with torch.no_grad():
    preds = model_trained(torch.FloatTensor(X_real_full).to(trainer.device)).cpu().numpy().flatten()
r2 = r2_score(y_real, preds)
print(f"  快速训练R2（真实数据）: {r2:.4f}")
print(f"  训练流程: OK")

# 5. 测试集成预测
print("\n[5] 测试集成预测...")
ensemble_preds = trainer._ensemble_predict(X_real_full)
ensemble_r2 = r2_score(y_real, ensemble_preds)
print(f"  集成预测R2: {ensemble_r2:.4f}")
print(f"  集成预测: OK")

# 6. 测试已保存的模型文件
print("\n[6] 验证已保存的模型文件...")
model_dir = Path(__file__).parent / "results" / "final_robust"
saved_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.json")) + list(model_dir.glob("*.csv"))
for f in sorted(saved_files):
    size_kb = f.stat().st_size / 1024
    print(f"  OK  {f.name} ({size_kb:.0f} KB)")

# 7. 加载已保存的模型并验证
print("\n[7] 加载已保存的模型验证...")
config_path = model_dir / "ensemble_config.pth"
if config_path.exists():
    config = torch.load(str(config_path), map_location='cpu', weights_only=False)
    print(f"  集成配置: {config['n_models']} 个模型, {len(config['input_features'])} 个特征")
    
    # 加载第一个模型
    model_path = model_dir / "ensemble_model_0.pth"
    if model_path.exists():
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        loaded_model = RobustHybridModel(
            n_features=len(INPUT_FEATURES),
            n_interactions=N_INTERACTIONS,
        )
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.eval()
        with torch.no_grad():
            test_out = loaded_model(torch.randn(4, len(INPUT_FEATURES) + N_INTERACTIONS))
        print(f"  已保存模型加载: OK (输出形状: {test_out.shape})")
else:
    print("  [警告] 未找到已保存的模型配置文件")

print("\n" + "=" * 70)
print("模型完整性验证通过！")
print("=" * 70)
print("\n模型架构摘要:")
print(f"  - 输入特征: {len(INPUT_FEATURES)} 个原始特征 + {N_INTERACTIONS} 个交互特征")
print(f"  - 模型类型: RobustHybridModel (线性+FM+Deep+门控融合)")
print(f"  - 参数量: {total_params:,}")
print(f"  - 集成数量: {config['n_models'] if config_path.exists() else 'N/A'} 个模型")
print(f"  - 快速测试R2: {r2:.4f}")
