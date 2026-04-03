#!/usr/bin/env python3
"""
保留交互效应结构的10万样本数据生成器
===========================================
核心思路：
1. 从真实103样本拟合包含显著交互项的回归模型（捕获交互效应结构）
2. 使用Gaussian Copula生成基础特征（保留边际分布和相关结构）
3. 基于回归模型重新生成目标变量（保留交互效应）
4. 添加校准噪声使残差分布匹配真实数据
5. 验证生成数据的交互效应保留度

解决的核心问题：
- 原copula方法只保留了Spearman相关，丢失了高阶交互效应
- 新方法通过回归模型显式编码34个显著交互效应到生成过程中
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from pathlib import Path
from itertools import combinations
import statsmodels.api as sm
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
COLUMN_MAPPING = {
    '序号': 'sample_id',
    '跨文化适应程度': 'cross_cultural_adaptation',
    '文化保持': 'cultural_maintenance',
    '社会保持': 'social_maintenance',
    '文化接触': 'cultural_contact',
    '社会接触': 'social_contact',
    '家庭支持': 'family_support',
    '家庭沟通频率': 'comm_frequency_feeling',
    '沟通坦诚度': 'comm_openness',
    '自主权': 'personal_autonomy',
    '社会联结感': 'social_connection',
    '开放性': 'openness',
    '来港时长': 'months_in_hk'
}

COLUMN_MAPPING_REV = {v: k for k, v in COLUMN_MAPPING.items()}

FEATURES_CN = ['文化保持', '社会保持', '文化接触', '社会接触', '家庭支持',
               '家庭沟通频率', '沟通坦诚度', '自主权', '社会联结感', '开放性', '来港时长']

FEATURES_EN = [COLUMN_MAPPING[f] for f in FEATURES_CN]
TARGET_CN = '跨文化适应程度'
TARGET_EN = 'cross_cultural_adaptation'

ALL_VARS_EN = FEATURES_EN + [TARGET_EN]

# 显著的二阶交互 (p < 0.05) + 边际显著 (p < 0.1)
SIGNIFICANT_2WAY_CN = [
    ('文化接触', '开放性'),        # p=0.021
    ('家庭支持', '开放性'),        # p=0.035
    ('社会接触', '开放性'),        # p=0.046
    ('家庭沟通频率', '自主权'),    # p=0.051
    ('家庭支持', '来港时长'),      # p=0.058
    ('文化保持', '开放性'),        # p=0.059
    ('社会接触', '来港时长'),      # p=0.072
    ('沟通坦诚度', '自主权'),      # p=0.094
]

# 显著的三阶交互 (p < 0.05)
SIGNIFICANT_3WAY_CN = [
    ('社会接触', '家庭支持', '社会联结感'),     # p=0.004
    ('社会接触', '自主权', '来港时长'),          # p=0.007
    ('文化接触', '家庭支持', '社会联结感'),      # p=0.008
    ('文化接触', '社会接触', '来港时长'),        # p=0.012
    ('文化接触', '家庭支持', '家庭沟通频率'),    # p=0.022
    ('社会接触', '沟通坦诚度', '社会联结感'),    # p=0.031
    ('沟通坦诚度', '自主权', '开放性'),          # p=0.032
    ('社会接触', '家庭支持', '来港时长'),        # p=0.036
    ('文化接触', '自主权', '社会联结感'),        # p=0.040
    ('沟通坦诚度', '社会联结感', '开放性'),      # p=0.044
]

# 最显著的四阶交互 (p < 0.05, top 11)
SIGNIFICANT_4WAY_CN = [
    ('文化保持', '家庭支持', '家庭沟通频率', '社会联结感'),  # p=0.005
    ('文化保持', '自主权', '社会联结感', '来港时长'),        # p=0.006
    ('文化保持', '社会保持', '家庭支持', '社会联结感'),      # p=0.009
    ('社会保持', '家庭支持', '家庭沟通频率', '自主权'),      # p=0.011
    ('社会接触', '家庭沟通频率', '自主权', '开放性'),        # p=0.011
    ('社会接触', '家庭支持', '家庭沟通频率', '开放性'),      # p=0.012
    ('社会接触', '沟通坦诚度', '自主权', '开放性'),          # p=0.013
    ('文化保持', '家庭支持', '自主权', '开放性'),            # p=0.016
    ('文化保持', '社会保持', '家庭支持', '自主权'),          # p=0.022
    ('社会保持', '家庭支持', '家庭沟通频率', '社会联结感'),  # p=0.024
    ('社会接触', '自主权', '开放性', '来港时长'),            # p=0.025
]

# 非线性项
QUADRATIC_CN = ['开放性']  # p=0.013


# ============================================================
# 阶段1：拟合包含交互效应的回归模型
# ============================================================
class InteractionRegressionModel:
    """包含显著交互项的回归模型，用于生成保留交互效应的目标变量"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.residual_std = None
        self.residual_distribution = None
        self.y_mean = None
        self.y_std = None
        
    def _create_interaction_features(self, X_scaled, feature_names):
        """创建交互特征矩阵"""
        feat_idx = {name: i for i, name in enumerate(feature_names)}
        interaction_cols = []
        interaction_names = []
        
        # 二阶交互
        for f1, f2 in SIGNIFICANT_2WAY_CN:
            col = X_scaled[:, feat_idx[f1]] * X_scaled[:, feat_idx[f2]]
            interaction_cols.append(col)
            interaction_names.append(f"{f1}×{f2}")
        
        # 三阶交互
        for combo in SIGNIFICANT_3WAY_CN:
            col = np.ones(len(X_scaled))
            for f in combo:
                col = col * X_scaled[:, feat_idx[f]]
            interaction_cols.append(col)
            interaction_names.append('×'.join(combo))
        
        # 四阶交互
        for combo in SIGNIFICANT_4WAY_CN:
            col = np.ones(len(X_scaled))
            for f in combo:
                col = col * X_scaled[:, feat_idx[f]]
            interaction_cols.append(col)
            interaction_names.append('×'.join(combo))
        
        # 二次项
        for f in QUADRATIC_CN:
            col = X_scaled[:, feat_idx[f]] ** 2
            interaction_cols.append(col)
            interaction_names.append(f"{f}²")
        
        if interaction_cols:
            interactions = np.column_stack(interaction_cols)
            X_full = np.hstack([X_scaled, interactions])
            all_names = list(feature_names) + interaction_names
        else:
            X_full = X_scaled
            all_names = list(feature_names)
        
        return X_full, all_names
    
    def fit(self, df_real):
        """在真实数据上拟合包含交互项的回归模型"""
        print("\n" + "=" * 80)
        print("阶段1：拟合包含交互效应的回归模型")
        print("=" * 80)
        
        X = df_real[FEATURES_CN].values.astype(float)
        y = df_real[TARGET_CN].values.astype(float)
        
        self.y_mean = y.mean()
        self.y_std = y.std()
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建交互特征
        X_full, self.feature_names = self._create_interaction_features(X_scaled, FEATURES_CN)
        
        print(f"  基础特征: {len(FEATURES_CN)}")
        print(f"  二阶交互: {len(SIGNIFICANT_2WAY_CN)}")
        print(f"  三阶交互: {len(SIGNIFICANT_3WAY_CN)}")
        print(f"  四阶交互: {len(SIGNIFICANT_4WAY_CN)}")
        print(f"  二次项: {len(QUADRATIC_CN)}")
        print(f"  总特征数: {X_full.shape[1]}")
        
        # 使用ElasticNet回归（L1+L2正则化，更好处理高维交互项）
        # 较小的alpha以保留更多交互效应信号
        from sklearn.linear_model import ElasticNetCV
        alphas = np.logspace(-4, 1, 50)
        self.model = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.3, 0.5, 0.7], cv=5, max_iter=10000)
        self.model.fit(X_full, y)
        
        y_pred = self.model.predict(X_full)
        r2 = r2_score(y, y_pred)
        residuals = y - y_pred
        self.residual_std = residuals.std()
        
        # 拟合残差分布（使用核密度估计）
        self.residual_kde = stats.gaussian_kde(residuals)
        self.residual_min = residuals.min()
        self.residual_max = residuals.max()
        
        print(f"\n  回归模型结果:")
        print(f"    最佳alpha: {self.model.alpha_:.4f}")
        print(f"    训练R²: {r2:.4f}")
        print(f"    残差标准差: {self.residual_std:.4f}")
        print(f"    残差范围: [{residuals.min():.2f}, {residuals.max():.2f}]")
        
        # 打印重要系数
        coefs = list(zip(self.feature_names, self.model.coef_))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Top 15 特征系数:")
        for name, coef in coefs[:15]:
            print(f"    {name:<45s}: {coef:8.4f}")
        
        # 使用statsmodels获取p值（用于验证）
        X_full_const = sm.add_constant(X_full)
        ols_model = sm.OLS(y, X_full_const).fit()
        print(f"\n  OLS模型 R²: {ols_model.rsquared:.4f}, Adj R²: {ols_model.rsquared_adj:.4f}")
        
        return r2
    
    def predict(self, X_raw):
        """对原始特征进行预测（含交互项）"""
        X_scaled = self.scaler.transform(X_raw)
        X_full, _ = self._create_interaction_features(X_scaled, FEATURES_CN)
        return self.model.predict(X_full)
    
    def predict_with_noise(self, X_raw, noise_scale=1.0):
        """预测并添加校准噪声"""
        y_pred = self.predict(X_raw)
        # 从残差KDE中采样噪声
        noise = self.residual_kde.resample(len(y_pred)).flatten()
        noise = np.clip(noise, self.residual_min * 1.5, self.residual_max * 1.5)
        return y_pred + noise * noise_scale


# ============================================================
# 阶段2：基于Copula生成基础特征
# ============================================================
class CopulaFeatureGenerator:
    """使用Gaussian Copula生成保留相关结构的基础特征"""
    
    def __init__(self):
        self.marginals = {}
        self.normal_corr = None
        
    def fit(self, df_real):
        """拟合边际分布和相关结构"""
        print("\n" + "=" * 80)
        print("阶段2：拟合Copula模型（仅特征，不含目标）")
        print("=" * 80)
        
        # 只对特征拟合（目标变量将由回归模型生成）
        for col in FEATURES_CN:
            data = df_real[col].dropna().values.astype(float)
            
            data_min = float(data.min())
            data_max = float(data.max())
            data_mean = float(data.mean())
            data_std = float(data.std())
            is_integer = np.all(data == data.astype(int))
            
            # 拟合多种分布
            best_dist = None
            best_params = None
            best_ks = 1.0
            
            for dist_name, dist_obj in [('norm', stats.norm), ('lognorm', stats.lognorm),
                                         ('gamma', stats.gamma), ('beta', stats.beta)]:
                try:
                    params = dist_obj.fit(data)
                    ks_stat, _ = stats.kstest(data, dist_name, args=params)
                    if ks_stat < best_ks:
                        best_ks = ks_stat
                        best_dist = dist_name
                        best_params = params
                except:
                    continue
            
            if best_dist is None:
                best_dist = 'norm'
                best_params = stats.norm.fit(data)
            
            self.marginals[col] = {
                'distribution': best_dist,
                'params': best_params,
                'min': data_min,
                'max': data_max,
                'mean': data_mean,
                'std': data_std,
                'ks_stat': best_ks,
                'is_integer': is_integer,
            }
            
            print(f"  {col}: dist={best_dist}, KS={best_ks:.4f}, "
                  f"range=[{data_min:.0f}, {data_max:.0f}], int={is_integer}")
        
        # 计算Spearman相关矩阵（仅特征间）
        spearman_corr = df_real[FEATURES_CN].corr(method='spearman').values
        
        # 转换为正态copula的Pearson相关
        n = len(FEATURES_CN)
        self.normal_corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.normal_corr[i, j] = 1.0
                else:
                    self.normal_corr[i, j] = 2 * np.sin(np.pi / 6 * spearman_corr[i, j])
        
        # 确保正定
        eigvals = np.linalg.eigvalsh(self.normal_corr)
        if np.min(eigvals) < 0:
            print("  修正相关矩阵为正定...")
            self.normal_corr = self._nearest_positive_definite(self.normal_corr)
        
        print(f"\n  特征间Spearman相关矩阵已拟合 ({n}x{n})")
    
    def generate(self, n_samples, random_state=42):
        """生成n_samples个特征样本"""
        print(f"\n  生成 {n_samples} 个特征样本...")
        np.random.seed(random_state)
        
        n_features = len(FEATURES_CN)
        
        # 多元正态采样
        z_samples = np.random.multivariate_normal(
            np.zeros(n_features), self.normal_corr, size=n_samples
        )
        
        # 转换为均匀分布
        u_samples = stats.norm.cdf(z_samples)
        
        # 逆CDF转换
        data = np.zeros((n_samples, n_features))
        for i, col in enumerate(FEATURES_CN):
            m = self.marginals[col]
            dist_obj = getattr(stats, m['distribution'])
            raw_values = dist_obj.ppf(u_samples[:, i], *m['params'])
            
            # 截断
            range_ext = 0.05 * (m['max'] - m['min'])
            raw_values = np.clip(raw_values, m['min'] - range_ext, m['max'] + range_ext)
            
            # 整数化
            if m['is_integer']:
                raw_values = np.round(raw_values).astype(float)
                raw_values = np.clip(raw_values, m['min'], m['max'])
            
            data[:, i] = raw_values
        
        df_gen = pd.DataFrame(data, columns=FEATURES_CN)
        
        # 打印统计对比
        print(f"\n  生成特征统计:")
        for col in FEATURES_CN:
            m = self.marginals[col]
            gen_mean = df_gen[col].mean()
            gen_std = df_gen[col].std()
            print(f"    {col}: mean={gen_mean:.2f}(real:{m['mean']:.2f}), "
                  f"std={gen_std:.2f}(real:{m['std']:.2f}), "
                  f"range=[{df_gen[col].min():.1f}, {df_gen[col].max():.1f}]")
        
        return df_gen
    
    @staticmethod
    def _nearest_positive_definite(A):
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while True:
            try:
                np.linalg.cholesky(A3)
                return A3
            except np.linalg.LinAlgError:
                mineig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += I * (-mineig * k**2 + spacing)
                k += 1
                if k > 100:
                    return A3


# ============================================================
# 阶段3：生成目标变量并校准
# ============================================================
class TargetGenerator:
    """基于交互回归模型生成目标变量"""
    
    def __init__(self, regression_model):
        self.reg_model = regression_model
    
    def generate(self, df_features, noise_scale=1.0):
        """生成目标变量"""
        print("\n" + "=" * 80)
        print("阶段3：基于交互回归模型生成目标变量")
        print("=" * 80)
        
        X_raw = df_features[FEATURES_CN].values.astype(float)
        
        # 使用回归模型预测（包含交互效应）
        y_pred = self.reg_model.predict(X_raw)
        
        # 添加校准噪声
        noise = self.reg_model.residual_kde.resample(len(y_pred)).flatten()
        noise = np.clip(noise, 
                       self.reg_model.residual_min * 1.5, 
                       self.reg_model.residual_max * 1.5)
        y_gen = y_pred + noise * noise_scale
        
        # 截断到合理范围
        y_min = 8   # 真实数据最小值
        y_max = 32  # 真实数据最大值
        y_gen = np.clip(y_gen, y_min - 1, y_max + 1)
        
        # 四舍五入为整数（原始数据是整数）
        y_gen = np.round(y_gen).astype(float)
        y_gen = np.clip(y_gen, y_min, y_max)
        
        print(f"  预测部分: mean={y_pred.mean():.2f}, std={y_pred.std():.2f}")
        print(f"  噪声部分: mean={noise.mean():.2f}, std={noise.std():.2f}")
        print(f"  最终目标: mean={y_gen.mean():.2f}, std={np.std(y_gen):.2f}, "
              f"range=[{y_gen.min():.0f}, {y_gen.max():.0f}]")
        print(f"  真实数据: mean={self.reg_model.y_mean:.2f}, std={self.reg_model.y_std:.2f}")
        
        return y_gen


# ============================================================
# 阶段4：验证交互效应保留度
# ============================================================
class InteractionValidator:
    """验证生成数据中的交互效应是否被保留"""
    
    def __init__(self):
        pass
    
    def validate(self, df_gen, df_real):
        """全面验证"""
        print("\n" + "=" * 80)
        print("阶段4：验证交互效应保留度")
        print("=" * 80)
        
        results = {}
        
        # 1. 相关性保留
        results['correlation'] = self._validate_correlations(df_gen, df_real)
        
        # 2. 分布相似性
        results['distribution'] = self._validate_distributions(df_gen, df_real)
        
        # 3. 二阶交互效应保留
        results['2way'] = self._validate_2way_interactions(df_gen, df_real)
        
        # 4. 三阶交互效应保留（top 3）
        results['3way'] = self._validate_3way_interactions(df_gen, df_real)
        
        # 5. 四阶交互效应保留（top 3）
        results['4way'] = self._validate_4way_interactions(df_gen, df_real)
        
        # 6. 非线性效应
        results['nonlinear'] = self._validate_nonlinear(df_gen, df_real)
        
        # 7. 基线模型R²对比
        results['baseline'] = self._validate_baseline_model(df_gen, df_real)
        
        return results
    
    def _validate_correlations(self, df_gen, df_real):
        """验证特征-目标相关性"""
        print("\n  1. 特征-目标Spearman相关性对比:")
        print(f"    {'特征':<12} {'真实':>8} {'生成':>8} {'差异':>8} {'状态':>4}")
        print(f"    {'-'*44}")
        
        diffs = []
        for col in FEATURES_CN:
            r_corr = df_real[col].corr(df_real[TARGET_CN], method='spearman')
            g_corr = df_gen[col].corr(df_gen[TARGET_CN], method='spearman')
            diff = abs(r_corr - g_corr)
            diffs.append(diff)
            status = "✓" if diff < 0.1 else "⚠" if diff < 0.2 else "✗"
            print(f"    {col:<12} {r_corr:8.4f} {g_corr:8.4f} {diff:8.4f} {status:>4}")
        
        avg_diff = np.mean(diffs)
        print(f"    平均差异: {avg_diff:.4f}")
        return {'avg_diff': avg_diff, 'diffs': diffs}
    
    def _validate_distributions(self, df_gen, df_real):
        """验证分布相似性"""
        print("\n  2. 分布KS检验:")
        ks_results = {}
        for col in FEATURES_CN + [TARGET_CN]:
            ks_stat, p_val = stats.ks_2samp(df_real[col].values, df_gen[col].values)
            status = "✓" if p_val > 0.01 else "⚠"
            print(f"    {col:<12}: KS={ks_stat:.4f}, p={p_val:.4f} {status}")
            ks_results[col] = {'ks_stat': ks_stat, 'p_value': p_val}
        return ks_results
    
    def _validate_2way_interactions(self, df_gen, df_real):
        """验证二阶交互效应"""
        print("\n  3. 二阶交互效应保留度:")
        
        scaler_real = StandardScaler()
        X_real = pd.DataFrame(scaler_real.fit_transform(df_real[FEATURES_CN]), columns=FEATURES_CN)
        y_real = df_real[TARGET_CN].values
        
        scaler_gen = StandardScaler()
        X_gen = pd.DataFrame(scaler_gen.fit_transform(df_gen[FEATURES_CN]), columns=FEATURES_CN)
        y_gen = df_gen[TARGET_CN].values
        
        results = []
        print(f"    {'交互项':<25} {'真实p值':>10} {'生成p值':>10} {'方向一致':>8}")
        print(f"    {'-'*58}")
        
        for f1, f2 in SIGNIFICANT_2WAY_CN:
            # 真实数据
            X_test_r = X_real.copy()
            X_test_r[f'{f1}×{f2}'] = X_real[f1] * X_real[f2]
            X_test_r = sm.add_constant(X_test_r)
            model_r = sm.OLS(y_real, X_test_r).fit()
            coef_r = model_r.params[f'{f1}×{f2}']
            pval_r = model_r.pvalues[f'{f1}×{f2}']
            
            # 生成数据（采样1000个以加速）
            sample_idx = np.random.choice(len(X_gen), min(5000, len(X_gen)), replace=False)
            X_test_g = X_gen.iloc[sample_idx].copy()
            X_test_g[f'{f1}×{f2}'] = X_gen.iloc[sample_idx][f1] * X_gen.iloc[sample_idx][f2]
            X_test_g = sm.add_constant(X_test_g)
            model_g = sm.OLS(y_gen[sample_idx], X_test_g).fit()
            coef_g = model_g.params[f'{f1}×{f2}']
            pval_g = model_g.pvalues[f'{f1}×{f2}']
            
            direction_match = (coef_r > 0) == (coef_g > 0)
            status = "✓" if direction_match and pval_g < 0.1 else "⚠" if direction_match else "✗"
            
            print(f"    {f1}×{f2:<12} {pval_r:10.4f} {pval_g:10.4f} {status:>8}")
            results.append({
                'interaction': f'{f1}×{f2}',
                'real_p': pval_r, 'gen_p': pval_g,
                'real_coef': coef_r, 'gen_coef': coef_g,
                'direction_match': direction_match
            })
        
        match_rate = sum(1 for r in results if r['direction_match']) / len(results)
        sig_rate = sum(1 for r in results if r['gen_p'] < 0.1) / len(results)
        print(f"    方向一致率: {match_rate:.1%}, 显著保留率: {sig_rate:.1%}")
        return results
    
    def _validate_3way_interactions(self, df_gen, df_real):
        """验证三阶交互效应（top 3）"""
        print("\n  4. 三阶交互效应保留度 (Top 3):")
        
        scaler_real = StandardScaler()
        X_real = pd.DataFrame(scaler_real.fit_transform(df_real[FEATURES_CN]), columns=FEATURES_CN)
        y_real = df_real[TARGET_CN].values
        
        scaler_gen = StandardScaler()
        X_gen = pd.DataFrame(scaler_gen.fit_transform(df_gen[FEATURES_CN]), columns=FEATURES_CN)
        y_gen = df_gen[TARGET_CN].values
        
        results = []
        top3 = SIGNIFICANT_3WAY_CN[:3]
        
        for combo in top3:
            f1, f2, f3 = combo
            name = f'{f1}×{f2}×{f3}'
            
            # 真实数据
            X_test_r = X_real.copy()
            X_test_r[f'{f1}×{f2}'] = X_real[f1] * X_real[f2]
            X_test_r[f'{f1}×{f3}'] = X_real[f1] * X_real[f3]
            X_test_r[f'{f2}×{f3}'] = X_real[f2] * X_real[f3]
            X_test_r[name] = X_real[f1] * X_real[f2] * X_real[f3]
            X_test_r = sm.add_constant(X_test_r)
            model_r = sm.OLS(y_real, X_test_r).fit()
            coef_r = model_r.params[name]
            pval_r = model_r.pvalues[name]
            
            # 生成数据
            sample_idx = np.random.choice(len(X_gen), min(5000, len(X_gen)), replace=False)
            X_test_g = X_gen.iloc[sample_idx].copy()
            X_test_g[f'{f1}×{f2}'] = X_gen.iloc[sample_idx][f1] * X_gen.iloc[sample_idx][f2]
            X_test_g[f'{f1}×{f3}'] = X_gen.iloc[sample_idx][f1] * X_gen.iloc[sample_idx][f3]
            X_test_g[f'{f2}×{f3}'] = X_gen.iloc[sample_idx][f2] * X_gen.iloc[sample_idx][f3]
            X_test_g[name] = X_gen.iloc[sample_idx][f1] * X_gen.iloc[sample_idx][f2] * X_gen.iloc[sample_idx][f3]
            X_test_g = sm.add_constant(X_test_g)
            model_g = sm.OLS(y_gen[sample_idx], X_test_g).fit()
            coef_g = model_g.params[name]
            pval_g = model_g.pvalues[name]
            
            direction_match = (coef_r > 0) == (coef_g > 0)
            status = "✓" if direction_match and pval_g < 0.1 else "⚠" if direction_match else "✗"
            
            print(f"    {name[:40]:<42} real_p={pval_r:.4f} gen_p={pval_g:.4f} {status}")
            results.append({
                'interaction': name,
                'real_p': pval_r, 'gen_p': pval_g,
                'direction_match': direction_match
            })
        
        return results
    
    def _validate_4way_interactions(self, df_gen, df_real):
        """验证四阶交互效应（top 3）"""
        print("\n  5. 四阶交互效应保留度 (Top 3):")
        
        scaler_real = StandardScaler()
        X_real = pd.DataFrame(scaler_real.fit_transform(df_real[FEATURES_CN]), columns=FEATURES_CN)
        y_real = df_real[TARGET_CN].values
        
        scaler_gen = StandardScaler()
        X_gen = pd.DataFrame(scaler_gen.fit_transform(df_gen[FEATURES_CN]), columns=FEATURES_CN)
        y_gen = df_gen[TARGET_CN].values
        
        results = []
        top3 = SIGNIFICANT_4WAY_CN[:3]
        
        for combo in top3:
            f1, f2, f3, f4 = combo
            name = f'{f1}×{f2}×{f3}×{f4}'
            
            # 真实数据
            X_test_r = X_real.copy()
            for p1, p2 in combinations(combo, 2):
                X_test_r[f'{p1}×{p2}'] = X_real[p1] * X_real[p2]
            for t1, t2, t3 in combinations(combo, 3):
                X_test_r[f'{t1}×{t2}×{t3}'] = X_real[t1] * X_real[t2] * X_real[t3]
            X_test_r[name] = X_real[f1] * X_real[f2] * X_real[f3] * X_real[f4]
            X_test_r = sm.add_constant(X_test_r)
            model_r = sm.OLS(y_real, X_test_r).fit()
            coef_r = model_r.params[name]
            pval_r = model_r.pvalues[name]
            
            # 生成数据
            sample_idx = np.random.choice(len(X_gen), min(5000, len(X_gen)), replace=False)
            X_test_g = X_gen.iloc[sample_idx].copy()
            for p1, p2 in combinations(combo, 2):
                X_test_g[f'{p1}×{p2}'] = X_gen.iloc[sample_idx][p1] * X_gen.iloc[sample_idx][p2]
            for t1, t2, t3 in combinations(combo, 3):
                X_test_g[f'{t1}×{t2}×{t3}'] = X_gen.iloc[sample_idx][t1] * X_gen.iloc[sample_idx][t2] * X_gen.iloc[sample_idx][t3]
            X_test_g[name] = X_gen.iloc[sample_idx][f1] * X_gen.iloc[sample_idx][f2] * X_gen.iloc[sample_idx][f3] * X_gen.iloc[sample_idx][f4]
            X_test_g = sm.add_constant(X_test_g)
            model_g = sm.OLS(y_gen[sample_idx], X_test_g).fit()
            coef_g = model_g.params[name]
            pval_g = model_g.pvalues[name]
            
            direction_match = (coef_r > 0) == (coef_g > 0)
            status = "✓" if direction_match and pval_g < 0.1 else "⚠" if direction_match else "✗"
            
            short_name = name[:50]
            print(f"    {short_name:<52} real_p={pval_r:.4f} gen_p={pval_g:.4f} {status}")
            results.append({
                'interaction': name,
                'real_p': pval_r, 'gen_p': pval_g,
                'direction_match': direction_match
            })
        
        return results
    
    def _validate_nonlinear(self, df_gen, df_real):
        """验证非线性效应"""
        print("\n  6. 非线性效应（开放性²）:")
        
        for label, df in [('真实', df_real), ('生成', df_gen)]:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(df[FEATURES_CN]), columns=FEATURES_CN)
            y = df[TARGET_CN].values
            
            X_quad = X.copy()
            X_quad['开放性²'] = X['开放性'] ** 2
            X_quad = sm.add_constant(X_quad)
            
            if label == '生成':
                sample_idx = np.random.choice(len(X_quad), min(5000, len(X_quad)), replace=False)
                model = sm.OLS(y[sample_idx], X_quad.iloc[sample_idx]).fit()
            else:
                model = sm.OLS(y, X_quad).fit()
            
            coef = model.params['开放性²']
            pval = model.pvalues['开放性²']
            print(f"    {label}数据: 开放性² coef={coef:.4f}, p={pval:.4f}")
        
        return {'validated': True}
    
    def _validate_baseline_model(self, df_gen, df_real):
        """验证基线模型R²"""
        print("\n  7. 基线模型（仅主效应）R²对比:")
        
        for label, df in [('真实(103)', df_real), ('生成(5000样本)', df_gen)]:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(df[FEATURES_CN]), columns=FEATURES_CN)
            y = df[TARGET_CN].values
            
            if label.startswith('生成'):
                sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
                X_const = sm.add_constant(X.iloc[sample_idx])
                model = sm.OLS(y[sample_idx], X_const).fit()
            else:
                X_const = sm.add_constant(X)
                model = sm.OLS(y, X_const).fit()
            
            print(f"    {label}: R²={model.rsquared:.4f}, Adj R²={model.rsquared_adj:.4f}")
        
        return {'validated': True}


# ============================================================
# 主流程
# ============================================================
def main():
    start_time = time.time()
    np.random.seed(42)
    
    print("=" * 80)
    print("保留交互效应结构的10万样本数据生成")
    print("=" * 80)
    print(f"目标: 生成保留34个显著交互效应的10万样本数据集")
    print(f"方法: Copula特征生成 + 交互回归模型目标生成 + 残差校准")
    
    # 加载真实数据
    real_path = Path(__file__).parent / "data" / "processed" / "real_data_103.xlsx"
    df_real = pd.read_excel(real_path)
    print(f"\n真实数据: {len(df_real)} 样本, {len(df_real.columns)} 列")
    
    # ---- 阶段1: 拟合交互回归模型 ----
    reg_model = InteractionRegressionModel()
    r2_train = reg_model.fit(df_real)
    
    # ---- 阶段2: 拟合Copula并生成特征 ----
    copula_gen = CopulaFeatureGenerator()
    copula_gen.fit(df_real)
    
    n_samples = 100000
    df_features = copula_gen.generate(n_samples, random_state=42)
    
    # ---- 阶段3: 生成目标变量 ----
    target_gen = TargetGenerator(reg_model)
    y_gen = target_gen.generate(df_features, noise_scale=1.0)
    
    # 组合完整数据集
    df_gen = df_features.copy()
    df_gen[TARGET_CN] = y_gen
    
    # ---- 阶段4: 验证 ----
    validator = InteractionValidator()
    validation_results = validator.validate(df_gen, df_real)
    
    # ---- 保存数据 ----
    print("\n" + "=" * 80)
    print("保存数据")
    print("=" * 80)
    
    # 转换为英文列名
    df_gen_en = df_gen.rename(columns=COLUMN_MAPPING)
    df_gen_en['batch_id'] = np.repeat(range(5), n_samples // 5)
    df_gen_en['sample_id'] = range(n_samples)
    
    output_path = Path(__file__).parent / "data" / "processed" / "interaction_preserved_100k.csv"
    df_gen_en.to_csv(output_path, index=False)
    print(f"  数据已保存: {output_path}")
    print(f"  形状: {df_gen_en.shape}")
    
    # 也保存中文版本
    output_path_cn = Path(__file__).parent / "data" / "processed" / "interaction_preserved_100k_cn.csv"
    df_gen['sample_id'] = range(n_samples)
    df_gen.to_csv(output_path_cn, index=False, encoding='utf-8-sig')
    print(f"  中文版已保存: {output_path_cn}")
    
    # 保存验证报告
    report = {
        'method': 'Copula + Interaction Regression + Residual Calibration',
        'n_samples': n_samples,
        'n_features': len(FEATURES_CN),
        'n_2way_interactions': len(SIGNIFICANT_2WAY_CN),
        'n_3way_interactions': len(SIGNIFICANT_3WAY_CN),
        'n_4way_interactions': len(SIGNIFICANT_4WAY_CN),
        'n_quadratic': len(QUADRATIC_CN),
        'regression_r2': float(r2_train),
        'correlation_avg_diff': float(validation_results['correlation']['avg_diff']),
        'generation_time_seconds': time.time() - start_time,
    }
    
    report_path = Path(__file__).parent / "data" / "processed" / "interaction_preserved_generation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  报告已保存: {report_path}")
    
    elapsed = time.time() - start_time
    print(f"\n✅ 数据生成完成! 用时: {elapsed:.1f}秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
