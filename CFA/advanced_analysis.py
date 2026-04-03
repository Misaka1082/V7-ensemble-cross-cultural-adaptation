# -*- coding: utf-8 -*-
"""
高级分析脚本：交互效应、非线性关系、Fisher's Z检验
法国样本分析 + 与香港样本对比
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os

# ============================================================
# 数据加载
# ============================================================

def load_data():
    """加载香港和法国原始条目数据"""
    hk_df = pd.read_excel('f:/Project/4.1.9.final/CFA/HKN=75.xlsx', header=0)
    fr_df = pd.read_excel('f:/Project/4.1.9.final/CFA/Franchn=249.xlsx', header=0)
    
    col_names = [
        'duration',
        'adapt1', 'adapt2', 'adapt3', 'adapt4', 'adapt5', 'adapt6', 'adapt7', 'adapt8',
        'cultural_maintenance', 'social_maintenance',
        'cultural_contact', 'social_contact',
        'family_support1', 'family_support2', 'family_support3', 'family_support4',
        'family_support5', 'family_support6', 'family_support7', 'family_support8',
        'comm_frequency', 'comm_openness',
        'autonomy1', 'autonomy2',
        'social_connect1', 'social_connect2', 'social_connect3', 'social_connect4', 'social_connect5',
        'openness'
    ]
    
    hk_df.columns = col_names
    fr_df.columns = col_names
    hk_df = hk_df.dropna(how='all').reset_index(drop=True)
    fr_df = fr_df.dropna(how='all').reset_index(drop=True)
    
    # 计算量表总分
    for df in [hk_df, fr_df]:
        df['adaptation'] = df[['adapt1','adapt2','adapt3','adapt4','adapt5','adapt6','adapt7','adapt8']].mean(axis=1)
        df['family_support'] = df[['family_support1','family_support2','family_support3','family_support4',
                                    'family_support5','family_support6','family_support7','family_support8']].mean(axis=1)
        df['autonomy'] = df[['autonomy1','autonomy2']].mean(axis=1)
        df['social_connectedness'] = df[['social_connect1','social_connect2','social_connect3',
                                          'social_connect4','social_connect5']].mean(axis=1)
    
    print(f"香港样本量: {len(hk_df)}, 法国样本量: {len(fr_df)}")
    return hk_df, fr_df


def get_predictor_vars():
    """V7模型使用的预测变量"""
    return {
        'duration': '居留时长',
        'cultural_maintenance': '文化保持',
        'social_maintenance': '社会保持',
        'cultural_contact': '文化接触',
        'social_contact': '社会接触',
        'family_support': '家庭支持',
        'comm_frequency': '沟通频率',
        'comm_openness': '沟通坦诚度',
        'autonomy': '自主权',
        'social_connectedness': '社会联结感',
        'openness': '开放性'
    }


# ============================================================
# 第一部分：交互效应分析
# ============================================================

def compute_interaction_effects(df, dv='adaptation'):
    """计算2阶交互效应"""
    from itertools import combinations
    
    predictors = get_predictor_vars()
    pred_cols = list(predictors.keys())
    
    # 标准化
    df_std = df.copy()
    for col in pred_cols + [dv]:
        if df_std[col].std() > 0:
            df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
    
    results = []
    
    # 基线模型 R²
    X_base = df_std[pred_cols].dropna()
    y = df_std[dv].loc[X_base.index]
    
    try:
        from numpy.linalg import lstsq
        X_mat = np.column_stack([np.ones(len(X_base)), X_base.values])
        beta, _, _, _ = lstsq(X_mat, y.values, rcond=None)
        y_pred = X_mat @ beta
        ss_res = np.sum((y.values - y_pred) ** 2)
        ss_tot = np.sum((y.values - y.values.mean()) ** 2)
        r2_base = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    except:
        r2_base = 0
    
    # 2阶交互
    for f1, f2 in combinations(pred_cols, 2):
        interaction = df_std[f1] * df_std[f2]
        X_int = np.column_stack([np.ones(len(X_base)), X_base.values, interaction.loc[X_base.index].values])
        
        try:
            beta, _, _, _ = lstsq(X_int, y.values, rcond=None)
            y_pred = X_int @ beta
            ss_res = np.sum((y.values - y_pred) ** 2)
            r2_int = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # 交互项系数和显著性
            coef = beta[-1]
            n = len(y)
            p = X_int.shape[1]
            mse = ss_res / (n - p) if (n - p) > 0 else 1
            
            try:
                var_beta = mse * np.linalg.inv(X_int.T @ X_int)
                se = np.sqrt(abs(var_beta[-1, -1]))
                t_stat = coef / se if se > 0 else 0
                from scipy import stats
                p_val = 2 * stats.t.sf(abs(t_stat), n - p)
            except:
                p_val = 1.0
                t_stat = 0
            
            delta_r2 = r2_int - r2_base
            
            results.append({
                'f1': f1,
                'f2': f2,
                'f1_cn': predictors[f1],
                'f2_cn': predictors[f2],
                'interaction': f"{predictors[f1]}×{predictors[f2]}",
                'coef': round(coef, 4),
                'p_val': round(p_val, 4),
                'r2': round(r2_int, 4),
                'delta_r2': round(delta_r2, 4),
                'abs_coef': round(abs(coef), 4)
            })
        except:
            continue
    
    return pd.DataFrame(results).sort_values('abs_coef', ascending=False)


def compute_3way_interactions(df, dv='adaptation', top_n=20):
    """计算3阶交互效应（仅计算top组合）"""
    from itertools import combinations
    
    predictors = get_predictor_vars()
    pred_cols = list(predictors.keys())
    
    df_std = df.copy()
    for col in pred_cols + [dv]:
        if df_std[col].std() > 0:
            df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
    
    X_base = df_std[pred_cols].dropna()
    y = df_std[dv].loc[X_base.index]
    
    from numpy.linalg import lstsq
    X_mat = np.column_stack([np.ones(len(X_base)), X_base.values])
    beta, _, _, _ = lstsq(X_mat, y.values, rcond=None)
    y_pred = X_mat @ beta
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    ss_res_base = np.sum((y.values - y_pred) ** 2)
    r2_base = 1 - ss_res_base / ss_tot if ss_tot > 0 else 0
    
    results = []
    
    for f1, f2, f3 in combinations(pred_cols, 3):
        int_12 = df_std[f1] * df_std[f2]
        int_13 = df_std[f1] * df_std[f3]
        int_23 = df_std[f2] * df_std[f3]
        int_123 = df_std[f1] * df_std[f2] * df_std[f3]
        
        X_int = np.column_stack([
            np.ones(len(X_base)), X_base.values,
            int_12.loc[X_base.index].values,
            int_13.loc[X_base.index].values,
            int_23.loc[X_base.index].values,
            int_123.loc[X_base.index].values
        ])
        
        try:
            beta, _, _, _ = lstsq(X_int, y.values, rcond=None)
            y_pred = X_int @ beta
            ss_res = np.sum((y.values - y_pred) ** 2)
            r2_int = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            coef_3way = beta[-1]
            delta_r2 = r2_int - r2_base
            
            n = len(y)
            p = X_int.shape[1]
            mse = ss_res / (n - p) if (n - p) > 0 else 1
            try:
                var_beta = mse * np.linalg.inv(X_int.T @ X_int)
                se = np.sqrt(abs(var_beta[-1, -1]))
                t_stat = coef_3way / se if se > 0 else 0
                from scipy import stats
                p_val = 2 * stats.t.sf(abs(t_stat), n - p)
            except:
                p_val = 1.0
            
            results.append({
                'f1': f1, 'f2': f2, 'f3': f3,
                'interaction': f"{predictors[f1]}×{predictors[f2]}×{predictors[f3]}",
                'coef_3way': round(coef_3way, 4),
                'p_val': round(p_val, 4),
                'r2': round(r2_int, 4),
                'delta_r2': round(delta_r2, 4),
                'abs_coef': round(abs(coef_3way), 4)
            })
        except:
            continue
    
    return pd.DataFrame(results).sort_values('abs_coef', ascending=False).head(top_n)


def run_interaction_analysis(hk_df, fr_df):
    """运行交互效应分析"""
    print("\n" + "="*80)
    print("交互效应分析（法国样本 + 与香港对比）")
    print("="*80)
    
    # 2阶交互
    print("\n━━━ 2阶交互效应 ━━━")
    
    print("\n── 法国样本 Top 20 ──")
    fr_2way = compute_interaction_effects(fr_df)
    print(f"{'排名':>4} {'交互项':<30} {'系数':>8} {'p值':>8} {'ΔR²':>8} {'显著性':>6}")
    for i, (_, row) in enumerate(fr_2way.head(20).iterrows()):
        sig = "***" if row['p_val'] < 0.001 else "**" if row['p_val'] < 0.01 else "*" if row['p_val'] < 0.05 else "†" if row['p_val'] < 0.1 else ""
        print(f"{i+1:>4} {row['interaction']:<30} {row['coef']:>8.4f} {row['p_val']:>8.4f} {row['delta_r2']:>8.4f} {sig:>6}")
    
    print("\n── 香港样本 Top 20 ──")
    hk_2way = compute_interaction_effects(hk_df)
    print(f"{'排名':>4} {'交互项':<30} {'系数':>8} {'p值':>8} {'ΔR²':>8} {'显著性':>6}")
    for i, (_, row) in enumerate(hk_2way.head(20).iterrows()):
        sig = "***" if row['p_val'] < 0.001 else "**" if row['p_val'] < 0.01 else "*" if row['p_val'] < 0.05 else "†" if row['p_val'] < 0.1 else ""
        print(f"{i+1:>4} {row['interaction']:<30} {row['coef']:>8.4f} {row['p_val']:>8.4f} {row['delta_r2']:>8.4f} {sig:>6}")
    
    # 跨文化对比
    print("\n── 2阶交互效应跨文化对比 (Top 10) ──")
    print(f"{'交互项':<30} {'香港系数':>10} {'法国系数':>10} {'方向一致':>10}")
    
    fr_dict = {row['interaction']: row['coef'] for _, row in fr_2way.iterrows()}
    hk_dict = {row['interaction']: row['coef'] for _, row in hk_2way.iterrows()}
    
    all_interactions = set(list(fr_dict.keys())[:10] + list(hk_dict.keys())[:10])
    for interaction in sorted(all_interactions, key=lambda x: abs(fr_dict.get(x, 0)) + abs(hk_dict.get(x, 0)), reverse=True)[:15]:
        hk_c = hk_dict.get(interaction, 0)
        fr_c = fr_dict.get(interaction, 0)
        same_dir = "✓" if (hk_c * fr_c > 0) else "✗" if (hk_c != 0 and fr_c != 0) else "—"
        print(f"{interaction:<30} {hk_c:>10.4f} {fr_c:>10.4f} {same_dir:>10}")
    
    # 3阶交互
    print("\n\n━━━ 3阶交互效应 ━━━")
    
    print("\n── 法国样本 Top 15 ──")
    fr_3way = compute_3way_interactions(fr_df, top_n=15)
    print(f"{'排名':>4} {'交互项':<45} {'系数':>8} {'p值':>8} {'ΔR²':>8}")
    for i, (_, row) in enumerate(fr_3way.iterrows()):
        sig = "***" if row['p_val'] < 0.001 else "**" if row['p_val'] < 0.01 else "*" if row['p_val'] < 0.05 else "†" if row['p_val'] < 0.1 else ""
        print(f"{i+1:>4} {row['interaction']:<45} {row['coef_3way']:>8.4f} {row['p_val']:>8.4f} {row['delta_r2']:>8.4f} {sig}")
    
    print("\n── 香港样本 Top 15 ──")
    hk_3way = compute_3way_interactions(hk_df, top_n=15)
    print(f"{'排名':>4} {'交互项':<45} {'系数':>8} {'p值':>8} {'ΔR²':>8}")
    for i, (_, row) in enumerate(hk_3way.iterrows()):
        sig = "***" if row['p_val'] < 0.001 else "**" if row['p_val'] < 0.01 else "*" if row['p_val'] < 0.05 else "†" if row['p_val'] < 0.1 else ""
        print(f"{i+1:>4} {row['interaction']:<45} {row['coef_3way']:>8.4f} {row['p_val']:>8.4f} {row['delta_r2']:>8.4f} {sig}")
    
    return fr_2way, hk_2way, fr_3way, hk_3way


# ============================================================
# 第二部分：非线性关系分析
# ============================================================

def analyze_nonlinear(df, x_col, y_col='adaptation', x_label='', sample_name=''):
    """分析非线性关系（线性 vs 二次）"""
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].values
    y = data[y_col].values
    
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    # 线性模型
    X_lin = np.column_stack([np.ones(n), x])
    beta_lin, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
    y_pred_lin = X_lin @ beta_lin
    ss_res_lin = np.sum((y - y_pred_lin) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0
    
    # 二次模型
    X_quad = np.column_stack([np.ones(n), x, x**2])
    beta_quad, _, _, _ = np.linalg.lstsq(X_quad, y, rcond=None)
    y_pred_quad = X_quad @ beta_quad
    ss_res_quad = np.sum((y - y_pred_quad) ** 2)
    r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0
    
    # F检验（二次项是否显著）
    df1 = 1  # 增加的参数数
    df2 = n - 3  # 二次模型残差自由度
    if df2 > 0 and ss_res_quad > 0:
        f_stat = ((ss_res_lin - ss_res_quad) / df1) / (ss_res_quad / df2)
        try:
            from scipy import stats
            p_val = stats.f.sf(f_stat, df1, df2)
        except:
            p_val = 0.05 if f_stat > 4 else 0.5
    else:
        f_stat = 0
        p_val = 1.0
    
    # 判断曲线类型
    quad_coef = beta_quad[2]
    if quad_coef < 0:
        curve_type = "倒U型" if p_val < 0.1 else "近似线性"
    else:
        curve_type = "U型" if p_val < 0.1 else "近似线性"
    
    # 极值点
    if abs(beta_quad[2]) > 1e-10:
        vertex_x = -beta_quad[1] / (2 * beta_quad[2])
        vertex_y = beta_quad[0] + beta_quad[1] * vertex_x + beta_quad[2] * vertex_x**2
    else:
        vertex_x = vertex_y = np.nan
    
    return {
        'variable': x_label,
        'sample': sample_name,
        'linear_coef': round(beta_lin[1], 4),
        'r2_linear': round(r2_lin, 4),
        'quad_coef': round(beta_quad[2], 4),
        'r2_quad': round(r2_quad, 4),
        'delta_r2': round(r2_quad - r2_lin, 4),
        'f_stat': round(f_stat, 4),
        'p_val': round(p_val, 4),
        'curve_type': curve_type,
        'vertex_x': round(vertex_x, 2) if not np.isnan(vertex_x) else np.nan,
        'vertex_y': round(vertex_y, 2) if not np.isnan(vertex_y) else np.nan,
        'x_range': f"[{x.min():.1f}, {x.max():.1f}]",
        'x_mean': round(x.mean(), 2),
        'beta_quad': beta_quad
    }


def run_nonlinear_analysis(hk_df, fr_df):
    """运行非线性关系分析"""
    print("\n" + "="*80)
    print("非线性关系分析（法国样本 + 与香港对比）")
    print("="*80)
    
    variables = [
        ('openness', '开放性'),
        ('duration', '居留时长'),
        ('autonomy', '自主权'),
        ('cultural_contact', '文化接触'),
        ('social_contact', '社会接触'),
        ('family_support', '家庭支持'),
        ('social_connectedness', '社会联结感'),
        ('cultural_maintenance', '文化保持'),
        ('social_maintenance', '社会保持'),
        ('comm_frequency', '沟通频率'),
        ('comm_openness', '沟通坦诚度'),
    ]
    
    all_results = []
    
    for var_col, var_label in variables:
        hk_res = analyze_nonlinear(hk_df, var_col, x_label=var_label, sample_name='香港')
        fr_res = analyze_nonlinear(fr_df, var_col, x_label=var_label, sample_name='法国')
        all_results.extend([hk_res, fr_res])
    
    # 打印结果
    print(f"\n{'变量':<15} {'样本':>6} {'线性R²':>8} {'二次R²':>8} {'ΔR²':>6} {'二次系数':>10} {'F值':>8} {'p值':>8} {'曲线类型':>10} {'极值点':>10}")
    print("─" * 110)
    
    for res in all_results:
        sig = "***" if res['p_val'] < 0.001 else "**" if res['p_val'] < 0.01 else "*" if res['p_val'] < 0.05 else "†" if res['p_val'] < 0.1 else ""
        vertex = f"x={res['vertex_x']}" if not np.isnan(res['vertex_x']) else "—"
        print(f"{res['variable']:<15} {res['sample']:>6} {res['r2_linear']:>8.4f} {res['r2_quad']:>8.4f} {res['delta_r2']:>6.4f} {res['quad_coef']:>10.4f} {res['f_stat']:>8.2f} {res['p_val']:>8.4f} {res['curve_type']:>10} {vertex:>10} {sig}")
    
    # 重点变量详细分析
    print("\n\n━━━ 重点非线性关系详细分析 ━━━")
    
    focus_vars = ['openness', 'duration', 'autonomy']
    focus_labels = {'openness': '开放性', 'duration': '居留时长', 'autonomy': '自主权'}
    
    for var_col in focus_vars:
        var_label = focus_labels[var_col]
        print(f"\n{'─'*60}")
        print(f"  {var_label} 与跨文化适应的非线性关系")
        print(f"{'─'*60}")
        
        for sample_name, df in [('香港', hk_df), ('法国', fr_df)]:
            res = analyze_nonlinear(df, var_col, x_label=var_label, sample_name=sample_name)
            print(f"\n  {sample_name}样本:")
            print(f"    线性模型: y = {res['linear_coef']:.4f}x + b, R² = {res['r2_linear']:.4f}")
            print(f"    二次模型: y = {res['quad_coef']:.4f}x² + bx + c, R² = {res['r2_quad']:.4f}")
            print(f"    ΔR² = {res['delta_r2']:.4f}, F = {res['f_stat']:.2f}, p = {res['p_val']:.4f}")
            print(f"    曲线类型: {res['curve_type']}")
            if not np.isnan(res['vertex_x']):
                in_range = "在数据范围内" if float(res['x_range'].split(',')[0][1:]) <= res['vertex_x'] <= float(res['x_range'].split(',')[1][:-1]) else "在数据范围外"
                print(f"    极值点: x = {res['vertex_x']}, y = {res['vertex_y']} ({in_range})")
            print(f"    数据范围: {res['x_range']}, 均值 = {res['x_mean']}")
    
    # 生成ASCII图
    print("\n\n━━━ 非线性关系可视化（ASCII图） ━━━")
    
    for var_col in focus_vars:
        var_label = focus_labels[var_col]
        print(f"\n  {var_label} → 跨文化适应")
        
        for sample_name, df in [('香港', hk_df), ('法国', fr_df)]:
            res = analyze_nonlinear(df, var_col, x_label=var_label, sample_name=sample_name)
            beta = res['beta_quad']
            
            data = df[[var_col, 'adaptation']].dropna()
            x_min, x_max = data[var_col].min(), data[var_col].max()
            x_range = np.linspace(x_min, x_max, 40)
            y_pred = beta[0] + beta[1] * x_range + beta[2] * x_range**2
            
            # 简单ASCII图
            y_min, y_max = y_pred.min(), y_pred.max()
            height = 10
            width = 40
            
            print(f"\n  {sample_name} ({res['curve_type']}, R²={res['r2_quad']:.3f}):")
            
            grid = [[' ' for _ in range(width)] for _ in range(height)]
            for i, (xi, yi) in enumerate(zip(x_range, y_pred)):
                col = int((xi - x_min) / (x_max - x_min + 1e-10) * (width - 1))
                row = height - 1 - int((yi - y_min) / (y_max - y_min + 1e-10) * (height - 1))
                row = max(0, min(height - 1, row))
                col = max(0, min(width - 1, col))
                grid[row][col] = '●'
            
            for row in grid:
                print(f"  │{''.join(row)}│")
            print(f"  └{'─' * width}┘")
            print(f"   {x_min:.1f}{' ' * (width - 8)}{x_max:.1f}")
    
    return all_results


# ============================================================
# 第三部分：Fisher's Z检验
# ============================================================

def fisher_z_test(r1, n1, r2, n2):
    """Fisher's Z检验：比较两个相关系数是否显著不同"""
    # Fisher Z变换
    z1 = 0.5 * np.log((1 + r1) / (1 - r1)) if abs(r1) < 1 else np.sign(r1) * 3
    z2 = 0.5 * np.log((1 + r2) / (1 - r2)) if abs(r2) < 1 else np.sign(r2) * 3
    
    # 标准误
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    
    # Z统计量
    z_stat = (z1 - z2) / se
    
    # 双尾p值
    try:
        from scipy import stats
        p_val = 2 * stats.norm.sf(abs(z_stat))
    except:
        p_val = 2 * (1 - 0.5 * (1 + np.math.erf(abs(z_stat) / np.sqrt(2))))
    
    return z_stat, p_val


def run_fisher_z_analysis(hk_df, fr_df):
    """运行Fisher's Z检验"""
    print("\n" + "="*80)
    print("Fisher's Z检验：相关系数跨文化差异显著性")
    print("="*80)
    
    predictors = get_predictor_vars()
    pred_cols = list(predictors.keys())
    dv = 'adaptation'
    
    n_hk = len(hk_df)
    n_fr = len(fr_df)
    
    # 1. 各预测变量与因变量的相关
    print(f"\n━━━ 预测变量与跨文化适应的相关系数比较 ━━━")
    print(f"\n{'变量':<15} {'香港r':>8} {'法国r':>8} {'差异':>8} {'Fisher Z':>10} {'p值':>8} {'显著性':>6} {'解释':>15}")
    print("─" * 90)
    
    fisher_results = []
    
    for col in pred_cols:
        r_hk = hk_df[[col, dv]].dropna().corr().iloc[0, 1]
        r_fr = fr_df[[col, dv]].dropna().corr().iloc[0, 1]
        
        z_stat, p_val = fisher_z_test(r_hk, n_hk, r_fr, n_fr)
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "†" if p_val < 0.1 else "ns"
        
        diff = r_hk - r_fr
        if p_val < 0.05:
            if abs(diff) > 0.3:
                explain = "显著差异(大)"
            elif abs(diff) > 0.15:
                explain = "显著差异(中)"
            else:
                explain = "显著差异(小)"
        else:
            explain = "无显著差异"
        
        fisher_results.append({
            'variable': predictors[col],
            'r_hk': round(r_hk, 4),
            'r_fr': round(r_fr, 4),
            'diff': round(diff, 4),
            'z_stat': round(z_stat, 4),
            'p_val': round(p_val, 4),
            'sig': sig,
            'explain': explain
        })
        
        print(f"{predictors[col]:<15} {r_hk:>8.4f} {r_fr:>8.4f} {diff:>8.4f} {z_stat:>10.4f} {p_val:>8.4f} {sig:>6} {explain:>15}")
    
    # 2. 预测变量间的相关矩阵比较
    print(f"\n\n━━━ 预测变量间相关系数差异检验 (仅显示显著差异) ━━━")
    print(f"\n{'变量对':<35} {'香港r':>8} {'法国r':>8} {'Z值':>8} {'p值':>8} {'显著性':>6}")
    print("─" * 80)
    
    sig_count = 0
    total_count = 0
    
    from itertools import combinations
    for c1, c2 in combinations(pred_cols, 2):
        r_hk = hk_df[[c1, c2]].dropna().corr().iloc[0, 1]
        r_fr = fr_df[[c1, c2]].dropna().corr().iloc[0, 1]
        
        z_stat, p_val = fisher_z_test(r_hk, n_hk, r_fr, n_fr)
        total_count += 1
        
        if p_val < 0.05:
            sig_count += 1
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            pair_name = f"{predictors[c1]}-{predictors[c2]}"
            print(f"{pair_name:<35} {r_hk:>8.4f} {r_fr:>8.4f} {z_stat:>8.4f} {p_val:>8.4f} {sig:>6}")
    
    print(f"\n  共检验 {total_count} 对相关系数，其中 {sig_count} 对存在显著差异 ({sig_count/total_count*100:.1f}%)")
    
    # 3. 总结
    print(f"\n\n━━━ Fisher's Z检验总结 ━━━")
    
    sig_vars = [r for r in fisher_results if r['p_val'] < 0.05]
    nonsig_vars = [r for r in fisher_results if r['p_val'] >= 0.05]
    
    print(f"\n  与跨文化适应的相关系数存在显著跨文化差异的变量 ({len(sig_vars)}个):")
    for r in sig_vars:
        direction = "香港更强" if abs(r['r_hk']) > abs(r['r_fr']) else "法国更强"
        print(f"    - {r['variable']}: 香港r={r['r_hk']:.3f}, 法国r={r['r_fr']:.3f} ({direction}, p={r['p_val']:.4f})")
    
    print(f"\n  无显著跨文化差异的变量 ({len(nonsig_vars)}个):")
    for r in nonsig_vars:
        print(f"    - {r['variable']}: 香港r={r['r_hk']:.3f}, 法国r={r['r_fr']:.3f} (p={r['p_val']:.4f})")
    
    return fisher_results


# ============================================================
# 主函数
# ============================================================

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  高级分析：交互效应 + 非线性关系 + Fisher's Z检验               ║")
    print("║  法国样本分析 + 与香港样本跨文化对比                             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    hk_df, fr_df = load_data()
    
    # 1. 交互效应分析
    fr_2way, hk_2way, fr_3way, hk_3way = run_interaction_analysis(hk_df, fr_df)
    
    # 2. 非线性关系分析
    nonlinear_results = run_nonlinear_analysis(hk_df, fr_df)
    
    # 3. Fisher's Z检验
    fisher_results = run_fisher_z_analysis(hk_df, fr_df)
    
    print("\n\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
