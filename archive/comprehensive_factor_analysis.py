#!/usr/bin/env python3
"""
跨文化适应影响因素的全面分析
系统探索所有变量的直接效应、二阶交互、三阶交互
寻找类似开放性的关键影响因素和潜在逻辑关系
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置
COLUMN_MAPPING = {
    '序号': 'sample_id', '跨文化适应程度': 'cross_cultural_adaptation',
    '文化保持': 'cultural_maintenance', '社会保持': 'social_maintenance',
    '文化接触': 'cultural_contact', '社会接触': 'social_contact',
    '家庭支持': 'family_support', '家庭沟通频率': 'comm_frequency_feeling',
    '沟通坦诚度': 'comm_openness', '自主权': 'personal_autonomy',
    '社会联结感': 'social_connection', '开放性': 'openness', '来港时长': 'months_in_hk'
}

FEATURES_CN = ['文化保持', '社会保持', '文化接触', '社会接触', '家庭支持',
               '家庭沟通频率', '沟通坦诚度', '自主权', '社会联结感', '开放性', '来港时长']
TARGET_CN = '跨文化适应程度'

def load_data():
    """加载真实数据和合成数据"""
    base_path = Path(__file__).parent
    
    # 加载真实数据
    real_path = base_path / "data" / "processed" / "real_data_103.xlsx"
    df_real = pd.read_excel(real_path)
    print(f"真实数据: {len(df_real)} 样本")
    
    # 加载合成数据
    synthetic_path = base_path / "data" / "processed" / "interaction_preserved_100k.csv"
    df_synthetic = pd.read_csv(synthetic_path)
    df_synthetic = df_synthetic.rename(columns={v: k for k, v in COLUMN_MAPPING.items()})
    print(f"合成数据: {len(df_synthetic)} 样本")
    
    return df_real, df_synthetic

def analyze_all_main_effects(df, title):
    """分析所有变量的主效应（包括二次项）"""
    print(f"\n{'='*80}")
    print(f"{title} - 所有变量主效应分析")
    print(f"{'='*80}")
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[FEATURES_CN]), columns=FEATURES_CN)
    y = df[TARGET_CN].values
    
    results = []
    
    for feature in FEATURES_CN:
        # 线性模型
        X_linear = X[[feature]].copy()
        X_linear = sm.add_constant(X_linear)
        model_linear = sm.OLS(y, X_linear).fit()
        
        # 二次模型
        X_quad = X[[feature]].copy()
        X_quad[f'{feature}²'] = X[feature] ** 2
        X_quad = sm.add_constant(X_quad)
        model_quad = sm.OLS(y, X_quad).fit()
        
        # 完整模型中的效应
        X_full = X.copy()
        X_full = sm.add_constant(X_full)
        model_full = sm.OLS(y, X_full).fit()
        
        # 完整模型+二次项
        X_full_quad = X.copy()
        X_full_quad[f'{feature}²'] = X[feature] ** 2
        X_full_quad = sm.add_constant(X_full_quad)
        model_full_quad = sm.OLS(y, X_full_quad).fit()
        
        results.append({
            'feature': feature,
            'linear_r2': model_linear.rsquared,
            'linear_coef': model_linear.params[feature],
            'linear_p': model_linear.pvalues[feature],
            'quad_r2': model_quad.rsquared,
            'quad_coef': model_quad.params[feature],
            'quad_p': model_quad.pvalues[feature],
            'quad2_coef': model_quad.params[f'{feature}²'],
            'quad2_p': model_quad.pvalues[f'{feature}²'],
            'full_coef': model_full.params[feature],
            'full_p': model_full.pvalues[feature],
            'full_quad_coef': model_full_quad.params[feature],
            'full_quad_p': model_full_quad.pvalues[feature],
            'full_quad2_coef': model_full_quad.params[f'{feature}²'],
            'full_quad2_p': model_full_quad.pvalues[f'{feature}²'],
        })
    
    # 转换为DataFrame并排序
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('linear_r2', ascending=False)
    
    print(f"\n{'特征':<12} {'线性R²':<10} {'线性系数':<10} {'线性p值':<10} {'二次R²':<10} {'二次²系数':<12} {'二次²p值':<10}")
    print("-" * 90)
    for _, row in df_results.iterrows():
        sig_linear = '***' if row['linear_p'] < 0.01 else '**' if row['linear_p'] < 0.05 else '*' if row['linear_p'] < 0.1 else ''
        sig_quad = '***' if row['quad2_p'] < 0.01 else '**' if row['quad2_p'] < 0.05 else '*' if row['quad2_p'] < 0.1 else ''
        print(f"{row['feature']:<12} {row['linear_r2']:<10.4f} {row['linear_coef']:<10.4f} {row['linear_p']:<10.4f}{sig_linear:<3} "
              f"{row['quad_r2']:<10.4f} {row['quad2_coef']:<12.4f} {row['quad2_p']:<10.4f}{sig_quad:<3}")
    
    return df_results

def analyze_all_2way_interactions(df, title, top_n=20):
    """分析所有可能的二阶交互效应"""
    print(f"\n{'='*80}")
    print(f"{title} - 所有二阶交互效应分析（Top {top_n}）")
    print(f"{'='*80}")
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[FEATURES_CN]), columns=FEATURES_CN)
    y = df[TARGET_CN].values
    
    results = []
    
    # 测试所有可能的二阶交互
    for f1, f2 in combinations(FEATURES_CN, 2):
        X_test = X.copy()
        X_test[f'{f1}×{f2}'] = X[f1] * X[f2]
        X_test = sm.add_constant(X_test)
        model = sm.OLS(y, X_test).fit()
        
        coef = model.params[f'{f1}×{f2}']
        pval = model.pvalues[f'{f1}×{f2}']
        r2 = model.rsquared
        
        results.append({
            'interaction': f'{f1}×{f2}',
            'f1': f1,
            'f2': f2,
            'coef': coef,
            'pval': pval,
            'r2': r2,
            'abs_coef': abs(coef)
        })
    
    # 转换为DataFrame并排序
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('abs_coef', ascending=False)
    
    print(f"\n{'排名':<6} {'交互项':<40} {'系数':<12} {'p值':<12} {'R²':<10} {'显著性':<6}")
    print("-" * 90)
    for i, (_, row) in enumerate(df_results.head(top_n).iterrows(), 1):
        sig = '***' if row['pval'] < 0.01 else '**' if row['pval'] < 0.05 else '*' if row['pval'] < 0.1 else ''
        print(f"{i:<6} {row['interaction']:<40} {row['coef']:<12.4f} {row['pval']:<12.6f} {row['r2']:<10.4f} {sig:<6}")
    
    return df_results

def analyze_all_3way_interactions(df, title, top_n=20):
    """分析重要的三阶交互效应"""
    print(f"\n{'='*80}")
    print(f"{title} - 三阶交互效应分析（Top {top_n}）")
    print(f"{'='*80}")
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[FEATURES_CN]), columns=FEATURES_CN)
    y = df[TARGET_CN].values
    
    results = []
    
    # 选择重要的变量进行三阶交互测试（避免组合爆炸）
    important_features = ['文化接触', '社会接触', '家庭支持', '沟通坦诚度', 
                         '自主权', '社会联结感', '开放性', '家庭沟通频率']
    
    print(f"\n正在测试 {len(list(combinations(important_features, 3)))} 个三阶交互...")
    
    for f1, f2, f3 in combinations(important_features, 3):
        X_test = X.copy()
        # 添加二阶交互
        X_test[f'{f1}×{f2}'] = X[f1] * X[f2]
        X_test[f'{f1}×{f3}'] = X[f1] * X[f3]
        X_test[f'{f2}×{f3}'] = X[f2] * X[f3]
        # 添加三阶交互
        X_test[f'{f1}×{f2}×{f3}'] = X[f1] * X[f2] * X[f3]
        X_test = sm.add_constant(X_test)
        model = sm.OLS(y, X_test).fit()
        
        coef = model.params[f'{f1}×{f2}×{f3}']
        pval = model.pvalues[f'{f1}×{f2}×{f3}']
        r2 = model.rsquared
        
        results.append({
            'interaction': f'{f1}×{f2}×{f3}',
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'coef': coef,
            'pval': pval,
            'r2': r2,
            'abs_coef': abs(coef)
        })
    
    # 转换为DataFrame并排序
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('abs_coef', ascending=False)
    
    print(f"\n{'排名':<6} {'三阶交互项':<60} {'系数':<12} {'p值':<12} {'R²':<10} {'显著性':<6}")
    print("-" * 110)
    for i, (_, row) in enumerate(df_results.head(top_n).iterrows(), 1):
        sig = '***' if row['pval'] < 0.01 else '**' if row['pval'] < 0.05 else '*' if row['pval'] < 0.1 else ''
        print(f"{i:<6} {row['interaction']:<60} {row['coef']:<12.4f} {row['pval']:<12.6f} {row['r2']:<10.4f} {sig:<6}")
    
    return df_results

def identify_key_factors(main_effects, interactions_2way, interactions_3way):
    """识别关键影响因素"""
    print(f"\n{'='*80}")
    print("关键影响因素识别")
    print(f"{'='*80}")
    
    # 1. 主效应强的因素
    print("\n【1. 主效应最强的因素】")
    top_main = main_effects.nlargest(5, 'linear_r2')
    for i, (_, row) in enumerate(top_main.iterrows(), 1):
        print(f"{i}. {row['feature']}: R²={row['linear_r2']:.4f}, 系数={row['linear_coef']:.4f}, p={row['linear_p']:.4f}")
    
    # 2. 有显著非线性效应的因素
    print("\n【2. 显著非线性效应（二次项显著）】")
    nonlinear = main_effects[main_effects['quad2_p'] < 0.05].sort_values('quad2_p')
    for i, (_, row) in enumerate(nonlinear.iterrows(), 1):
        print(f"{i}. {row['feature']}: 二次项系数={row['quad2_coef']:.4f}, p={row['quad2_p']:.4f}")
    
    # 3. 在二阶交互中频繁出现的因素
    print("\n【3. 二阶交互中最活跃的因素】")
    sig_interactions = interactions_2way[interactions_2way['pval'] < 0.05]
    factor_counts = {}
    for _, row in sig_interactions.iterrows():
        factor_counts[row['f1']] = factor_counts.get(row['f1'], 0) + 1
        factor_counts[row['f2']] = factor_counts.get(row['f2'], 0) + 1
    
    sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (factor, count) in enumerate(sorted_factors[:8], 1):
        print(f"{i}. {factor}: 参与 {count} 个显著交互")
    
    # 4. 在三阶交互中频繁出现的因素
    print("\n【4. 三阶交互中最活跃的因素】")
    sig_interactions_3way = interactions_3way[interactions_3way['pval'] < 0.05]
    factor_counts_3way = {}
    for _, row in sig_interactions_3way.iterrows():
        factor_counts_3way[row['f1']] = factor_counts_3way.get(row['f1'], 0) + 1
        factor_counts_3way[row['f2']] = factor_counts_3way.get(row['f2'], 0) + 1
        factor_counts_3way[row['f3']] = factor_counts_3way.get(row['f3'], 0) + 1
    
    sorted_factors_3way = sorted(factor_counts_3way.items(), key=lambda x: x[1], reverse=True)
    for i, (factor, count) in enumerate(sorted_factors_3way[:8], 1):
        print(f"{i}. {factor}: 参与 {count} 个显著三阶交互")
    
    return {
        'top_main': top_main,
        'nonlinear': nonlinear,
        'active_2way': sorted_factors[:8],
        'active_3way': sorted_factors_3way[:8]
    }

def plot_top_factors_comparison(df_real, df_synthetic, main_effects_real, main_effects_synthetic, save_path):
    """绘制关键因素对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 主效应R²对比
    ax1 = axes[0, 0]
    top_features = main_effects_real.nlargest(8, 'linear_r2')['feature'].tolist()
    
    real_r2 = [main_effects_real[main_effects_real['feature']==f]['linear_r2'].values[0] for f in top_features]
    synth_r2 = [main_effects_synthetic[main_effects_synthetic['feature']==f]['linear_r2'].values[0] for f in top_features]
    
    x = np.arange(len(top_features))
    width = 0.35
    
    ax1.bar(x - width/2, real_r2, width, label='真实数据(N=103)', alpha=0.8)
    ax1.bar(x + width/2, synth_r2, width, label='合成数据(N=100k)', alpha=0.8)
    ax1.set_xlabel('特征', fontsize=11)
    ax1.set_ylabel('R² (解释力)', fontsize=11)
    ax1.set_title('主效应解释力对比（Top 8）', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_features, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 线性系数对比
    ax2 = axes[0, 1]
    real_coef = [main_effects_real[main_effects_real['feature']==f]['linear_coef'].values[0] for f in top_features]
    synth_coef = [main_effects_synthetic[main_effects_synthetic['feature']==f]['linear_coef'].values[0] for f in top_features]
    
    ax2.bar(x - width/2, real_coef, width, label='真实数据(N=103)', alpha=0.8)
    ax2.bar(x + width/2, synth_coef, width, label='合成数据(N=100k)', alpha=0.8)
    ax2.set_xlabel('特征', fontsize=11)
    ax2.set_ylabel('标准化系数', fontsize=11)
    ax2.set_title('主效应系数对比（Top 8）', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_features, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. 非线性效应（二次项系数）
    ax3 = axes[1, 0]
    nonlinear_df = main_effects_real[main_effects_real['quad2_p'] < 0.1].copy()
    nonlinear_df['abs_quad2_coef'] = nonlinear_df['quad2_coef'].abs()
    nonlinear_features = nonlinear_df.nlargest(8, 'abs_quad2_coef')['feature'].tolist()
    
    if len(nonlinear_features) > 0:
        real_quad = [main_effects_real[main_effects_real['feature']==f]['quad2_coef'].values[0] for f in nonlinear_features]
        synth_quad = [main_effects_synthetic[main_effects_synthetic['feature']==f]['quad2_coef'].values[0] for f in nonlinear_features]
        
        x_nl = np.arange(len(nonlinear_features))
        ax3.bar(x_nl - width/2, real_quad, width, label='真实数据(N=103)', alpha=0.8)
        ax3.bar(x_nl + width/2, synth_quad, width, label='合成数据(N=100k)', alpha=0.8)
        ax3.set_xlabel('特征', fontsize=11)
        ax3.set_ylabel('二次项系数', fontsize=11)
        ax3.set_title('非线性效应对比（二次项p<0.1）', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_nl)
        ax3.set_xticklabels(nonlinear_features, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    else:
        ax3.text(0.5, 0.5, '无显著非线性效应', ha='center', va='center', fontsize=14)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    # 4. 特征重要性热力图（真实数据）
    ax4 = axes[1, 1]
    importance_data = []
    for _, row in main_effects_real.iterrows():
        importance_data.append({
            '特征': row['feature'],
            '主效应': row['linear_r2'],
            '非线性': abs(row['quad2_coef']) if row['quad2_p'] < 0.1 else 0
        })
    
    df_importance = pd.DataFrame(importance_data).set_index('特征')
    df_importance = df_importance.nlargest(8, '主效应')
    
    sns.heatmap(df_importance.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': '重要性'})
    ax4.set_title('特征重要性热力图（真实数据）', fontsize=12, fontweight='bold')
    ax4.set_xlabel('')
    ax4.set_ylabel('效应类型', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("="*80)
    print("跨文化适应影响因素全面分析")
    print("="*80)
    
    # 加载数据
    df_real, df_synthetic = load_data()
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "results" / "comprehensive_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 真实数据分析 ==========
    print("\n\n" + "="*80)
    print("【真实数据分析 (N=103)】")
    print("="*80)
    
    main_effects_real = analyze_all_main_effects(df_real, "真实数据(N=103)")
    interactions_2way_real = analyze_all_2way_interactions(df_real, "真实数据(N=103)", top_n=30)
    interactions_3way_real = analyze_all_3way_interactions(df_real, "真实数据(N=103)", top_n=30)
    
    key_factors_real = identify_key_factors(main_effects_real, interactions_2way_real, interactions_3way_real)
    
    # ========== 合成数据分析 ==========
    print("\n\n" + "="*80)
    print("【合成数据分析 (N=100,000)】")
    print("="*80)
    
    main_effects_synthetic = analyze_all_main_effects(df_synthetic, "合成数据(N=100,000)")
    interactions_2way_synthetic = analyze_all_2way_interactions(df_synthetic, "合成数据(N=100,000)", top_n=30)
    interactions_3way_synthetic = analyze_all_3way_interactions(df_synthetic, "合成数据(N=100,000)", top_n=30)
    
    key_factors_synthetic = identify_key_factors(main_effects_synthetic, interactions_2way_synthetic, interactions_3way_synthetic)
    
    # ========== 生成可视化 ==========
    plot_top_factors_comparison(df_real, df_synthetic, main_effects_real, main_effects_synthetic,
                                output_dir / "factors_comparison.png")
    
    # ========== 保存结果 ==========
    main_effects_real.to_csv(output_dir / "main_effects_real.csv", index=False, encoding='utf-8-sig')
    interactions_2way_real.to_csv(output_dir / "interactions_2way_real.csv", index=False, encoding='utf-8-sig')
    interactions_3way_real.to_csv(output_dir / "interactions_3way_real.csv", index=False, encoding='utf-8-sig')
    
    main_effects_synthetic.to_csv(output_dir / "main_effects_synthetic.csv", index=False, encoding='utf-8-sig')
    interactions_2way_synthetic.to_csv(output_dir / "interactions_2way_synthetic.csv", index=False, encoding='utf-8-sig')
    interactions_3way_synthetic.to_csv(output_dir / "interactions_3way_synthetic.csv", index=False, encoding='utf-8-sig')
    
    print("\n\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n所有结果已保存到: {output_dir}")
    print("\n生成的文件:")
    print("  1. factors_comparison.png - 关键因素对比图")
    print("  2. main_effects_real.csv - 真实数据主效应")
    print("  3. interactions_2way_real.csv - 真实数据二阶交互")
    print("  4. interactions_3way_real.csv - 真实数据三阶交互")
    print("  5. main_effects_synthetic.csv - 合成数据主效应")
    print("  6. interactions_2way_synthetic.csv - 合成数据二阶交互")
    print("  7. interactions_3way_synthetic.csv - 合成数据三阶交互")

if __name__ == "__main__":
    main()
