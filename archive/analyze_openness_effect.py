#!/usr/bin/env python3
"""
开放性对跨文化适应的倒U型效应分析
分析开放性的直接效应、二阶交互、三阶交互
生成103真实样本和10万合成数据的可视化图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
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
    # 转换为中文列名
    df_synthetic = df_synthetic.rename(columns={v: k for k, v in COLUMN_MAPPING.items()})
    # 随机采样5000个样本用于可视化
    df_synthetic_sample = df_synthetic.sample(n=5000, random_state=42)
    print(f"合成数据: {len(df_synthetic)} 样本 (采样5000用于可视化)")
    
    return df_real, df_synthetic_sample, df_synthetic

def analyze_direct_effect(df, title):
    """分析开放性的直接效应（包括二次项）"""
    print(f"\n{'='*60}")
    print(f"{title} - 开放性直接效应分析")
    print(f"{'='*60}")
    
    # 标准化
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[FEATURES_CN]), columns=FEATURES_CN)
    y = df[TARGET_CN].values
    
    # 模型1：仅线性项
    X_linear = X[['开放性']].copy()
    X_linear = sm.add_constant(X_linear)
    model_linear = sm.OLS(y, X_linear).fit()
    
    # 模型2：线性+二次项
    X_quad = X[['开放性']].copy()
    X_quad['开放性²'] = X['开放性'] ** 2
    X_quad = sm.add_constant(X_quad)
    model_quad = sm.OLS(y, X_quad).fit()
    
    # 模型3：包含所有主效应
    X_full = X.copy()
    X_full = sm.add_constant(X_full)
    model_full = sm.OLS(y, X_full).fit()
    
    # 模型4：所有主效应+开放性²
    X_full_quad = X.copy()
    X_full_quad['开放性²'] = X['开放性'] ** 2
    X_full_quad = sm.add_constant(X_full_quad)
    model_full_quad = sm.OLS(y, X_full_quad).fit()
    
    print(f"\n模型1（仅开放性线性项）:")
    print(f"  R² = {model_linear.rsquared:.4f}")
    print(f"  开放性系数 = {model_linear.params['开放性']:.4f}, p = {model_linear.pvalues['开放性']:.4f}")
    
    print(f"\n模型2（开放性+开放性²）:")
    print(f"  R² = {model_quad.rsquared:.4f}")
    print(f"  开放性系数 = {model_quad.params['开放性']:.4f}, p = {model_quad.pvalues['开放性']:.4f}")
    print(f"  开放性²系数 = {model_quad.params['开放性²']:.4f}, p = {model_quad.pvalues['开放性²']:.4f}")
    
    print(f"\n模型3（所有主效应）:")
    print(f"  R² = {model_full.rsquared:.4f}")
    print(f"  开放性系数 = {model_full.params['开放性']:.4f}, p = {model_full.pvalues['开放性']:.4f}")
    
    print(f"\n模型4（所有主效应+开放性²）:")
    print(f"  R² = {model_full_quad.rsquared:.4f}")
    print(f"  开放性系数 = {model_full_quad.params['开放性']:.4f}, p = {model_full_quad.pvalues['开放性']:.4f}")
    print(f"  开放性²系数 = {model_full_quad.params['开放性²']:.4f}, p = {model_full_quad.pvalues['开放性²']:.4f}")
    
    # 计算最优开放性值
    if model_full_quad.params['开放性²'] < 0:
        optimal_openness = -model_full_quad.params['开放性'] / (2 * model_full_quad.params['开放性²'])
        print(f"\n倒U型关系确认:")
        print(f"  最优开放性（标准化）= {optimal_openness:.4f}")
        # 反标准化
        openness_mean = df['开放性'].mean()
        openness_std = df['开放性'].std()
        optimal_openness_raw = optimal_openness * openness_std + openness_mean
        print(f"  最优开放性（原始值）= {optimal_openness_raw:.2f} 分")
    
    return {
        'linear': model_linear,
        'quad': model_quad,
        'full': model_full,
        'full_quad': model_full_quad
    }

def analyze_2way_interactions(df, title):
    """分析开放性的二阶交互效应"""
    print(f"\n{'='*60}")
    print(f"{title} - 开放性二阶交互效应分析")
    print(f"{'='*60}")
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[FEATURES_CN]), columns=FEATURES_CN)
    y = df[TARGET_CN].values
    
    # 测试开放性与其他变量的交互
    interactions_to_test = [
        ('文化接触', '开放性'),
        ('家庭支持', '开放性'),
        ('社会接触', '开放性'),
        ('文化保持', '开放性'),
        ('社会联结感', '开放性'),
    ]
    
    results = []
    for f1, f2 in interactions_to_test:
        X_test = X.copy()
        X_test[f'{f1}×{f2}'] = X[f1] * X[f2]
        X_test = sm.add_constant(X_test)
        model = sm.OLS(y, X_test).fit()
        
        coef = model.params[f'{f1}×{f2}']
        pval = model.pvalues[f'{f1}×{f2}']
        r2 = model.rsquared
        
        results.append({
            'interaction': f'{f1}×{f2}',
            'coef': coef,
            'pval': pval,
            'r2': r2,
            'significant': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        })
        
        print(f"\n{f1}×{f2}:")
        print(f"  系数 = {coef:.4f}, p = {pval:.4f} {results[-1]['significant']}")
        print(f"  模型R² = {r2:.4f}")
    
    return results

def analyze_3way_interactions(df, title):
    """分析开放性的三阶交互效应"""
    print(f"\n{'='*60}")
    print(f"{title} - 开放性三阶交互效应分析")
    print(f"{'='*60}")
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[FEATURES_CN]), columns=FEATURES_CN)
    y = df[TARGET_CN].values
    
    # 测试包含开放性的三阶交互
    interactions_to_test = [
        ('沟通坦诚度', '自主权', '开放性'),
        ('沟通坦诚度', '社会联结感', '开放性'),
        ('家庭支持', '社会联结感', '开放性'),
        ('社会接触', '家庭沟通频率', '开放性'),
    ]
    
    results = []
    for f1, f2, f3 in interactions_to_test:
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
            'coef': coef,
            'pval': pval,
            'r2': r2,
            'significant': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        })
        
        print(f"\n{f1}×{f2}×{f3}:")
        print(f"  系数 = {coef:.4f}, p = {pval:.4f} {results[-1]['significant']}")
        print(f"  模型R² = {r2:.4f}")
    
    return results

def plot_u_curve(df, title, save_path):
    """绘制开放性与适应程度的倒U型曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：散点图+拟合曲线
    ax1 = axes[0]
    openness = df['开放性'].values
    adaptation = df[TARGET_CN].values
    
    # 散点图
    ax1.scatter(openness, adaptation, alpha=0.5, s=50, edgecolors='black', linewidths=0.5)
    
    # 拟合二次曲线
    z = np.polyfit(openness, adaptation, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(openness.min(), openness.max(), 100)
    y_smooth = p(x_smooth)
    ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'拟合曲线: y={z[0]:.3f}x²+{z[1]:.3f}x+{z[2]:.3f}')
    
    # 标记最优点
    if z[0] < 0:
        optimal_x = -z[1] / (2 * z[0])
        optimal_y = p(optimal_x)
        ax1.plot(optimal_x, optimal_y, 'r*', markersize=20, label=f'最优点: ({optimal_x:.2f}, {optimal_y:.2f})')
    
    ax1.set_xlabel('开放性 (1-7分)', fontsize=12)
    ax1.set_ylabel('跨文化适应程度 (8-32分)', fontsize=12)
    ax1.set_title(f'{title}\n开放性与适应程度的关系', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：分组箱线图
    ax2 = axes[1]
    # 将开放性分为5组
    df_plot = df.copy()
    df_plot['开放性分组'] = pd.cut(df_plot['开放性'], bins=5, labels=['很低(1-2)', '低(2-3)', '中(3-5)', '高(5-6)', '很高(6-7)'])
    
    # 箱线图
    df_plot.boxplot(column=TARGET_CN, by='开放性分组', ax=ax2)
    ax2.set_xlabel('开放性分组', fontsize=12)
    ax2.set_ylabel('跨文化适应程度', fontsize=12)
    ax2.set_title(f'{title}\n不同开放性水平的适应程度分布', fontsize=13, fontweight='bold')
    plt.sca(ax2)
    plt.xticks(rotation=15)
    
    # 添加均值线
    means = df_plot.groupby('开放性分组')[TARGET_CN].mean()
    ax2.plot(range(1, len(means)+1), means.values, 'ro-', linewidth=2, markersize=8, label='均值')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {save_path}")
    plt.close()

def plot_interaction_effects(df, title, save_path):
    """绘制开放性的交互效应图 - 使用简化的预测方法"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 标准化数据
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[FEATURES_CN]), columns=FEATURES_CN)
    y = df[TARGET_CN].values
    
    # 反标准化函数
    def unstandardize(col_name, std_value):
        mean = df[col_name].mean()
        std = df[col_name].std()
        return std_value * std + mean
    
    # 交互1：文化接触×开放性
    ax1 = axes[0, 0]
    openness_levels = [-1, 0, 1]  # 低、中、高
    openness_labels = ['低开放性(-1SD)', '中开放性(均值)', '高开放性(+1SD)']
    colors = ['blue', 'green', 'red']
    
    cultural_contact_range = np.linspace(X['文化接触'].min(), X['文化接触'].max(), 50)
    
    for i, (level, label, color) in enumerate(zip(openness_levels, openness_labels, colors)):
        # 构建预测模型
        X_pred = X.copy()
        X_pred['文化接触×开放性'] = X_pred['文化接触'] * X_pred['开放性']
        X_pred = sm.add_constant(X_pred)
        model = sm.OLS(y, X_pred).fit()
        
        # 使用模型系数手动计算预测值
        predictions = []
        base_values = X.mean().values  # 所有特征的均值
        
        for cc in cultural_contact_range:
            # 手动构建特征向量
            features = base_values.copy()
            features[FEATURES_CN.index('文化接触')] = cc
            features[FEATURES_CN.index('开放性')] = level
            interaction = cc * level
            
            # 手动计算预测：intercept + sum(coef * feature) + interaction_coef * interaction
            pred = model.params['const']
            for j, feat_name in enumerate(FEATURES_CN):
                pred += model.params[feat_name] * features[j]
            pred += model.params['文化接触×开放性'] * interaction
            predictions.append(pred)
        
        # 反标准化x轴
        cc_raw = [unstandardize('文化接触', x) for x in cultural_contact_range]
        ax1.plot(cc_raw, predictions, color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('文化接触 (1-7分)', fontsize=11)
    ax1.set_ylabel('预测的适应程度', fontsize=11)
    ax1.set_title('文化接触×开放性交互效应', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 交互2：社会接触×开放性
    ax2 = axes[0, 1]
    social_contact_range = np.linspace(X['社会接触'].min(), X['社会接触'].max(), 50)
    
    for i, (level, label, color) in enumerate(zip(openness_levels, openness_labels, colors)):
        X_pred = X.copy()
        X_pred['社会接触×开放性'] = X_pred['社会接触'] * X_pred['开放性']
        X_pred = sm.add_constant(X_pred)
        model = sm.OLS(y, X_pred).fit()
        
        predictions = []
        base_values = X.mean().values
        
        for sc in social_contact_range:
            features = base_values.copy()
            features[FEATURES_CN.index('社会接触')] = sc
            features[FEATURES_CN.index('开放性')] = level
            interaction = sc * level
            
            pred = model.params['const']
            for j, feat_name in enumerate(FEATURES_CN):
                pred += model.params[feat_name] * features[j]
            pred += model.params['社会接触×开放性'] * interaction
            predictions.append(pred)
        
        sc_raw = [unstandardize('社会接触', x) for x in social_contact_range]
        ax2.plot(sc_raw, predictions, color=color, linewidth=2, label=label)
    
    ax2.set_xlabel('社会接触 (1-7分)', fontsize=11)
    ax2.set_ylabel('预测的适应程度', fontsize=11)
    ax2.set_title('社会接触×开放性交互效应', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 交互3：家庭支持×开放性
    ax3 = axes[1, 0]
    family_support_range = np.linspace(X['家庭支持'].min(), X['家庭支持'].max(), 50)
    
    for i, (level, label, color) in enumerate(zip(openness_levels, openness_labels, colors)):
        X_pred = X.copy()
        X_pred['家庭支持×开放性'] = X_pred['家庭支持'] * X_pred['开放性']
        X_pred = sm.add_constant(X_pred)
        model = sm.OLS(y, X_pred).fit()
        
        predictions = []
        base_values = X.mean().values
        
        for fs in family_support_range:
            features = base_values.copy()
            features[FEATURES_CN.index('家庭支持')] = fs
            features[FEATURES_CN.index('开放性')] = level
            interaction = fs * level
            
            pred = model.params['const']
            for j, feat_name in enumerate(FEATURES_CN):
                pred += model.params[feat_name] * features[j]
            pred += model.params['家庭支持×开放性'] * interaction
            predictions.append(pred)
        
        fs_raw = [unstandardize('家庭支持', x) for x in family_support_range]
        ax3.plot(fs_raw, predictions, color=color, linewidth=2, label=label)
    
    ax3.set_xlabel('家庭支持 (8-40分)', fontsize=11)
    ax3.set_ylabel('预测的适应程度', fontsize=11)
    ax3.set_title('家庭支持×开放性交互效应', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 交互4：开放性的边际效应（在不同文化接触水平下）
    ax4 = axes[1, 1]
    cultural_contact_levels = [-1, 0, 1]
    cc_labels = ['低文化接触(-1SD)', '中文化接触(均值)', '高文化接触(+1SD)']
    
    openness_range = np.linspace(X['开放性'].min(), X['开放性'].max(), 50)
    
    for i, (level, label, color) in enumerate(zip(cultural_contact_levels, cc_labels, colors)):
        X_pred = X.copy()
        X_pred['文化接触×开放性'] = X_pred['文化接触'] * X_pred['开放性']
        X_pred = sm.add_constant(X_pred)
        model = sm.OLS(y, X_pred).fit()
        
        predictions = []
        base_values = X.mean().values
        
        for op in openness_range:
            features = base_values.copy()
            features[FEATURES_CN.index('开放性')] = op
            features[FEATURES_CN.index('文化接触')] = level
            interaction = op * level
            
            pred = model.params['const']
            for j, feat_name in enumerate(FEATURES_CN):
                pred += model.params[feat_name] * features[j]
            pred += model.params['文化接触×开放性'] * interaction
            predictions.append(pred)
        
        op_raw = [unstandardize('开放性', x) for x in openness_range]
        ax4.plot(op_raw, predictions, color=color, linewidth=2, label=label)
    
    ax4.set_xlabel('开放性 (1-7分)', fontsize=11)
    ax4.set_ylabel('预测的适应程度', fontsize=11)
    ax4.set_title('开放性效应在不同文化接触水平下的变化', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - 开放性交互效应可视化', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("="*80)
    print("开放性对跨文化适应的倒U型效应分析")
    print("="*80)
    
    # 加载数据
    df_real, df_synthetic_sample, df_synthetic_full = load_data()
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "results" / "openness_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析真实数据
    print("\n\n" + "="*80)
    print("【真实数据分析 (N=103)】")
    print("="*80)
    
    models_real = analyze_direct_effect(df_real, "真实数据(N=103)")
    interactions_2way_real = analyze_2way_interactions(df_real, "真实数据(N=103)")
    interactions_3way_real = analyze_3way_interactions(df_real, "真实数据(N=103)")
    
    # 绘制真实数据图表
    plot_u_curve(df_real, "真实数据(N=103)", output_dir / "openness_u_curve_real.png")
    plot_interaction_effects(df_real, "真实数据(N=103)", output_dir / "openness_interactions_real.png")
    
    # 分析合成数据
    print("\n\n" + "="*80)
    print("【合成数据分析 (N=100,000, 采样5000用于可视化)】")
    print("="*80)
    
    models_synthetic = analyze_direct_effect(df_synthetic_full, "合成数据(N=100,000)")
    interactions_2way_synthetic = analyze_2way_interactions(df_synthetic_full, "合成数据(N=100,000)")
    interactions_3way_synthetic = analyze_3way_interactions(df_synthetic_full, "合成数据(N=100,000)")
    
    # 绘制合成数据图表（使用采样数据）
    plot_u_curve(df_synthetic_sample, "合成数据(N=5000采样)", output_dir / "openness_u_curve_synthetic.png")
    plot_interaction_effects(df_synthetic_sample, "合成数据(N=5000采样)", output_dir / "openness_interactions_synthetic.png")
    
    # 生成总结报告
    print("\n\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n所有图表已保存到: {output_dir}")
    print("\n生成的文件:")
    print("  1. openness_u_curve_real.png - 真实数据的倒U型曲线")
    print("  2. openness_interactions_real.png - 真实数据的交互效应")
    print("  3. openness_u_curve_synthetic.png - 合成数据的倒U型曲线")
    print("  4. openness_interactions_synthetic.png - 合成数据的交互效应")

if __name__ == "__main__":
    main()
