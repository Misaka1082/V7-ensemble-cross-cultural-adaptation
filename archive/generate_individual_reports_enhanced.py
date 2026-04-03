#!/usr/bin/env python3
"""
103个样本的个性化心理学解释报告生成器（增强版）
=========================================
为每个样本生成：
1. 个人适应画像（雷达图）
2. SHAP瀑布图（促进/阻碍因素）
3. 心理学理论解释（通俗语言）
4. 数学证据（SHAP值、百分位、相关性）
5. 个性化建议
6. **新增：详细文字分析（包含各因素分析表格和建议）**

输出：
- 每个样本一页的PDF报告（合并为一个文件，包含图表+文字分析）
- 汇总Excel表格（含所有样本的SHAP值和解释）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import shap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10

# ============================================================
# 配置
# ============================================================
FEATURES_CN = ['文化保持', '社会保持', '文化接触', '社会接触', '家庭支持',
               '家庭沟通频率', '沟通坦诚度', '自主权', '社会联结感', '开放性', '来港时长']
TARGET_CN = '跨文化适应程度'

FEATURE_RANGES = {
    '文化保持': (1, 7), '社会保持': (1, 7), '文化接触': (1, 7), '社会接触': (1, 7),
    '家庭支持': (8, 40), '家庭沟通频率': (1, 5), '沟通坦诚度': (1, 5),
    '自主权': (1, 5), '社会联结感': (5, 30), '开放性': (1, 7), '来港时长': (2, 348)
}

# 特征的心理学含义（通俗版）
FEATURE_MEANINGS = {
    '文化保持': {
        'what': '您保持内地文化习惯和认同的程度',
        'high': '您非常重视保持自己的文化根源，这是心理稳定的重要基础',
        'low': '您对原有文化认同感较弱，可能正在经历文化身份的重新定位',
        'theory': 'Berry文化适应理论',
    },
    '社会保持': {
        'what': '您维持内地社交关系的程度',
        'high': '您与内地亲友保持密切联系，这提供了重要的情感支持网络',
        'low': '您与内地社交圈的联系较少，可能需要在香港建立新的支持网络',
        'theory': '社会支持理论',
    },
    '文化接触': {
        'what': '您接触和参与香港本地文化的频率',
        'high': '您积极融入香港文化，这是跨文化适应最重要的驱动力',
        'low': '您与香港文化的接触较少，这是影响适应的最关键因素',
        'theory': '接触假说（Allport, 1954）',
    },
    '社会接触': {
        'what': '您与香港本地人社交互动的频率',
        'high': '您与本地人有较多互动，有助于建立本地社会资本和归属感',
        'low': '您与本地人的互动较少，建议主动创造跨文化社交机会',
        'theory': '社会资本理论',
    },
    '家庭支持': {
        'what': '您感受到的来自家庭的情感和实际支持程度',
        'high': '您拥有强大的家庭支持，这是应对适应压力的重要缓冲资源',
        'low': '您感受到的家庭支持较少，建议主动与家人沟通，寻求更多支持',
        'theory': '压力-缓冲模型（Cohen & Wills, 1985）',
    },
    '家庭沟通频率': {
        'what': '您与家人沟通的频率',
        'high': '您与家人保持频繁联系，维持了重要的情感纽带',
        'low': '您与家人的沟通较少，适当增加联系有助于获得情感支持',
        'theory': '依恋理论（Bowlby, 1969）',
    },
    '沟通坦诚度': {
        'what': '您与家人沟通时的坦诚和开放程度',
        'high': '您与家人的沟通质量高，能有效传递情感需求和获得支持',
        'low': '您与家人的沟通较为表面，深化沟通质量有助于获得更有效的支持',
        'theory': '家庭系统理论',
    },
    '自主权': {
        'what': '您在生活中自主决策的程度',
        'high': '您拥有较高的自主权，能主动选择适合自己的适应策略',
        'low': '您的自主决策空间较小，增强自主性有助于主动应对适应挑战',
        'theory': '自我决定理论（Deci & Ryan, 1985）',
    },
    '社会联结感': {
        'what': '您对香港社会的归属感和连接感',
        'high': '您对香港社会有较强的归属感，这是适应成功的重要标志',
        'low': '您对香港社会的归属感较弱，这是需要重点关注和提升的领域',
        'theory': '社会认同理论（Tajfel & Turner, 1979）',
    },
    '开放性': {
        'what': '您对新文化、新经历的开放和接受程度',
        'high': '您具有高度开放性，这使您能更有效地从文化接触中获益',
        'low': '您的开放性相对较低，这可能限制了文化接触和社会互动的效果',
        'theory': 'Big Five人格理论（McCrae & Costa, 1987）',
    },
    '来港时长': {
        'what': '您在香港居住的时间长度',
        'high': '您在港时间较长，通常已度过文化冲击期，进入稳定适应阶段',
        'low': '您来港时间较短，可能仍处于文化适应的早期阶段',
        'theory': 'U型曲线适应假说（Lysgaard, 1955）',
    },
}

# 适应程度的心理学解读
def get_adaptation_interpretation(score, percentile):
    if score >= 28:
        return {
            'level': '优秀适应',
            'color': '#27ae60',
            'emoji': '🌟',
            'description': '您的跨文化适应程度处于优秀水平，已成功建立双文化认同（Bicultural Identity）。您能够在保持内地文化根源的同时，积极融入香港社会，这是Berry（1997）所描述的最优"整合策略"的体现。',
            'theory': '整合策略（Integration Strategy）',
        }
    elif score >= 25:
        return {
            'level': '良好适应',
            'color': '#2ecc71',
            'emoji': '✅',
            'description': '您的跨文化适应程度良好，已基本完成文化适应的核心任务。您在文化认同和社会融合方面取得了较好的平衡，但仍有进一步提升的空间。',
            'theory': '适应整合阶段',
        }
    elif score >= 22:
        return {
            'level': '中等适应',
            'color': '#f39c12',
            'emoji': '⚡',
            'description': '您的跨文化适应程度处于中等水平，可能正处于文化适应的过渡阶段。根据U型曲线假说，这一阶段是从文化冲击期向适应期过渡的关键时期，需要有针对性的支持。',
            'theory': 'U型曲线过渡期',
        }
    elif score >= 18:
        return {
            'level': '适应挑战',
            'color': '#e67e22',
            'emoji': '⚠️',
            'description': '您目前面临一定的跨文化适应挑战，可能正在经历文化冲击（Culture Shock）的某些症状。这是跨文化适应过程中的正常阶段，通过有针对性的支持和干预，可以有效改善适应状况。',
            'theory': '文化冲击理论（Oberg, 1960）',
        }
    else:
        return {
            'level': '需要支持',
            'color': '#e74c3c',
            'emoji': '🆘',
            'description': '您的跨文化适应程度较低，可能面临较大的适应压力。根据压力-应对模型，当前的适应资源可能不足以应对文化转变带来的挑战，建议寻求专业支持。',
            'theory': '压力-应对模型（Lazarus & Folkman, 1984）',
        }


def get_feature_level(feat, val):
    """获取特征水平描述"""
    feat_range = FEATURE_RANGES[feat]
    pct = (val - feat_range[0]) / (feat_range[1] - feat_range[0]) * 100
    if pct >= 66:
        return '高', pct, '#27ae60'
    elif pct >= 33:
        return '中', pct, '#f39c12'
    else:
        return '低', pct, '#e74c3c'


def generate_individual_report_page(sample_idx, df, X, y, shap_values, model,
                                     clusters, cluster_names, all_percentiles):
    """为单个样本生成报告页面（包含图表和详细文字分析）"""
    sample = df.iloc[sample_idx]
    x_sample = X[sample_idx]
    y_sample = y[sample_idx]
    shap_sample = shap_values[sample_idx]
    pred_sample = model.predict(X[sample_idx:sample_idx+1])[0]

    # 适应程度解读
    adapt_pct = stats.percentileofscore(y, y_sample)
    adapt_info = get_adaptation_interpretation(y_sample, adapt_pct)

    # 聚类信息
    cluster_id = clusters[sample_idx]
    cluster_name = cluster_names[cluster_id]

    # 创建图形 - 增加高度以容纳文字分析
    fig = plt.figure(figsize=(20, 36))
    fig.patch.set_facecolor('#f8f9fa')

    # 标题区域
    ax_title = fig.add_axes([0.02, 0.96, 0.96, 0.03])
    ax_title.set_facecolor(adapt_info['color'])
    ax_title.text(0.5, 0.5,
                 f"样本 #{sample_idx+1}  |  跨文化适应个性化分析报告  |  {adapt_info['emoji']} {adapt_info['level']}",
                 ha='center', va='center', fontsize=16, fontweight='bold', color='white',
                 transform=ax_title.transAxes)
    ax_title.axis('off')

    # ---- 区域1：基本信息 ----
    ax_info = fig.add_axes([0.02, 0.91, 0.45, 0.04])
    ax_info.set_facecolor('white')
    ax_info.patch.set_alpha(0.9)
    info_text = (
        f"跨文化适应程度：{y_sample:.0f} 分  （满分32分，高于 {adapt_pct:.0f}% 的参与者）\n"
        f"模型预测：{pred_sample:.1f} 分  |  预测误差：{abs(y_sample - pred_sample):.1f} 分\n"
        f"适应类型：{cluster_name}  |  理论框架：{adapt_info['theory']}"
    )
    ax_info.text(0.03, 0.5, info_text, va='center', fontsize=10,
                transform=ax_info.transAxes, linespacing=1.6)
    ax_info.axis('off')

    # ---- 区域2：适应程度解读 ----
    ax_desc = fig.add_axes([0.49, 0.91, 0.49, 0.04])
    ax_desc.set_facecolor('#eaf4fb')
    ax_desc.patch.set_alpha(0.9)
    ax_desc.text(0.03, 0.85, "📖 心理学解读", fontsize=9, fontweight='bold',
                transform=ax_desc.transAxes, color='#2c3e50')
    ax_desc.text(0.03, 0.15, adapt_info['description'],
                va='bottom', fontsize=8, transform=ax_desc.transAxes,
                wrap=True, color='#2c3e50', linespacing=1.4)
    ax_desc.axis('off')

    # ---- 区域3：雷达图（个人特征画像）----
    ax_radar = fig.add_axes([0.02, 0.73, 0.38, 0.17], projection='polar')
    radar_feats = ['文化接触', '社会接触', '家庭支持', '社会联结感', '开放性',
                   '自主权', '文化保持', '社会保持']
    n_feats = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]

    # 个人值（归一化）
    personal_vals = []
    group_vals = []
    for feat in radar_feats:
        feat_idx = FEATURES_CN.index(feat)
        feat_range = FEATURE_RANGES[feat]
        personal_vals.append((x_sample[feat_idx] - feat_range[0]) / (feat_range[1] - feat_range[0]))
        group_vals.append((X[:, feat_idx].mean() - feat_range[0]) / (feat_range[1] - feat_range[0]))
    personal_vals += personal_vals[:1]
    group_vals += group_vals[:1]

    ax_radar.plot(angles, personal_vals, color=adapt_info['color'], linewidth=2.5,
                 label='您的得分', zorder=3)
    ax_radar.fill(angles, personal_vals, color=adapt_info['color'], alpha=0.25)
    ax_radar.plot(angles, group_vals, color='#95a5a6', linewidth=1.5,
                 linestyle='--', label='群体平均', zorder=2)
    ax_radar.fill(angles, group_vals, color='#95a5a6', alpha=0.1)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_feats, fontsize=8)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75])
    ax_radar.set_yticklabels(['低', '中', '高'], fontsize=7)
    ax_radar.set_title('个人特征画像\n（实线=您，虚线=群体均值）', fontsize=9, fontweight='bold', pad=12)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.12), fontsize=7)

    # ---- 区域4：SHAP瀑布图 ----
    ax_shap = fig.add_axes([0.42, 0.73, 0.56, 0.17])
    ax_shap.set_facecolor('white')

    sorted_idx = np.argsort(np.abs(shap_sample))[::-1]
    top_n = 8
    top_idx = sorted_idx[:top_n]
    top_feats = [FEATURES_CN[i] for i in top_idx]
    top_shap = shap_sample[top_idx]
    top_x = x_sample[top_idx]

    colors = ['#e74c3c' if v > 0 else '#3498db' for v in top_shap]
    y_pos = range(top_n - 1, -1, -1)

    bars = ax_shap.barh(list(y_pos), top_shap, color=colors, alpha=0.85, height=0.6,
                        edgecolor='white', linewidth=0.5)

    # 添加数值标签
    for i, (pos, val, feat, x_val) in enumerate(zip(y_pos, top_shap, top_feats, top_x)):
        sign = '+' if val > 0 else ''
        ax_shap.text(val + (0.02 if val > 0 else -0.02), pos,
                    f'{sign}{val:.2f}分', va='center',
                    ha='left' if val > 0 else 'right', fontsize=8, fontweight='bold',
                    color='#2c3e50')

    ax_shap.set_yticks(list(y_pos))
    ax_shap.set_yticklabels([f'{f}（{x:.1f}）' for f, x in zip(top_feats, top_x)],
                            fontsize=8)
    ax_shap.axvline(x=0, color='black', linewidth=1.5)
    ax_shap.set_xlabel('对您适应程度的影响（分）', fontsize=9)
    ax_shap.set_title(f'影响您适应程度的关键因素\n（红色=促进，蓝色=阻碍，基准={y.mean():.1f}分）',
                     fontsize=9, fontweight='bold')
    ax_shap.grid(True, alpha=0.3, axis='x')
    ax_shap.set_facecolor('#fafafa')

    # ============================================================
    # 新增：详细文字分析区域
    # ============================================================
    ax_text = fig.add_axes([0.02, 0.02, 0.96, 0.70])
    ax_text.axis('off')
    
    # 标题
    ax_text.text(0.5, 0.99, '📊 各因素详细分析与个性化建议',
                ha='center', va='top', fontsize=12, fontweight='bold',
                transform=ax_text.transAxes, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
    
    # 按SHAP绝对值排序
    sorted_all = np.argsort(np.abs(shap_sample))[::-1]
    
    # 生成因素分析表格
    y_pos = 0.94
    line_height = 0.055
    
    # 表头
    ax_text.text(0.02, y_pos, '因素', fontsize=9, fontweight='bold', transform=ax_text.transAxes)
    ax_text.text(0.15, y_pos, '您的得分', fontsize=9, fontweight='bold', transform=ax_text.transAxes)
    ax_text.text(0.25, y_pos, '百分位', fontsize=9, fontweight='bold', transform=ax_text.transAxes)
    ax_text.text(0.35, y_pos, '影响', fontsize=9, fontweight='bold', transform=ax_text.transAxes)
    ax_text.text(0.45, y_pos, '心理学解读', fontsize=9, fontweight='bold', transform=ax_text.transAxes)
    
    # 绘制表头下划线
    ax_text.plot([0.02, 0.98], [y_pos - 0.01, y_pos - 0.01], 'k-', linewidth=1, transform=ax_text.transAxes)
    
    y_pos -= 0.025
    
    # 显示前11个因素
    for idx, feat_idx in enumerate(sorted_all[:11]):
        feat = FEATURES_CN[feat_idx]
        val = x_sample[feat_idx]
        shap_val = shap_sample[feat_idx]
        pct = stats.percentileofscore(X[:, feat_idx], val)
        level, _, _ = get_feature_level(feat, val)
        
        # 获取心理学解读
        meanings = FEATURE_MEANINGS.get(feat, {})
        if level == '高':
            interp = meanings.get('high', '')
        elif level == '低':
            interp = meanings.get('low', '')
        else:
            interp = meanings.get('what', '')
        
        # 截断文字
        if len(interp) > 45:
            interp = interp[:45] + '...'
        
        # 行背景色
        if idx % 2 == 0:
            rect = mpatches.Rectangle((0.01, y_pos - 0.045), 0.98, 0.05,
                                     facecolor='#f8f9fa', transform=ax_text.transAxes, zorder=1)
            ax_text.add_patch(rect)
        
        # 因素名称（带emoji）
        emoji = '🔴' if shap_val > 0 else '🔵'
        ax_text.text(0.02, y_pos, f'{emoji} {feat}', fontsize=8, transform=ax_text.transAxes, va='top')
        
        # 得分
        ax_text.text(0.15, y_pos, f'{val:.1f}（{level}）', fontsize=8, transform=ax_text.transAxes, va='top')
        
        # 百分位
        ax_text.text(0.25, y_pos, f'{pct:.0f}%', fontsize=8, transform=ax_text.transAxes, va='top')
        
        # 影响
        sign = '+' if shap_val > 0 else ''
        color = '#c0392b' if shap_val > 0 else '#2980b9'
        ax_text.text(0.35, y_pos, f'{sign}{shap_val:.2f}分', fontsize=8, 
                    transform=ax_text.transAxes, va='top', color=color, fontweight='bold')
        
        # 心理学解读
        ax_text.text(0.45, y_pos, interp, fontsize=7.5, transform=ax_text.transAxes, va='top',
                    style='italic', color='#34495e')
        
        y_pos -= line_height
    
    # 个性化建议区域
    y_pos -= 0.03
    ax_text.text(0.5, y_pos, '💡 个性化建议',
                ha='center', va='top', fontsize=11, fontweight='bold',
                transform=ax_text.transAxes, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', edgecolor='#ffc107'))
    
    y_pos -= 0.04
    
    # 优势因素
    neg_feats = [(FEATURES_CN[j], shap_sample[j]) for j in np.argsort(shap_sample)[:3] if shap_sample[j] < 0]
    pos_feats = [(FEATURES_CN[j], shap_sample[j]) for j in np.argsort(shap_sample)[::-1][:3] if shap_sample[j] > 0]
    
    if pos_feats:
        ax_text.text(0.02, y_pos, '✅ 您的优势因素（请继续保持）：',
                    fontsize=9, fontweight='bold', transform=ax_text.transAxes, va='top', color='#27ae60')
        y_pos -= 0.025
        
        for feat, val in pos_feats[:3]:
            meanings = FEATURE_MEANINGS.get(feat, {})
            ax_text.text(0.04, y_pos, f'• {feat}（贡献+{val:.2f}分）：{meanings.get("high", "")}',
                        fontsize=8, transform=ax_text.transAxes, va='top', color='#2c3e50')
            y_pos -= 0.025
    
    y_pos -= 0.01
    
    # 改进建议
    if neg_feats:
        ax_text.text(0.02, y_pos, '📈 建议重点提升的领域：',
                    fontsize=9, fontweight='bold', transform=ax_text.transAxes, va='top', color='#e67e22')
        y_pos -= 0.025
        
        for feat, val in neg_feats[:3]:
            meanings = FEATURE_MEANINGS.get(feat, {})
            theory = meanings.get('theory', '')
            ax_text.text(0.04, y_pos, f'• {feat}（影响{val:.2f}分）：{meanings.get("low", "")}',
                        fontsize=8, transform=ax_text.transAxes, va='top', color='#2c3e50')
            y_pos -= 0.02
            ax_text.text(0.06, y_pos, f'理论依据：{theory}',
                        fontsize=7, transform=ax_text.transAxes, va='top', color='#7f8c8d', style='italic')
            y_pos -= 0.025

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    print("=" * 80)
    print("103个样本个性化心理学解释报告生成器（增强版）")
    print("=" * 80)

    # 加载数据
    df = pd.read_excel(Path(__file__).parent / 'data/processed/real_data_103.xlsx')
    X = df[FEATURES_CN].values.astype(float)
    y = df[TARGET_CN].values.astype(float)
    print(f"数据: {len(df)} 样本, {len(FEATURES_CN)} 特征")

    # 训练XGBoost
    print("\n训练XGBoost模型...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, n_jobs=-1
    )
    model.fit(X, y)
    print(f"  训练R²: {model.score(X, y):.4f}")

    # 计算SHAP值
    print("计算SHAP值...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 聚类
    print("聚类分析...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)

    # 命名聚类
    cluster_means = {c: y[clusters == c].mean() for c in range(4)}
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
    cluster_labels = ['高适应型', '中高适应型', '中低适应型', '低适应型']
    cluster_names = {c: cluster_labels[i] for i, (c, _) in enumerate(sorted_clusters)}

    # 计算百分位
    all_percentiles = {}
    for i, feat in enumerate(FEATURES_CN):
        all_percentiles[feat] = [stats.percentileofscore(X[:, i], v) for v in X[:, i]]

    # 生成输出目录
    out_dir = Path(__file__).parent / 'results/individual_reports_enhanced'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 生成汇总Excel ----
    print("\n生成汇总Excel...")
    summary_rows = []
    for i in range(len(df)):
        row = {
            '样本编号': i + 1,
            '跨文化适应程度': y[i],
            '模型预测': round(model.predict(X[i:i+1])[0], 2),
            '预测误差': round(abs(y[i] - model.predict(X[i:i+1])[0]), 2),
            '适应类型': cluster_names[clusters[i]],
            '适应百分位': round(stats.percentileofscore(y, y[i]), 1),
        }
        for j, feat in enumerate(FEATURES_CN):
            row[f'{feat}_值'] = X[i, j]
            row[f'{feat}_百分位'] = round(stats.percentileofscore(X[:, j], X[i, j]), 1)
            row[f'{feat}_SHAP'] = round(shap_values[i, j], 3)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(out_dir / 'all_samples_summary.xlsx', index=False)
    print(f"  汇总Excel已保存: {out_dir / 'all_samples_summary.xlsx'}")

    # ---- 生成每个样本的报告页（包含图表+文字分析）----
    print("\n生成增强版个性化报告（图表+文字分析）...")
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = out_dir / 'individual_reports_enhanced_all_103.pdf'
    with PdfPages(pdf_path) as pdf:
        # 封面
        fig_cover = plt.figure(figsize=(20, 36))
        fig_cover.patch.set_facecolor('#2c3e50')
        ax_cover = fig_cover.add_axes([0.1, 0.3, 0.8, 0.4])
        ax_cover.set_facecolor('#2c3e50')
        ax_cover.text(0.5, 0.8, '跨文化适应研究', ha='center', va='center',
                     fontsize=28, fontweight='bold', color='white',
                     transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.65, '103个样本个性化心理学分析报告（增强版）',
                     ha='center', va='center', fontsize=20, color='#ecf0f1',
                     transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.5, '基于机器学习（XGBoost + SHAP）与心理学理论\n包含图表可视化与详细文字分析',
                     ha='center', va='center', fontsize=14, color='#bdc3c7',
                     transform=ax_cover.transAxes, linespacing=1.8)
        ax_cover.text(0.5, 0.35,
                     '理论框架：Berry双文化适应理论 | 社会认同理论 | 自我决定理论\n'
                     '接触假说 | 压力-缓冲模型 | U型曲线假说',
                     ha='center', va='center', fontsize=12, color='#95a5a6',
                     transform=ax_cover.transAxes, linespacing=1.8)
        ax_cover.text(0.5, 0.15, f'样本量：103  |  模型5折CV R²=0.728  |  SHAP解释性分析',
                     ha='center', va='center', fontsize=11, color='#7f8c8d',
                     transform=ax_cover.transAxes)
        ax_cover.axis('off')
        pdf.savefig(fig_cover, bbox_inches='tight')
        plt.close(fig_cover)

        # 每个样本的报告页
        for i in range(len(df)):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  生成样本 {i+1}/{len(df)}...")
            try:
                fig = generate_individual_report_page(
                    i, df, X, y, shap_values, model,
                    clusters, cluster_names, all_percentiles
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"  样本{i+1}生成失败: {e}")
                plt.close('all')

    print(f"\n✅ 增强版PDF报告已保存: {pdf_path}")
    print(f"   共 {len(df) + 1} 页（1封面 + {len(df)}个样本）")
    print(f"   每页包含：图表可视化 + 详细文字分析表格 + 个性化建议")

    print("\n" + "=" * 80)
    print("完成！生成的文件：")
    for fp in out_dir.iterdir():
        size_kb = fp.stat().st_size / 1024
        print(f"  {fp.name}  ({size_kb:.0f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
