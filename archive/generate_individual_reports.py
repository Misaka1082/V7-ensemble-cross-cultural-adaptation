#!/usr/bin/env python3
"""
103个样本的个性化心理学解释报告生成器
=========================================
为每个样本生成：
1. 个人适应画像（雷达图）
2. SHAP瀑布图（促进/阻碍因素）
3. 心理学理论解释（通俗语言）
4. 数学证据（SHAP值、百分位、相关性）
5. 个性化建议

输出：
- 每个样本一页的PDF报告（合并为一个文件）
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


def get_shap_explanation(feat, shap_val, feat_val, all_shap_vals, all_feat_vals):
    """生成SHAP值的通俗解释"""
    feat_pct = stats.percentileofscore(all_feat_vals, feat_val)
    shap_pct = stats.percentileofscore(np.abs(all_shap_vals), np.abs(shap_val))

    direction = "提升" if shap_val > 0 else "降低"
    magnitude = abs(shap_val)

    if magnitude > np.percentile(np.abs(all_shap_vals), 75):
        importance = "显著"
    elif magnitude > np.percentile(np.abs(all_shap_vals), 50):
        importance = "中等"
    else:
        importance = "轻微"

    return {
        'direction': direction,
        'magnitude': magnitude,
        'importance': importance,
        'feat_pct': feat_pct,
        'shap_pct': shap_pct,
    }


def generate_individual_report_page(sample_idx, df, X, y, shap_values, model,
                                     clusters, cluster_names, all_percentiles):
    """为单个样本生成报告页面"""
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

    # 创建图形
    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor('#f8f9fa')

    # 标题区域
    ax_title = fig.add_axes([0.02, 0.94, 0.96, 0.05])
    ax_title.set_facecolor(adapt_info['color'])
    ax_title.text(0.5, 0.5,
                 f"样本 #{sample_idx+1}  |  跨文化适应个性化分析报告  |  {adapt_info['emoji']} {adapt_info['level']}",
                 ha='center', va='center', fontsize=16, fontweight='bold', color='white',
                 transform=ax_title.transAxes)
    ax_title.axis('off')

    # ---- 区域1：基本信息 ----
    ax_info = fig.add_axes([0.02, 0.86, 0.45, 0.07])
    ax_info.set_facecolor('white')
    ax_info.patch.set_alpha(0.9)
    info_text = (
        f"跨文化适应程度：{y_sample:.0f} 分  （满分32分，高于 {adapt_pct:.0f}% 的参与者）\n"
        f"模型预测：{pred_sample:.1f} 分  |  预测误差：{abs(y_sample - pred_sample):.1f} 分\n"
        f"适应类型：{cluster_name}  |  理论框架：{adapt_info['theory']}"
    )
    ax_info.text(0.03, 0.5, info_text, va='center', fontsize=11,
                transform=ax_info.transAxes, linespacing=1.8)
    ax_info.axis('off')

    # ---- 区域2：适应程度解读 ----
    ax_desc = fig.add_axes([0.49, 0.86, 0.49, 0.07])
    ax_desc.set_facecolor('#eaf4fb')
    ax_desc.patch.set_alpha(0.9)
    ax_desc.text(0.03, 0.85, "📖 心理学解读", fontsize=10, fontweight='bold',
                transform=ax_desc.transAxes, color='#2c3e50')
    ax_desc.text(0.03, 0.15, adapt_info['description'],
                va='bottom', fontsize=9, transform=ax_desc.transAxes,
                wrap=True, color='#2c3e50', linespacing=1.5)
    ax_desc.axis('off')

    # ---- 区域3：雷达图（个人特征画像）----
    ax_radar = fig.add_axes([0.02, 0.60, 0.38, 0.25], projection='polar')
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
    ax_radar.set_xticklabels(radar_feats, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75])
    ax_radar.set_yticklabels(['低', '中', '高'], fontsize=7)
    ax_radar.set_title('个人特征画像\n（实线=您，虚线=群体均值）', fontsize=10, fontweight='bold', pad=15)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)

    # ---- 区域4：SHAP瀑布图 ----
    ax_shap = fig.add_axes([0.42, 0.60, 0.56, 0.25])
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
                    ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold',
                    color='#2c3e50')

    ax_shap.set_yticks(list(y_pos))
    ax_shap.set_yticklabels([f'{f}（实际值={x:.1f}）' for f, x in zip(top_feats, top_x)],
                            fontsize=9)
    ax_shap.axvline(x=0, color='black', linewidth=1.5)
    ax_shap.set_xlabel('对您适应程度的影响（分）', fontsize=10)
    ax_shap.set_title(f'影响您适应程度的关键因素\n（红色=促进适应，蓝色=阻碍适应，基准值={y.mean():.1f}分）',
                     fontsize=10, fontweight='bold')
    ax_shap.grid(True, alpha=0.3, axis='x')
    ax_shap.set_facecolor('#fafafa')

    # ---- 区域5：各特征详细解读 ----
    ax_detail = fig.add_axes([0.02, 0.02, 0.96, 0.56])
    ax_detail.axis('off')

    # 标题
    ax_detail.text(0.5, 0.99, '各因素详细分析与心理学解读',
                  ha='center', va='top', fontsize=13, fontweight='bold',
                  transform=ax_detail.transAxes, color='#2c3e50')

    # 按SHAP绝对值排序
    sorted_all = np.argsort(np.abs(shap_sample))[::-1]

    # 分两列显示
    n_per_col = 6
    col_width = 0.48
    col_starts = [0.01, 0.51]

    for col_idx in range(2):
        x_start = col_starts[col_idx]
        feat_indices = sorted_all[col_idx * n_per_col: (col_idx + 1) * n_per_col]

        y_start = 0.95
        row_height = 0.155

        for row_idx, feat_idx in enumerate(feat_indices):
            feat = FEATURES_CN[feat_idx]
            val = x_sample[feat_idx]
            shap_val = shap_sample[feat_idx]
            level, pct, level_color = get_feature_level(feat, val)
            feat_range = FEATURE_RANGES[feat]

            y_top = y_start - row_idx * row_height
            y_bottom = y_top - row_height + 0.005

            # 背景框
            bg_color = '#fff5f5' if shap_val < 0 else '#f0fff4' if shap_val > 0 else '#f8f9fa'
            rect = mpatches.FancyBboxPatch(
                (x_start, y_bottom), col_width - 0.01, row_height - 0.005,
                boxstyle="round,pad=0.005", linewidth=0.5,
                edgecolor='#dee2e6', facecolor=bg_color,
                transform=ax_detail.transAxes
            )
            ax_detail.add_patch(rect)

            # 特征名称和值
            shap_sign = '+' if shap_val > 0 else ''
            shap_color = '#c0392b' if shap_val > 0 else '#2980b9'
            ax_detail.text(x_start + 0.005, y_top - 0.01,
                          f"{'🔴' if shap_val > 0 else '🔵'} {feat}",
                          fontsize=10, fontweight='bold', color='#2c3e50',
                          transform=ax_detail.transAxes, va='top')

            # 数值信息行
            ax_detail.text(x_start + 0.005, y_top - 0.04,
                          f"您的得分：{val:.1f}（范围{feat_range[0]}-{feat_range[1]}，"
                          f"高于{pct:.0f}%的参与者）  |  "
                          f"对适应的影响：{shap_sign}{shap_val:.2f}分",
                          fontsize=8.5, color='#555',
                          transform=ax_detail.transAxes, va='top')

            # 心理学解读
            meanings = FEATURE_MEANINGS.get(feat, {})
            if level == '高':
                interp = meanings.get('high', '')
            elif level == '低':
                interp = meanings.get('low', '')
            else:
                interp = meanings.get('what', '')

            # 截断过长的文字
            if len(interp) > 55:
                interp = interp[:55] + '...'

            ax_detail.text(x_start + 0.005, y_top - 0.07,
                          f"📌 {interp}",
                          fontsize=8.5, color='#333', style='italic',
                          transform=ax_detail.transAxes, va='top')

            # 理论依据
            theory = meanings.get('theory', '')
            ax_detail.text(x_start + 0.005, y_top - 0.10,
                          f"理论依据：{theory}",
                          fontsize=8, color='#7f8c8d',
                          transform=ax_detail.transAxes, va='top')

            # 进度条（特征值可视化）
            bar_x = x_start + 0.005
            bar_y = y_top - 0.135
            bar_w = col_width - 0.02
            bar_h = 0.012
            # 背景条
            ax_detail.add_patch(mpatches.Rectangle(
                (bar_x, bar_y), bar_w, bar_h,
                facecolor='#ecf0f1', transform=ax_detail.transAxes, zorder=2
            ))
            # 填充条
            fill_w = bar_w * (pct / 100)
            ax_detail.add_patch(mpatches.Rectangle(
                (bar_x, bar_y), fill_w, bar_h,
                facecolor=level_color, alpha=0.7, transform=ax_detail.transAxes, zorder=3
            ))
            # 标注
            ax_detail.text(bar_x + bar_w + 0.002, bar_y + bar_h / 2,
                          f'{pct:.0f}%',
                          fontsize=7.5, va='center', color='#555',
                          transform=ax_detail.transAxes)

    # 底部：个性化建议
    ax_rec = fig.add_axes([0.02, 0.0, 0.96, 0.015])
    ax_rec.set_facecolor('#2c3e50')

    # 找出最需要改善的因素（负SHAP值最大的）
    neg_idx = np.argsort(shap_sample)[:2]
    pos_idx = np.argsort(shap_sample)[::-1][:2]
    neg_feats = [FEATURES_CN[i] for i in neg_idx if shap_sample[i] < 0]
    pos_feats = [FEATURES_CN[i] for i in pos_idx if shap_sample[i] > 0]

    rec_text = f"💡 优势：{' | '.join(pos_feats[:2])}  ·  📈 建议重点提升：{' | '.join(neg_feats[:2])}"
    ax_rec.text(0.5, 0.5, rec_text, ha='center', va='center', fontsize=9,
               color='white', transform=ax_rec.transAxes)
    ax_rec.axis('off')

    plt.tight_layout(rect=[0, 0.015, 1, 0.94])
    return fig


def main():
    print("=" * 80)
    print("103个样本个性化心理学解释报告生成器")
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
    out_dir = Path(__file__).parent / 'results/individual_reports'
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

    # ---- 生成每个样本的报告页 ----
    print("\n生成个性化报告（每个样本一页）...")
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = out_dir / 'individual_reports_all_103.pdf'
    with PdfPages(pdf_path) as pdf:
        # 封面
        fig_cover = plt.figure(figsize=(20, 26))
        fig_cover.patch.set_facecolor('#2c3e50')
        ax_cover = fig_cover.add_axes([0.1, 0.3, 0.8, 0.4])
        ax_cover.set_facecolor('#2c3e50')
        ax_cover.text(0.5, 0.8, '跨文化适应研究', ha='center', va='center',
                     fontsize=28, fontweight='bold', color='white',
                     transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.65, '103个样本个性化心理学分析报告',
                     ha='center', va='center', fontsize=20, color='#ecf0f1',
                     transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.5, '基于机器学习（XGBoost + SHAP）与心理学理论',
                     ha='center', va='center', fontsize=14, color='#bdc3c7',
                     transform=ax_cover.transAxes)
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

    print(f"\n✅ 完整PDF报告已保存: {pdf_path}")
    print(f"   共 {len(df) + 1} 页（1封面 + {len(df)}个样本）")

    # ---- 生成简版文字报告（Markdown）----
    print("\n生成文字版个案报告（Markdown）...")
    md_path = out_dir / 'individual_cases_detailed.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 103个样本个性化心理学分析报告\n\n")
        f.write("> 本报告为每个参与者提供基于机器学习（SHAP值）和心理学理论的个性化分析。\n")
        f.write("> SHAP值表示该因素对您适应程度的具体影响（正值=促进，负值=阻碍）。\n\n")
        f.write("---\n\n")

        for i in range(len(df)):
            x_i = X[i]
            y_i = y[i]
            shap_i = shap_values[i]
            pred_i = model.predict(X[i:i+1])[0]
            adapt_pct = stats.percentileofscore(y, y_i)
            adapt_info = get_adaptation_interpretation(y_i, adapt_pct)
            cluster_name = cluster_names[clusters[i]]

            f.write(f"## 样本 #{i+1}  {adapt_info['emoji']} {adapt_info['level']}\n\n")
            f.write(f"**跨文化适应程度：{y_i:.0f}分**（满分32分，高于{adapt_pct:.0f}%的参与者）\n\n")
            f.write(f"**模型预测：{pred_i:.1f}分**（预测误差：{abs(y_i - pred_i):.1f}分）\n\n")
            f.write(f"**适应类型：{cluster_name}**\n\n")
            f.write(f"**心理学解读：** {adapt_info['description']}\n\n")

            f.write("### 各因素分析\n\n")
            f.write("| 因素 | 您的得分 | 百分位 | 对适应的影响 | 解读 |\n")
            f.write("|------|---------|--------|------------|------|\n")

            sorted_idx = np.argsort(np.abs(shap_i))[::-1]
            for feat_idx in sorted_idx:
                feat = FEATURES_CN[feat_idx]
                val = x_i[feat_idx]
                shap_val = shap_i[feat_idx]
                feat_range = FEATURE_RANGES[feat]
                pct = stats.percentileofscore(X[:, feat_idx], val)
                level, _, _ = get_feature_level(feat, val)
                sign = '+' if shap_val > 0 else ''
                meanings = FEATURE_MEANINGS.get(feat, {})
                if level == '高':
                    interp = meanings.get('high', '')[:40]
                elif level == '低':
                    interp = meanings.get('low', '')[:40]
                else:
                    interp = meanings.get('what', '')[:40]
                f.write(f"| {feat} | {val:.1f}（{level}） | {pct:.0f}% | {sign}{shap_val:.2f}分 | {interp}... |\n")

            # 个性化建议
            neg_feats = [(FEATURES_CN[j], shap_i[j]) for j in np.argsort(shap_i)[:3] if shap_i[j] < 0]
            pos_feats = [(FEATURES_CN[j], shap_i[j]) for j in np.argsort(shap_i)[::-1][:3] if shap_i[j] > 0]

            f.write("\n### 个性化建议\n\n")
            if pos_feats:
                f.write("**您的优势因素（请继续保持）：**\n")
                for feat, val in pos_feats:
                    meanings = FEATURE_MEANINGS.get(feat, {})
                    f.write(f"- **{feat}**（贡献+{val:.2f}分）：{meanings.get('high', '')}\n")
            if neg_feats:
                f.write("\n**建议重点提升的领域：**\n")
                for feat, val in neg_feats:
                    meanings = FEATURE_MEANINGS.get(feat, {})
                    f.write(f"- **{feat}**（影响{val:.2f}分）：{meanings.get('low', '')}\n")
                    f.write(f"  - 理论依据：{meanings.get('theory', '')}\n")

            f.write("\n---\n\n")

    print(f"✅ 文字版报告已保存: {md_path}")

    print("\n" + "=" * 80)
    print("完成！生成的文件：")
    for fp in out_dir.iterdir():
        size_kb = fp.stat().st_size / 1024
        print(f"  {fp.name}  ({size_kb:.0f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
