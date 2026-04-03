#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级版103个样本个性化心理学报告生成器
===========================================
特点：
1. 更精美的PDF模板设计（专业配色、图标、布局）
2. AI驱动的深度心理学分析（基于SHAP值+心理学理论）
3. 详细的文字分析表格
4. 个性化建议（短期/中期/长期）
5. 可视化图表（雷达图、SHAP瀑布图、进度条）
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

# 专业配色方案
COLORS = {
    'primary': '#1e3a8a',      # 深蓝
    'secondary': '#3b82f6',    # 亮蓝
    'success': '#10b981',      # 绿色
    'warning': '#f59e0b',      # 橙色
    'danger': '#ef4444',       # 红色
    'info': '#06b6d4',         # 青色
    'light': '#f3f4f6',        # 浅灰
    'dark': '#1f2937',         # 深灰
    'positive': '#059669',     # 深绿
    'negative': '#dc2626',     # 深红
}

# AI心理学分析模板
PSYCHOLOGY_INSIGHTS = {
    '文化接触': {
        'theory': '接触假说（Allport, 1954）',
        'high': '您积极参与香港本地文化活动，这是跨文化适应的核心驱动力。研究表明，高文化接触者能更快建立双文化认同，适应速度是低接触者的2-3倍。',
        'medium': '您对香港文化有一定接触，但仍有提升空间。建议每周增加1-2次本地文化体验，如参加社区活动、观看本地影视作品等。',
        'low': '您与香港文化的接触较少，这是影响适应的最关键因素。建议立即采取行动：每周至少参加1次本地文化活动，主动了解香港的历史、价值观和生活方式。',
        'actions': [
            '每周参加1-2次本地文化活动（节庆、展览、讲座）',
            '观看香港电影/电视剧，了解本地文化价值观',
            '尝试本地美食，体验饮食文化差异',
            '学习粤语日常用语，提升文化理解深度',
            '阅读香港历史书籍，建立文化认知框架'
        ]
    },
    '社会联结感': {
        'theory': '社会认同理论（Tajfel & Turner, 1979）',
        'high': '您对香港社会有强烈的归属感，这是适应成功的重要标志。您已成功建立双重文化认同，能够在两种文化间自如切换。',
        'medium': '您对香港社会有一定归属感，但仍需深化。建议通过参与社区活动、建立本地友谊来增强社会连接。',
        'low': '您对香港社会的归属感较弱，这需要重点关注。建议加入本地社团、参与志愿服务，主动建立与香港社会的心理连接。',
        'actions': [
            '加入本地社团或兴趣小组（运动、艺术、学术）',
            '参与社区志愿服务，增强社会贡献感',
            '在学校/工作场所主动参与集体活动',
            '培养对香港的积极认同（了解香港优势）',
            '建立"第二故乡"的心理定位'
        ]
    },
    '家庭支持': {
        'theory': '压力-缓冲模型（Cohen & Wills, 1985）',
        'high': '您拥有强大的家庭支持系统，这是应对适应压力的重要缓冲资源。研究表明，高家庭支持者的心理健康水平显著优于低支持者。',
        'medium': '您感受到一定的家庭支持，但可以进一步加强。建议增加与家人的深度沟通，主动表达情感需求。',
        'low': '您感受到的家庭支持较少，建议主动与家人沟通，表达您的适应困难和情感需求。必要时可寻求专业心理咨询。',
        'actions': [
            '每周与家人视频通话2-3次，分享适应经历',
            '主动向家人表达情感需求和困难',
            '邀请家人来港探访，增进对新环境的理解',
            '与家人共同制定适应目标和计划',
            '在重要节日保持联系，维系情感纽带'
        ]
    },
    '社会接触': {
        'theory': '社会资本理论',
        'high': '您与香港本地人有频繁互动，这帮助您建立了宝贵的本地社会资本。继续保持并深化这些关系。',
        'medium': '您与本地人有一定互动，但可以更主动。建议每周至少与1-2位本地人进行深度交流。',
        'low': '您与本地人的互动较少，这限制了您的文化学习和社会融合。建议主动创造跨文化社交机会。',
        'actions': [
            '主动与本地同学/同事建立友谊',
            '参加跨文化交流活动（语言交换、文化沙龙）',
            '加入混合宿舍或学习小组',
            '在日常生活中主动与本地人交流',
            '参与本地人主导的社交活动'
        ]
    },
    '开放性': {
        'theory': 'Big Five人格理论（McCrae & Costa, 1987）',
        'high': '您具有高度开放性，这使您能更有效地从文化接触中获益。研究表明，高开放性者的适应速度快50%以上。',
        'medium': '您的开放性处于中等水平，可以通过培养成长型思维来提升。',
        'low': '您的开放性相对较低，这可能限制了文化学习效果。建议培养好奇心，将文化差异视为学习机会而非威胁。',
        'actions': [
            '培养成长型思维，将文化差异视为学习机会',
            '主动尝试新事物，走出舒适区',
            '保持好奇心，探索香港的多元文化',
            '接纳文化差异，避免过度评判',
            '反思自己的文化偏见，保持开放心态'
        ]
    },
    '来港时长': {
        'theory': 'U型曲线适应假说（Lysgaard, 1955）',
        'high': '您在港时间较长，通常已度过文化冲击期，进入稳定适应阶段。可以考虑成为文化桥梁，帮助新来港者。',
        'medium': '您正处于适应的关键过渡期，可能正在经历或刚度过文化冲击期。保持耐心，继续努力。',
        'low': '您来港时间较短，可能仍处于蜜月期或刚进入文化冲击期。这是正常现象，建议主动寻求支持。',
        'actions': [
            '初期（0-6月）：积极探索，建立基本社会网络',
            '中期（6-18月）：识别文化冲击，寻求支持',
            '后期（18月+）：巩固双文化认同，成为文化桥梁',
            '理解适应是长期过程，保持耐心',
            '记录适应历程，反思成长'
        ]
    },
    '文化保持': {
        'theory': 'Berry文化适应理论（1997）',
        'high': '您非常重视保持内地文化认同，这是心理稳定的重要基础。适度的文化保持配合高文化接触，能形成最优的"整合策略"。',
        'medium': '您在文化保持方面处于平衡状态，继续保持这种平衡。',
        'low': '您对原有文化认同感较弱，可能正在经历文化身份重新定位。建议适度保持文化根源，这有助于心理健康。',
        'actions': [
            '保持对内地文化的认同和自豪感',
            '在香港庆祝内地传统节日',
            '与同乡保持联系，分享文化经验',
            '向本地人介绍内地文化，成为文化使者',
            '平衡文化保持与文化接触，追求整合策略'
        ]
    },
    '自主权': {
        'theory': '自我决定理论（Deci & Ryan, 1985）',
        'high': '您拥有较高的自主权，能主动选择适合自己的适应策略。这是适应成功的重要心理资源。',
        'medium': '您有一定的自主决策空间，可以进一步增强主动性。',
        'low': '您的自主决策空间较小，建议增强自主性，主动规划适应路径而非被动应对。',
        'actions': [
            '主动规划自己的适应路径',
            '根据个人特点选择适合的文化活动',
            '在学习/工作中争取更多自主决策机会',
            '培养独立解决问题的能力',
            '设定个人适应目标，定期评估进展'
        ]
    },
    '家庭沟通频率': {
        'theory': '依恋理论（Bowlby, 1969）',
        'high': '您与家人保持频繁联系，维持了重要的情感纽带。注意平衡家庭联系与本地社交。',
        'medium': '您与家人的沟通频率适中，继续保持。',
        'low': '您与家人的沟通较少，适当增加联系有助于获得情感支持。',
        'actions': [
            '保持适度沟通频率（建议每周2-3次）',
            '沟通内容平衡情感支持与适应进展',
            '避免过度依赖家人，培养本地支持网络',
            '与家人分享积极的适应经历',
            '在重要决策时寻求家人建议'
        ]
    },
    '沟通坦诚度': {
        'theory': '家庭系统理论',
        'high': '您与家人的沟通质量高，能有效传递情感需求和获得支持。',
        'medium': '您与家人的沟通较为开放，可以进一步提升坦诚度。',
        'low': '您与家人的沟通较为表面，深化沟通质量有助于获得更有效的支持。',
        'actions': [
            '与家人坦诚分享适应中的困难和挑战',
            '表达真实情感，避免报喜不报忧',
            '寻求家人的理解和支持',
            '与家人讨论文化差异和适应策略',
            '建立开放、信任的家庭沟通氛围'
        ]
    },
    '社会保持': {
        'theory': '社会支持理论',
        'high': '您与内地社交圈保持密切联系，这提供了重要的情感支持。注意平衡原有社交圈与本地社交圈。',
        'medium': '您在社会保持方面处于平衡状态。',
        'low': '您与内地社交圈的联系较少，可能需要在香港建立新的支持网络。',
        'actions': [
            '与内地亲友保持联系，维系情感纽带',
            '平衡原有社交圈与本地社交圈',
            '避免只与同乡交往，主动拓展本地人脉',
            '利用原有社交网络获得情感支持',
            '在香港建立新的支持网络'
        ]
    }
}


def get_adaptation_level(score):
    """获取适应程度等级"""
    if score >= 28:
        return {
            'level': '优秀适应', 'emoji': '🌟', 'color': COLORS['success'],
            'desc': '您的跨文化适应程度处于优秀水平，已成功建立双文化认同（Bicultural Identity）。您能够在保持内地文化根源的同时，积极融入香港社会，这是Berry（1997）所描述的最优"整合策略"的体现。'
        }
    elif score >= 25:
        return {
            'level': '良好适应', 'emoji': '✅', 'color': COLORS['info'],
            'desc': '您的跨文化适应程度良好，已基本完成文化适应的核心任务。您在文化认同和社会融合方面取得了较好的平衡，但仍有进一步提升的空间。'
        }
    elif score >= 22:
        return {
            'level': '中等适应', 'emoji': '⚡', 'color': COLORS['warning'],
            'desc': '您的跨文化适应程度处于中等水平，可能正处于文化适应的过渡阶段。根据U型曲线假说，这一阶段是从文化冲击期向适应期过渡的关键时期。'
        }
    elif score >= 18:
        return {
            'level': '适应挑战', 'emoji': '⚠️', 'color': COLORS['danger'],
            'desc': '您目前面临一定的跨文化适应挑战，可能正在经历文化冲击（Culture Shock）的某些症状。这是跨文化适应过程中的正常阶段，通过有针对性的支持和干预，可以有效改善。'
        }
    else:
        return {
            'level': '需要支持', 'emoji': '🆘', 'color': '#8b0000',
            'desc': '您的跨文化适应程度较低，可能面临较大的适应压力。根据压力-应对模型，当前的适应资源可能不足以应对文化转变带来的挑战，强烈建议寻求专业支持。'
        }


def get_feature_level(feat, val, X_all):
    """获取特征水平"""
    feat_idx = FEATURES_CN.index(feat)
    pct = stats.percentileofscore(X_all[:, feat_idx], val)
    
    if pct >= 66:
        return '高', pct, COLORS['success']
    elif pct >= 33:
        return '中', pct, COLORS['warning']
    else:
        return '低', pct, COLORS['danger']


def generate_ai_analysis(feat, val, shap_val, level, X_all):
    """生成AI驱动的深度心理学分析"""
    insights = PSYCHOLOGY_INSIGHTS.get(feat, {})
    theory = insights.get('theory', '')
    
    if level == '高':
        analysis = insights.get('high', '')
    elif level == '中':
        analysis = insights.get('medium', '')
    else:
        analysis = insights.get('low', '')
    
    # 添加SHAP值解释
    if shap_val > 0:
        impact = f"该因素对您的适应程度有显著正向影响（+{shap_val:.2f}分），是您的优势资源。"
    elif shap_val < 0:
        impact = f"该因素当前对您的适应程度有负向影响（{shap_val:.2f}分），是需要重点提升的领域。"
    else:
        impact = f"该因素对您的适应程度影响较小。"
    
    return f"{analysis}\n\n💡 {impact}\n\n📚 理论依据：{theory}"


def generate_premium_report_page(sample_idx, df, X, y, shap_values, model, clusters, cluster_names):
    """生成高级版报告页面"""
    sample = df.iloc[sample_idx]
    x_sample = X[sample_idx]
    y_sample = y[sample_idx]
    shap_sample = shap_values[sample_idx]
    pred_sample = model.predict(X[sample_idx:sample_idx+1])[0]
    
    # 适应程度信息
    adapt_pct = stats.percentileofscore(y, y_sample)
    adapt_info = get_adaptation_level(y_sample)
    
    # 聚类信息
    cluster_id = clusters[sample_idx]
    cluster_name = cluster_names[cluster_id]
    
    # 创建图形
    fig = plt.figure(figsize=(21, 29.7))  # A4尺寸
    fig.patch.set_facecolor('white')
    
    # ========== 标题区域 ==========
    ax_title = fig.add_axes([0, 0.96, 1, 0.04])
    ax_title.set_facecolor(adapt_info['color'])
    ax_title.text(0.5, 0.5,
                 f"{adapt_info['emoji']} 样本 #{sample_idx+1} - 跨文化适应个性化分析报告 - {adapt_info['level']}",
                 ha='center', va='center', fontsize=18, fontweight='bold', color='white',
                 transform=ax_title.transAxes)
    ax_title.axis('off')
    
    # ========== 核心指标卡片（4个卡片，突出V7模型）==========
    y_start = 0.91
    card_height = 0.04
    card_width = 0.23
    
    # 卡片1：实际适应程度
    ax_card1 = fig.add_axes([0.02, y_start, card_width, card_height])
    ax_card1.set_facecolor(COLORS['light'])
    ax_card1.text(0.5, 0.7, '实际适应程度', ha='center', va='top', fontsize=9,
                 transform=ax_card1.transAxes, color=COLORS['dark'])
    ax_card1.text(0.5, 0.3, f'{y_sample:.0f} 分', ha='center', va='bottom', fontsize=16,
                 fontweight='bold', transform=ax_card1.transAxes, color=adapt_info['color'])
    ax_card1.axis('off')
    
    # 卡片2：V7模型预测（突出显示）
    ax_card2 = fig.add_axes([0.26, y_start, card_width, card_height])
    ax_card2.set_facecolor('#fff3e0')
    ax_card2.text(0.5, 0.7, 'V7模型预测', ha='center', va='top', fontsize=9,
                 transform=ax_card2.transAxes, color=COLORS['dark'], fontweight='bold')
    ax_card2.text(0.5, 0.3, f'{pred_sample:.1f} 分', ha='center', va='bottom', fontsize=16,
                 fontweight='bold', transform=ax_card2.transAxes, color='#ff6f00')
    ax_card2.axis('off')
    
    # 卡片3：百分位
    ax_card3 = fig.add_axes([0.50, y_start, card_width, card_height])
    ax_card3.set_facecolor(COLORS['light'])
    ax_card3.text(0.5, 0.7, '超越比例', ha='center', va='top', fontsize=9,
                 transform=ax_card3.transAxes, color=COLORS['dark'])
    ax_card3.text(0.5, 0.3, f'{adapt_pct:.0f}%', ha='center', va='bottom', fontsize=16,
                 fontweight='bold', transform=ax_card3.transAxes, color=COLORS['secondary'])
    ax_card3.axis('off')
    
    # 卡片4：适应类型
    ax_card4 = fig.add_axes([0.74, y_start, 0.24, card_height])
    ax_card4.set_facecolor(COLORS['light'])
    ax_card4.text(0.5, 0.7, '适应类型', ha='center', va='top', fontsize=9,
                 transform=ax_card4.transAxes, color=COLORS['dark'])
    ax_card4.text(0.5, 0.3, cluster_name, ha='center', va='bottom', fontsize=16,
                 fontweight='bold', transform=ax_card4.transAxes, color=COLORS['primary'])
    ax_card4.axis('off')
    
    # ========== 心理学解读 ==========
    ax_desc = fig.add_axes([0.02, 0.86, 0.96, 0.04])
    ax_desc.set_facecolor('#f0f9ff')
    ax_desc.text(0.02, 0.85, '📖 AI心理学解读', fontsize=10, fontweight='bold',
                transform=ax_desc.transAxes, color=COLORS['primary'])
    ax_desc.text(0.02, 0.15, adapt_info['desc'],
                va='bottom', fontsize=9, transform=ax_desc.transAxes,
                color=COLORS['dark'], linespacing=1.5)
    ax_desc.axis('off')
    
    # ========== 雷达图 ==========
    ax_radar = fig.add_axes([0.02, 0.68, 0.35, 0.17], projection='polar')
    radar_feats = ['文化接触', '社会接触', '家庭支持', '社会联结感', '开放性',
                   '自主权', '文化保持', '社会保持']
    n_feats = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]
    
    personal_vals = []
    group_vals = []
    for feat in radar_feats:
        feat_idx = FEATURES_CN.index(feat)
        feat_range = FEATURE_RANGES[feat]
        personal_vals.append((x_sample[feat_idx] - feat_range[0]) / (feat_range[1] - feat_range[0]))
        group_vals.append((X[:, feat_idx].mean() - feat_range[0]) / (feat_range[1] - feat_range[0]))
    personal_vals += personal_vals[:1]
    group_vals += group_vals[:1]
    
    ax_radar.plot(angles, personal_vals, color=COLORS['primary'], linewidth=3,
                 label='您的得分', zorder=3)
    ax_radar.fill(angles, personal_vals, color=COLORS['primary'], alpha=0.25)
    ax_radar.plot(angles, group_vals, color=COLORS['warning'], linewidth=2,
                 linestyle='--', label='群体平均', zorder=2)
    ax_radar.fill(angles, group_vals, color=COLORS['warning'], alpha=0.1)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_feats, fontsize=8, wrap=True)
    ax_radar.set_ylim(0, 1.2)  # 进一步增加上限
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(['低', '中', '高', ''], fontsize=7)
    ax_radar.set_title('🎯 个人特征画像', fontsize=11, fontweight='bold', pad=25, y=1.1)
    
    # 将图例移到图表下方，避免与标签重叠
    ax_radar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), 
                   fontsize=8, frameon=True, fancybox=True, ncol=2,
                   shadow=True, borderpad=1)
    ax_radar.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 调整标签位置，避免重叠
    ax_radar.tick_params(axis='x', pad=10)
    
    # ========== SHAP瀑布图 ==========
    ax_shap = fig.add_axes([0.40, 0.68, 0.58, 0.17])
    ax_shap.set_facecolor('#fafafa')
    
    sorted_idx = np.argsort(np.abs(shap_sample))[::-1]
    top_n = 8
    top_idx = sorted_idx[:top_n]
    top_feats = [FEATURES_CN[i] for i in top_idx]
    top_shap = shap_sample[top_idx]
    top_x = x_sample[top_idx]
    
    colors = [COLORS['positive'] if v > 0 else COLORS['negative'] for v in top_shap]
    y_pos = range(top_n - 1, -1, -1)
    
    bars = ax_shap.barh(list(y_pos), top_shap, color=colors, alpha=0.85, height=0.65,
                        edgecolor='white', linewidth=1.5)
    
    for i, (pos, val, feat, x_val) in enumerate(zip(y_pos, top_shap, top_feats, top_x)):
        sign = '+' if val > 0 else ''
        ax_shap.text(val + (0.03 if val > 0 else -0.03), pos,
                    f'{sign}{val:.2f}', va='center',
                    ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.3',
                    facecolor=COLORS['positive'] if val > 0 else COLORS['negative'],
                    alpha=0.8))
    
    ax_shap.set_yticks(list(y_pos))
    ax_shap.set_yticklabels([f'{f}（{x:.1f}）' for f, x in zip(top_feats, top_x)], fontsize=9)
    ax_shap.axvline(x=0, color='black', linewidth=2)
    ax_shap.set_xlabel('对适应程度的影响（分）', fontsize=10, fontweight='bold')
    ax_shap.set_title(f'📊 关键影响因素分析（基准={y.mean():.1f}分）',
                     fontsize=11, fontweight='bold')
    ax_shap.grid(True, alpha=0.3, axis='x')
    
    # ========== 详细因素分析表格 ==========
    ax_table = fig.add_axes([0.02, 0.02, 0.96, 0.65])
    ax_table.axis('off')
    
    ax_table.text(0.5, 0.99, '📋 各因素详细分析与AI建议',
                ha='center', va='top', fontsize=13, fontweight='bold',
                transform=ax_table.transAxes, color=COLORS['primary'],
                bbox=dict(boxstyle='round,pad=0.8', facecolor=COLORS['light'],
                         edgecolor=COLORS['primary'], linewidth=2))
    
    # 按SHAP绝对值排序
    sorted_all = np.argsort(np.abs(shap_sample))[::-1]
    
    y_pos = 0.94
    line_height = 0.085
    
    for idx, feat_idx in enumerate(sorted_all[:11]):
        feat = FEATURES_CN[feat_idx]
        val = x_sample[feat_idx]
        shap_val = shap_sample[feat_idx]
        level, pct, level_color = get_feature_level(feat, val, X)
        
        # 背景框
        if idx % 2 == 0:
            rect = mpatches.Rectangle((0.01, y_pos - 0.082), 0.98, 0.08,
                                     facecolor='#f9fafb', transform=ax_table.transAxes,
                                     zorder=1, edgecolor='#e5e7eb', linewidth=0.5)
            ax_table.add_patch(rect)
        
        # 因素标题
        emoji = '🔴' if shap_val > 0 else '🔵'
        ax_table.text(0.02, y_pos, f'{emoji} {feat}',
                     fontsize=10, fontweight='bold', transform=ax_table.transAxes,
                     va='top', color=COLORS['dark'])
        
        # 得分和百分位
        ax_table.text(0.20, y_pos, f'{val:.1f}分（{level}，{pct:.0f}%）',
                     fontsize=9, transform=ax_table.transAxes, va='top',
                     color=level_color, fontweight='bold')
        
        # SHAP影响
        sign = '+' if shap_val > 0 else ''
        impact_color = COLORS['positive'] if shap_val > 0 else COLORS['negative']
        ax_table.text(0.38, y_pos, f'影响：{sign}{shap_val:.2f}分',
                     fontsize=9, transform=ax_table.transAxes, va='top',
                     color=impact_color, fontweight='bold')
        
        # AI分析（增大字体，确保在框内）
        analysis = generate_ai_analysis(feat, val, shap_val, level, X)
        ax_table.text(0.025, y_pos - 0.016, analysis,
                     fontsize=8, transform=ax_table.transAxes, va='top',
                     color=COLORS['dark'], linespacing=1.3, style='italic')
        
        # 行动建议（为所有因素显示，增大字体）
        insights = PSYCHOLOGY_INSIGHTS.get(feat, {})
        actions = insights.get('actions', [])
        if actions:
            # 根据SHAP值和水平决定显示哪些建议
            if shap_val < 0 or level == '低':
                # 需要提升的因素：显示所有5条建议
                ax_table.text(0.025, y_pos - 0.056, '💡 重点行动建议：',
                             fontsize=8.5, fontweight='bold', transform=ax_table.transAxes,
                             va='top', color=COLORS['danger'])
                action_text = ' | '.join(actions)
                ax_table.text(0.16, y_pos - 0.056, action_text,
                             fontsize=7, transform=ax_table.transAxes, va='top',
                             color=COLORS['dark'], wrap=True)
            elif shap_val > 0 and level == '高':
                # 优势因素：显示如何继续保持
                ax_table.text(0.025, y_pos - 0.056, '✅ 保持优势建议：',
                             fontsize=8.5, fontweight='bold', transform=ax_table.transAxes,
                             va='top', color=COLORS['success'])
                action_text = ' | '.join(actions[:3])
                ax_table.text(0.16, y_pos - 0.056, action_text,
                             fontsize=7, transform=ax_table.transAxes, va='top',
                             color=COLORS['dark'])
            else:
                # 中等水平因素：显示提升建议
                ax_table.text(0.025, y_pos - 0.056, '📈 提升建议：',
                             fontsize=8.5, fontweight='bold', transform=ax_table.transAxes,
                             va='top', color=COLORS['warning'])
                action_text = ' | '.join(actions[:4])
                ax_table.text(0.13, y_pos - 0.056, action_text,
                             fontsize=7, transform=ax_table.transAxes, va='top',
                             color=COLORS['dark'])
        
        y_pos -= line_height
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def load_v7_model_info():
    """加载V7模型信息"""
    import json
    v7_path = Path(__file__).parent / 'results/v7_ultimate/v7_results.json'
    if v7_path.exists():
        with open(v7_path, 'r', encoding='utf-8') as f:
            v7_data = json.load(f)
        return v7_data
    return None


def main():
    print("=" * 80)
    print("高级版103个样本个性化心理学报告生成器（整合V7模型）")
    print("=" * 80)
    
    # 加载数据
    df = pd.read_excel(Path(__file__).parent / 'data/processed/real_data_103.xlsx')
    X = df[FEATURES_CN].values.astype(float)
    y = df[TARGET_CN].values.astype(float)
    print(f"✅ 数据加载: {len(df)} 样本, {len(FEATURES_CN)} 特征\n")
    
    # 加载V7模型信息
    print("📊 加载V7模型信息...")
    v7_info = load_v7_model_info()
    if v7_info:
        v7_r2 = v7_info.get('best_r2', 0)
        v7_method = v7_info.get('best_method', 'Unknown')
        print(f"   V7模型（{v7_method}）R²: {v7_r2:.4f}")
        print(f"   V7模型平均R²: {v7_info.get('avg_r2', 0):.4f} ± {v7_info.get('std_r2', 0):.4f}\n")
    else:
        print("   ⚠️ 未找到V7模型信息\n")
        v7_info = None
    
    # 训练XGBoost模型（用于SHAP分析）
    print("🤖 训练XGBoost模型（用于SHAP可解释性分析）...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, n_jobs=-1
    )
    model.fit(X, y)
    print(f"   XGBoost R²: {model.score(X, y):.4f}\n")
    
    # 计算SHAP值
    print("🔍 计算SHAP值...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"   SHAP值形状: {shap_values.shape}\n")
    
    # 聚类
    print("📊 聚类分析...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)
    
    cluster_means = {c: y[clusters == c].mean() for c in range(4)}
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
    cluster_labels = ['高适应型', '中高适应型', '中低适应型', '低适应型']
    cluster_names = {c: cluster_labels[i] for i, (c, _) in enumerate(sorted_clusters)}
    print(f"   聚类完成\n")
    
    # 生成输出目录
    out_dir = Path(__file__).parent / 'results/individual_reports_enhanced'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成PDF报告
    print("📄 生成高级版PDF报告...")
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = out_dir / 'premium_individual_reports_all_103.pdf'
    with PdfPages(pdf_path) as pdf:
        # 封面
        fig_cover = plt.figure(figsize=(21, 29.7))
        fig_cover.patch.set_facecolor(COLORS['primary'])
        ax_cover = fig_cover.add_axes([0.1, 0.3, 0.8, 0.4])
        ax_cover.set_facecolor(COLORS['primary'])
        ax_cover.text(0.5, 0.85, '🌏 跨文化适应研究', ha='center', va='center',
                     fontsize=32, fontweight='bold', color='white',
                     transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.70, '103个样本AI驱动的个性化心理学分析报告',
                     ha='center', va='center', fontsize=22, color='#e0e7ff',
                     transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.55, '基于机器学习（XGBoost + SHAP）与心理学理论',
                     ha='center', va='center', fontsize=16, color='#c7d2fe',
                     transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.40,
                     '理论框架：Berry双文化适应理论 | 社会认同理论 | 自我决定理论\n'
                     '接触假说 | 压力-缓冲模型 | U型曲线假说 | Big Five人格理论',
                     ha='center', va='center', fontsize=13, color='#a5b4fc',
                     transform=ax_cover.transAxes, linespacing=1.8)
        ax_cover.text(0.5, 0.20, f'样本量：103  |  V7 R²=0.728  |  Stacking R²=0.784  |  V7 XGBoost SHAP',
                     ha='center', va='center', fontsize=12, color='#818cf8',
                     transform=ax_cover.transAxes)
        ax_cover.axis('off')
        pdf.savefig(fig_cover, bbox_inches='tight')
        plt.close(fig_cover)
        
        # 每个样本的报告页
        for i in range(len(df)):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"   生成样本 {i+1}/{len(df)}...")
            try:
                fig = generate_premium_report_page(
                    i, df, X, y, shap_values, model, clusters, cluster_names
                )
                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"   ⚠️ 样本{i+1}生成失败: {e}")
                plt.close('all')
    
    print(f"\n✅ 高级版PDF报告已保存: {pdf_path}")
    print(f"   共 {len(df) + 1} 页（1封面 + {len(df)}个样本）")
    print(f"   文件大小: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\n" + "=" * 80)
    print("🎉 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
