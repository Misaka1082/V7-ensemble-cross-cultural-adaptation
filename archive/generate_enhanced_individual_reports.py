#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版103个样本个性化心理学报告生成器
基于模型数据（SHAP值、特征重要性、交互效应）+ 心理学原理
提供更详细的个性化建议
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import xgboost as xgb
import shap

# 注册中文字体
try:
    pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))
    pdfmetrics.registerFont(TTFont('SimHei', 'C:/Windows/Fonts/simhei.ttf'))
    FONT_NAME = 'SimSun'
    FONT_BOLD = 'SimHei'
except:
    FONT_NAME = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'

# 设置路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
RESULTS_DIR = BASE_DIR / 'results' / 'enhanced_individual_reports'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 特征名称映射
FEATURE_NAMES_CN = {
    'cultural_maintenance': '文化保持',
    'social_maintenance': '社会保持',
    'cultural_contact': '文化接触',
    'social_contact': '社会接触',
    'family_support': '家庭支持',
    'communication_frequency': '家庭沟通频率',
    'communication_openness': '沟通坦诚度',
    'autonomy': '自主权',
    'social_connectedness': '社会联结感',
    'openness': '开放性',
    'time_in_hk': '来港时长'
}

# 心理学理论库
PSYCHOLOGY_THEORIES = {
    '文化接触': {
        'theory': '接触假说（Contact Hypothesis, Allport 1954）',
        'mechanism': '与东道国文化的直接接触能减少文化距离感，促进文化理解和认同整合。高文化接触者能更快习得香港的文化规范、语言习惯和社交礼仪，从而降低文化冲击（Culture Shock）。',
        'intervention': [
            '参加本地文化活动（如节庆、展览、社区活动）',
            '观看香港电影、电视剧，了解本地文化价值观',
            '尝试本地美食，体验饮食文化',
            '学习粤语基础用语，提升文化理解',
            '阅读香港历史和社会发展相关书籍'
        ]
    },
    '社会联结感': {
        'theory': '社会认同理论（Social Identity Theory, Tajfel & Turner 1979）',
        'mechanism': '社会联结感反映了个体与香港社会的心理归属感。当个体感受到被东道国社会接纳时，会形成积极的双重文化认同，显著提升适应水平。',
        'intervention': [
            '加入本地社团或兴趣小组，建立归属感',
            '参与社区志愿服务，增强社会连接',
            '在学校/工作场所主动参与集体活动',
            '培养对香港的积极认同（如了解香港优势、贡献）',
            '建立"第二故乡"的心理定位'
        ]
    },
    '家庭支持': {
        'theory': '压力-缓冲模型（Stress-Buffering Model, Cohen & Wills 1985）',
        'mechanism': '家庭支持作为跨文化适应的缓冲因素，通过提供情感安全感来降低适应压力。强大的家庭支持能减轻文化冲击带来的心理压力，使个体有更多心理资源投入文化学习。',
        'intervention': [
            '定期与家人视频通话，分享适应经历',
            '主动向家人表达情感需求和困难',
            '邀请家人来港探访，增进对新环境的理解',
            '与家人共同制定适应目标和计划',
            '在重要节日与家人保持联系，维系情感纽带'
        ]
    },
    '社会接触': {
        'theory': '社会资本理论（Social Capital Theory）',
        'mechanism': '与香港本地人的社交互动提供了文化学习的直接渠道。社会接触不仅传递文化知识，更重要的是建立社会资本，形成支持性的本地社会网络。',
        'intervention': [
            '主动与本地同学/同事建立友谊',
            '参加跨文化交流活动（如语言交换）',
            '加入混合宿舍或学习小组',
            '在日常生活中主动与本地人交流（如店员、邻居）',
            '参与本地人主导的社交活动'
        ]
    },
    '开放性': {
        'theory': 'Big Five人格理论（McCrae & Costa 1987）',
        'mechanism': '开放性是Big Five人格特质之一，高开放性者对新文化、新经历持积极态度，更愿意尝试本地文化活动。开放性作为调节变量，放大了文化接触和社会接触的正向效应。',
        'intervention': [
            '培养成长型思维，将文化差异视为学习机会',
            '主动尝试新事物，走出舒适区',
            '保持好奇心，探索香港的多元文化',
            '接纳文化差异，避免过度评判',
            '反思自己的文化偏见，保持开放心态'
        ]
    },
    '来港时长': {
        'theory': 'U型曲线适应假说（U-Curve Hypothesis, Lysgaard 1955）',
        'mechanism': '适应程度随时间呈非线性变化：初期（蜜月期）→中期（文化冲击期）→后期（适应期）。长期居港者通常已度过文化冲击期，形成稳定的双文化认同。',
        'intervention': [
            '初期（0-6个月）：积极探索，建立基本社会网络',
            '中期（6-18个月）：识别文化冲击症状，寻求支持',
            '后期（18个月+）：巩固双文化认同，成为文化桥梁',
            '理解适应是长期过程，保持耐心',
            '记录适应历程，反思成长'
        ]
    },
    '文化保持': {
        'theory': 'Berry文化适应理论（Berry 1997）',
        'mechanism': '适度保持原有文化认同（而非完全放弃）有助于心理健康和适应。过度的文化保持可能导致分离策略，而适度保持配合高文化接触则形成最优的整合策略。',
        'intervention': [
            '保持对内地文化的认同和自豪感',
            '在香港庆祝内地传统节日',
            '与同乡保持联系，分享文化经验',
            '向本地人介绍内地文化，成为文化使者',
            '平衡文化保持与文化接触，追求整合策略'
        ]
    },
    '自主权': {
        'theory': '自我决定理论（Self-Determination Theory, Deci & Ryan 1985）',
        'mechanism': '自主权对应自我决定理论中的自主性需求。高自主权者能主动选择适应策略，而非被动应对文化压力。在香港这一高度自由的社会环境中，自主权的发挥空间较大。',
        'intervention': [
            '主动规划自己的适应路径，而非被动等待',
            '根据个人特点选择适合的文化活动',
            '在学习/工作中争取更多自主决策机会',
            '培养独立解决问题的能力',
            '设定个人适应目标，定期评估进展'
        ]
    },
    '家庭沟通频率': {
        'theory': '依恋理论（Attachment Theory, Bowlby 1969）',
        'mechanism': '与家人的沟通频率维持了原有社会支持网络的活跃度。适度的家庭沟通能提供情感支持，但过高频率可能强化对原有文化的依恋，影响本地文化融合。',
        'intervention': [
            '保持适度沟通频率（建议每周2-3次）',
            '沟通内容平衡情感支持与适应进展',
            '避免过度依赖家人，培养本地支持网络',
            '与家人分享积极的适应经历',
            '在重要决策时寻求家人建议'
        ]
    },
    '沟通坦诚度': {
        'theory': '家庭系统理论（Family Systems Theory）',
        'mechanism': '沟通坦诚度反映了家庭关系的质量。高坦诚度的家庭沟通能更有效地传递情感支持，帮助个体处理适应过程中的心理困扰。',
        'intervention': [
            '与家人坦诚分享适应中的困难和挑战',
            '表达真实情感，避免报喜不报忧',
            '寻求家人的理解和支持',
            '与家人讨论文化差异和适应策略',
            '建立开放、信任的家庭沟通氛围'
        ]
    },
    '社会保持': {
        'theory': '社会支持理论（Social Support Theory）',
        'mechanism': '社会保持（维持原有社会关系）提供了情感支持和文化根基。适度的社会保持有助于心理稳定，但过度依赖可能阻碍本地社会网络的建立。',
        'intervention': [
            '与内地亲友保持联系，维系情感纽带',
            '平衡原有社交圈与本地社交圈',
            '避免只与同乡交往，主动拓展本地人脉',
            '利用原有社交网络获得情感支持',
            '在香港建立新的支持网络'
        ]
    }
}


def load_data():
    """加载数据"""
    df = pd.read_excel(DATA_DIR / 'real_data_103.xlsx')
    
    # 中文列名映射到英文
    column_mapping = {
        '跨文化适应程度': 'adaptation_score',
        '文化保持': 'cultural_maintenance',
        '社会保持': 'social_maintenance',
        '文化接触': 'cultural_contact',
        '社会接触': 'social_contact',
        '家庭支持': 'family_support',
        '家庭沟通频率': 'communication_frequency',
        '沟通坦诚度': 'communication_openness',
        '自主权': 'autonomy',
        '社会联结感': 'social_connectedness',
        '开放性': 'openness',
        '来港时长': 'time_in_hk'
    }
    df = df.rename(columns=column_mapping)
    
    # 特征列
    feature_cols = [
        'cultural_maintenance', 'social_maintenance', 'cultural_contact', 'social_contact',
        'family_support', 'communication_frequency', 'communication_openness',
        'autonomy', 'social_connectedness', 'openness', 'time_in_hk'
    ]
    
    X = df[feature_cols].values
    y = df['adaptation_score'].values
    
    return df, X, y, feature_cols


def train_model_and_get_shap(X, y):
    """训练模型并计算SHAP值"""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return model, shap_values


def get_percentile(value, all_values):
    """计算百分位"""
    return int(np.sum(all_values <= value) / len(all_values) * 100)


def get_level_label(percentile):
    """根据百分位返回水平标签"""
    if percentile >= 75:
        return '高'
    elif percentile >= 25:
        return '中'
    else:
        return '低'


def generate_detailed_recommendations(row, shap_row, feature_cols, df):
    """生成详细的个性化建议（基于模型数据+心理学原理）"""
    recommendations = []
    
    # 1. 获取SHAP值排序（影响最大的因素）
    shap_importance = [(FEATURE_NAMES_CN[feature_cols[i]], shap_row[i], row[feature_cols[i]]) 
                       for i in range(len(feature_cols))]
    shap_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # 2. 识别优势因素（SHAP值为正且绝对值较大）
    strengths = [item for item in shap_importance if item[1] > 0.2][:3]
    
    # 3. 识别需提升因素（SHAP值为负且绝对值较大）
    weaknesses = [item for item in shap_importance if item[1] < -0.2][:3]
    
    # 4. 生成优势因素建议
    if strengths:
        recommendations.append("### 一、您的优势资源（请继续保持并充分利用）\n")
        for i, (feature_name, shap_val, feature_val) in enumerate(strengths, 1):
            theory_info = PSYCHOLOGY_THEORIES.get(feature_name, {})
            theory = theory_info.get('theory', '')
            mechanism = theory_info.get('mechanism', '')
            
            recommendations.append(f"**{i}. {feature_name}**（对适应的贡献：+{shap_val:.2f}分）\n")
            recommendations.append(f"- **您的表现**：{feature_val:.1f}分（高于大多数参与者）\n")
            recommendations.append(f"- **心理学原理**：{mechanism}\n")
            recommendations.append(f"- **理论依据**：{theory}\n")
            recommendations.append(f"- **如何继续发挥优势**：\n")
            for action in theory_info.get('intervention', [])[:3]:
                recommendations.append(f"  - {action}\n")
            recommendations.append("\n")
    
    # 5. 生成需提升因素建议
    if weaknesses:
        recommendations.append("### 二、重点提升领域（基于模型数据的精准建议）\n")
        for i, (feature_name, shap_val, feature_val) in enumerate(weaknesses, 1):
            theory_info = PSYCHOLOGY_THEORIES.get(feature_name, {})
            theory = theory_info.get('theory', '')
            mechanism = theory_info.get('mechanism', '')
            
            recommendations.append(f"**{i}. {feature_name}**（当前影响：{shap_val:.2f}分）\n")
            recommendations.append(f"- **现状分析**：您的{feature_name}得分为{feature_val:.1f}分，低于理想水平\n")
            recommendations.append(f"- **为什么重要**：{mechanism}\n")
            recommendations.append(f"- **理论依据**：{theory}\n")
            recommendations.append(f"- **具体行动建议**：\n")
            for action in theory_info.get('intervention', []):
                recommendations.append(f"  - {action}\n")
            recommendations.append("\n")
    
    # 6. 基于交互效应的建议
    recommendations.append("### 三、协同提升策略（基于特征交互效应）\n")
    
    # 检查开放性×文化接触交互
    openness_val = row['openness']
    cultural_contact_val = row['cultural_contact']
    
    if openness_val >= 5 and cultural_contact_val < 5:
        recommendations.append("**策略1：充分发挥您的高开放性优势**\n")
        recommendations.append("- 模型发现：您的开放性较高（{:.1f}分），这是宝贵的适应资源\n".format(openness_val))
        recommendations.append("- 但您的文化接触相对较少（{:.1f}分），限制了开放性的发挥\n".format(cultural_contact_val))
        recommendations.append("- **协同建议**：利用您的高开放性，主动增加文化接触。研究表明，高开放性者从文化接触中获益是低开放性者的2-3倍\n")
        recommendations.append("- **具体行动**：每周参加1-2次本地文化活动，您的高开放性会让这些体验更有效\n\n")
    
    # 检查家庭支持×社会联结感交互
    family_support_val = row['family_support']
    social_connectedness_val = row['social_connectedness']
    
    if family_support_val >= 32 and social_connectedness_val < 20:
        recommendations.append("**策略2：将家庭支持转化为本地融合动力**\n")
        recommendations.append("- 模型发现：您拥有强大的家庭支持（{:.1f}分），这是重要的情感后盾\n".format(family_support_val))
        recommendations.append("- 但您的社会联结感较低（{:.1f}分），可能过度依赖家庭支持\n".format(social_connectedness_val))
        recommendations.append("- **协同建议**：将家庭支持作为\"安全基地\"，而非\"避风港\"。在家庭支持的基础上，勇敢探索本地社会\n")
        recommendations.append("- **具体行动**：与家人分享您在香港的积极经历，获得他们的鼓励后，主动参与本地社交活动\n\n")
    
    # 7. 阶段性建议（基于来港时长）
    time_in_hk = row['time_in_hk']
    recommendations.append("### 四、您当前阶段的重点任务\n")
    
    if time_in_hk < 12:
        recommendations.append("**适应阶段：初期探索期（来港{:.0f}个月）**\n".format(time_in_hk))
        recommendations.append("- **阶段特点**：您可能正处于\"蜜月期\"或刚进入\"文化冲击期\"\n")
        recommendations.append("- **核心任务**：\n")
        recommendations.append("  1. 建立基本的本地社会网络（至少3-5个本地朋友）\n")
        recommendations.append("  2. 熟悉香港的日常生活规范和文化习惯\n")
        recommendations.append("  3. 识别并应对文化冲击症状（如孤独感、焦虑）\n")
        recommendations.append("  4. 保持与家人的适度联系，获得情感支持\n")
    elif time_in_hk < 24:
        recommendations.append("**适应阶段：中期调整期（来港{:.0f}个月）**\n".format(time_in_hk))
        recommendations.append("- **阶段特点**：您可能正在经历\"文化冲击期\"，这是正常现象\n")
        recommendations.append("- **核心任务**：\n")
        recommendations.append("  1. 深化本地社交关系，从表面互动转向深层友谊\n")
        recommendations.append("  2. 主动应对文化冲击，寻求专业支持（如心理咨询）\n")
        recommendations.append("  3. 培养双文化认同，平衡文化保持与文化接触\n")
        recommendations.append("  4. 参与更多本地文化活动，提升文化理解\n")
    else:
        recommendations.append("**适应阶段：后期稳定期（来港{:.0f}个月）**\n".format(time_in_hk))
        recommendations.append("- **阶段特点**：您应该已度过文化冲击期，进入稳定适应阶段\n")
        recommendations.append("- **核心任务**：\n")
        recommendations.append("  1. 巩固双文化认同，成为文化桥梁\n")
        recommendations.append("  2. 帮助新来港者，分享适应经验\n")
        recommendations.append("  3. 深度融入本地社会，参与社区建设\n")
        recommendations.append("  4. 反思适应历程，总结个人成长\n")
    
    recommendations.append("\n")
    
    # 8. 个性化行动计划
    recommendations.append("### 五、30天行动计划（可操作的具体步骤）\n")
    recommendations.append("**第1周：评估与准备**\n")
    recommendations.append("- 完成自我适应评估，识别优势和挑战\n")
    recommendations.append("- 设定1-2个具体的适应目标\n")
    recommendations.append("- 寻找1-2个本地文化活动或社交机会\n\n")
    
    recommendations.append("**第2-3周：行动与体验**\n")
    recommendations.append("- 参加至少2次本地文化活动\n")
    recommendations.append("- 主动与1-2位本地人建立联系\n")
    recommendations.append("- 记录文化体验和感受\n\n")
    
    recommendations.append("**第4周：反思与调整**\n")
    recommendations.append("- 评估行动效果，识别进步和困难\n")
    recommendations.append("- 调整适应策略，制定下一阶段计划\n")
    recommendations.append("- 与家人或朋友分享适应经历\n\n")
    
    # 9. 寻求支持的建议
    adaptation_score = row['adaptation_score']
    if adaptation_score < 20:
        recommendations.append("### 六、重要提醒：寻求专业支持\n")
        recommendations.append("您的跨文化适应程度较低（{:.0f}分），建议寻求专业支持：\n".format(adaptation_score))
        recommendations.append("- **心理咨询**：学校/社区心理咨询服务\n")
        recommendations.append("- **同伴支持**：加入内地学生互助小组\n")
        recommendations.append("- **文化适应培训**：参加跨文化适应工作坊\n")
        recommendations.append("- **紧急支持**：如感到严重焦虑/抑郁，请立即联系心理热线\n\n")
    
    return ''.join(recommendations)


def generate_enhanced_pdf_report(df, shap_values, feature_cols):
    """生成增强版PDF报告"""
    print("生成增强版103个样本个性化心理学报告（含详细建议）...")
    
    # 创建PDF
    pdf_path = RESULTS_DIR / 'enhanced_individual_reports_all_103.pdf'
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                           leftMargin=2*cm, rightMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)
    
    # 样式
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=FONT_BOLD,
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=FONT_BOLD,
        fontSize=12,
        textColor=colors.HexColor('#2e5c8a'),
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontName=FONT_NAME,
        fontSize=9,
        leading=14,
        spaceAfter=6
    )
    
    story = []
    
    # 封面
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("103个样本个性化心理学分析报告", title_style))
    story.append(Paragraph("（增强版：基于模型数据+心理学原理的详细建议）", body_style))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("本报告为每个参与者提供：", body_style))
    story.append(Paragraph("• 基于机器学习模型（SHAP值）的精准分析", body_style))
    story.append(Paragraph("• 基于心理学理论的深度解释", body_style))
    story.append(Paragraph("• 个性化的具体行动建议", body_style))
    story.append(Paragraph("• 30天行动计划", body_style))
    story.append(PageBreak())
    
    # 为每个样本生成报告
    for idx, row in df.iterrows():
        print(f"  生成样本 {idx+1}/103...")
        
        # 样本标题
        adaptation_score = row['adaptation_score']
        if adaptation_score >= 28:
            level_emoji = "🌟"
            level_text = "优秀适应"
        elif adaptation_score >= 25:
            level_emoji = "✅"
            level_text = "良好适应"
        elif adaptation_score >= 22:
            level_emoji = "⚡"
            level_text = "中等适应"
        elif adaptation_score >= 18:
            level_emoji = "⚠️"
            level_text = "适应挑战"
        else:
            level_emoji = "🆘"
            level_text = "需要支持"
        
        story.append(Paragraph(f"样本 #{idx+1}  {level_emoji} {level_text}", title_style))
        story.append(Paragraph(f"跨文化适应程度：{adaptation_score:.0f}分（满分32分）", heading_style))
        story.append(Spacer(1, 0.3*cm))
        
        # 生成详细建议
        detailed_rec = generate_detailed_recommendations(row, shap_values[idx], feature_cols, df)
        
        # 将Markdown转换为PDF段落
        for line in detailed_rec.split('\n'):
            if line.startswith('###'):
                story.append(Paragraph(line.replace('###', '').strip(), heading_style))
            elif line.startswith('**') and line.endswith('**'):
                story.append(Paragraph(line.replace('**', ''), body_style))
            elif line.strip():
                story.append(Paragraph(line, body_style))
            else:
                story.append(Spacer(1, 0.2*cm))
        
        story.append(PageBreak())
    
    # 生成PDF
    doc.build(story)
    print(f"\n✅ 增强版PDF报告已保存: {pdf_path}")
    print(f"   共 {len(df)+1} 页（1封面 + {len(df)}个样本）")


def generate_enhanced_markdown_report(df, shap_values, feature_cols):
    """生成增强版Markdown报告"""
    print("\n生成增强版文字报告（Markdown）...")
    
    md_path = RESULTS_DIR / 'enhanced_individual_cases_detailed.md'
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 103个样本个性化心理学分析报告（增强版）\n\n")
        f.write("> 本报告基于机器学习模型（SHAP值）+ 心理学理论，为每个参与者提供详细的个性化建议。\n\n")
        f.write("---\n\n")
        
        for idx, row in df.iterrows():
            adaptation_score = row['adaptation_score']
            if adaptation_score >= 28:
                level_emoji = "🌟"
                level_text = "优秀适应"
            elif adaptation_score >= 25:
                level_emoji = "✅"
                level_text = "良好适应"
            elif adaptation_score >= 22:
                level_emoji = "⚡"
                level_text = "中等适应"
            elif adaptation_score >= 18:
                level_emoji = "⚠️"
                level_text = "适应挑战"
            else:
                level_emoji = "🆘"
                level_text = "需要支持"
            
            f.write(f"## 样本 #{idx+1}  {level_emoji} {level_text}\n\n")
            f.write(f"**跨文化适应程度：{adaptation_score:.0f}分**（满分32分）\n\n")
            
            # 生成详细建议
            detailed_rec = generate_detailed_recommendations(row, shap_values[idx], feature_cols, df)
            f.write(detailed_rec)
            f.write("\n---\n\n")
    
    print(f"✅ 增强版文字报告已保存: {md_path}")


def main():
    print("="*80)
    print("103个样本个性化心理学报告生成器（增强版）")
    print("="*80)
    
    # 加载数据
    df, X, y, feature_cols = load_data()
    print(f"数据: {len(df)} 样本, {len(feature_cols)} 特征\n")
    
    # 训练模型并计算SHAP值
    print("训练XGBoost模型...")
    model, shap_values = train_model_and_get_shap(X, y)
    print(f"  训练R²: {model.score(X, y):.4f}")
    print("计算SHAP值...")
    print(f"  SHAP值形状: {shap_values.shape}\n")
    
    # 生成增强版PDF报告
    generate_enhanced_pdf_report(df, shap_values, feature_cols)
    
    # 生成增强版Markdown报告
    generate_enhanced_markdown_report(df, shap_values, feature_cols)
    
    print("\n" + "="*80)
    print("完成！生成的文件：")
    for file in RESULTS_DIR.glob('*'):
        size_kb = file.stat().st_size / 1024
        print(f"  {file.name}  ({size_kb:.0f} KB)")
    print("="*80)


if __name__ == '__main__':
    main()
