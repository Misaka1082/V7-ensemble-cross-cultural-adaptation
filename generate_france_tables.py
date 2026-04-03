"""
生成法国样本的三张表格图片（与SPSS格式一致）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_collinearity_table():
    """创建共线性诊断表（表4.2f）"""
    
    # 根据我们的计算结果创建数据
    data = {
        '维度': list(range(1, 13)),
        '特征值': [209205.22, 8981.68, 796.31, 564.95, 369.74, 332.24, 
                  313.23, 222.36, 178.89, 130.86, 51.96, 3.99],
        '条件指数': [1.00, 4.83, 16.21, 19.24, 23.79, 25.09,
                   25.84, 30.67, 34.20, 39.98, 63.45, 229.01]
    }
    
    # 添加方差比例列（简化显示）
    for i in range(11):
        data[f'变量{i+1}'] = [0] * 12
    
    df = pd.DataFrame(data)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 只显示前3列
    display_df = df[['维度', '特征值', '条件指数']]
    
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置交替行颜色
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('表4.2f 共线性诊断（法国样本）', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('f:/Project/4.1.9.final/linear_regression_results/法国_共线性诊断表.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 法国_共线性诊断表.png 已生成")

def create_residual_stats_table():
    """创建残差统计表（表4.2g）"""
    
    data = {
        '': ['预测值', '残差', '标准化预测值', '标准化残差'],
        '最小值': [1.75, -0.74, -2.37, -2.28],
        '最大值': [3.67, 0.97, 1.87, 2.98],
        '均值': [2.82, 0.00, 0.00, 0.00],
        '标准差': [0.45, 0.33, 1.00, 1.00],
        'N': [249, 249, 249, 249]
    }
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置交替行颜色
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('表4.2g 残差统计（法国样本）', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('f:/Project/4.1.9.final/linear_regression_results/法国_残差统计表.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 法国_残差统计表.png 已生成")

def create_regression_coefficients_table():
    """创建回归系数表（表4.2h）"""
    
    # 根据我们的计算结果
    data = {
        '变量': ['(常量)', '文化保持', '社会保持', '文化接触', '社会接触', '家庭支持',
                '沟通频率', '沟通坦诚度', '自主性', '社会联结感', '开放性', '居留时长'],
        'B': [0.375, 0.028, 0.023, 0.085, 0.056, 0.117,
              0.048, 0.088, -0.022, 0.147, 0.032, 0.005],
        'SE': [0.166, 0.016, 0.017, 0.016, 0.016, 0.045,
               0.022, 0.023, 0.024, 0.030, 0.017, 0.002],
        'β': ['', .046, .037, .150, .098, .079,
              .052, .098, -.020, .141, .042, .065],
        't': [2.257, 1.802, 1.403, 5.169, 3.433, 2.611,
              2.223, 3.896, -0.899, 4.968, 1.891, 2.449],
        'p': [.025, .073, .162, '<.001', '<.001', .010,
              .027, '<.001', .369, '<.001', .060, .015],
        '95% CI': ['[0.05, 0.70]', '[-0.00, 0.06]', '[-0.01, 0.06]', 
                   '[0.05, 0.12]', '[0.02, 0.09]', '[0.03, 0.21]',
                   '[0.01, 0.09]', '[0.04, 0.13]', '[-0.07, 0.03]',
                   '[0.09, 0.21]', '[-0.00, 0.06]', '[0.00, 0.01]'],
        'VIF': ['', 1.38, 1.41, 1.19, 1.21, 5.81,
                1.29, 1.34, 1.57, 2.89, 1.17, 0.61]
    }
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 格式化数据显示
    cell_text = []
    for idx, row in df.iterrows():
        formatted_row = []
        for col in df.columns:
            val = row[col]
            if col in ['B', 'SE'] and val != '':
                formatted_row.append(f'{val:.3f}')
            elif col == 'β' and val != '':
                formatted_row.append(f'{val:.3f}')
            elif col == 't' and val != '':
                formatted_row.append(f'{val:.3f}')
            elif col == 'VIF' and val != '':
                formatted_row.append(f'{val:.2f}')
            else:
                formatted_row.append(str(val))
        cell_text.append(formatted_row)
    
    table = ax.table(cellText=cell_text,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置交替行颜色和高亮显著结果
    for i in range(1, len(df) + 1):
        p_val = df.iloc[i-1]['p']
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            
            # 高亮显著结果
            if p_val in ['<.001', '.010', '.027', '.015'] and j > 0:
                table[(i, j)].set_text_props(weight='bold')
    
    plt.title('表4.2h 预测跨文化适应的多元线性回归结果（法国样本，N=249）', 
             fontsize=14, fontweight='bold', pad=20)
    
    # 添加注释
    note_text = '注：B = 未标准化回归系数；SE = 标准误；β = 标准化回归系数；CI = 置信区间。加粗表示p < .05。'
    plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=9, style='italic')
    
    plt.savefig('f:/Project/4.1.9.final/linear_regression_results/法国_回归系数表.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 法国_回归系数表.png 已生成")

def main():
    """主函数"""
    print("\n" + "="*60)
    print("生成法国样本的三张表格图片")
    print("="*60 + "\n")
    
    create_collinearity_table()
    create_residual_stats_table()
    create_regression_coefficients_table()
    
    print("\n" + "="*60)
    print("所有表格图片已生成完毕！")
    print("="*60)
    print("\n文件位置：f:/Project/4.1.9.final/linear_regression_results/")
    print("- 法国_共线性诊断表.png")
    print("- 法国_残差统计表.png")
    print("- 法国_回归系数表.png")

if __name__ == "__main__":
    main()
