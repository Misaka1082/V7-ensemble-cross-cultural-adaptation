"""
高分辨率矢量图生成脚本 - 适用于心理学论文
支持生成SVG、PDF、EPS等矢量格式
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# 设置中文字体和高质量输出
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'  # 确保SVG中文字为文本而非路径
plt.rcParams['pdf.fonttype'] = 42  # TrueType字体，便于编辑
plt.rcParams['ps.fonttype'] = 42

# 设置学术风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# 创建输出目录
output_dir = Path("vector_plots")
output_dir.mkdir(exist_ok=True)


def save_vector_plot(fig, filename, formats=['svg', 'pdf', 'eps', 'png']):
    """
    保存矢量图到多种格式
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        要保存的图形对象
    filename : str
        文件名（不含扩展名）
    formats : list
        要保存的格式列表
    """
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        if fmt == 'png':
            # PNG使用高DPI以保证质量
            fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        else:
            # 矢量格式
            fig.savefig(filepath, format=fmt, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        print(f"已保存: {filepath}")


def plot_scatter_with_regression(x, y, xlabel, ylabel, title, filename):
    """散点图 + 回归线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 散点图
    ax.scatter(x, y, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # 回归线
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r-", linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
    
    # 计算相关系数
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_vector_plot(fig, filename)
    plt.close()


def plot_bar_chart(categories, values, xlabel, ylabel, title, filename, colors=None):
    """柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if colors is None:
        colors = sns.color_palette("husl", len(categories))
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_vector_plot(fig, filename)
    plt.close()


def plot_line_chart(x, y_dict, xlabel, ylabel, title, filename):
    """多条线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(y_dict))
    
    for i, (label, y) in enumerate(y_dict.items()):
        ax.plot(x, y, marker='o', linewidth=2.5, markersize=8, 
                label=label, color=colors[i], alpha=0.8)
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_vector_plot(fig, filename)
    plt.close()


def plot_heatmap(data, xticklabels, yticklabels, title, filename, cmap='RdYlBu_r'):
    """热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # 设置刻度
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_yticklabels(yticklabels, fontsize=10)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值
    for i in range(len(yticklabels)):
        for j in range(len(xticklabels)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('相关系数', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    save_vector_plot(fig, filename)
    plt.close()


def plot_boxplot(data_dict, xlabel, ylabel, title, filename):
    """箱线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_list = list(data_dict.values())
    labels = list(data_dict.keys())
    
    bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                     notch=True, showmeans=True)
    
    # 设置颜色
    colors = sns.color_palette("Set2", len(data_list))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_vector_plot(fig, filename)
    plt.close()


def plot_violin(data_dict, xlabel, ylabel, title, filename):
    """小提琴图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 准备数据
    data_list = []
    labels = []
    for label, values in data_dict.items():
        data_list.extend(values)
        labels.extend([label] * len(values))
    
    df = pd.DataFrame({'group': labels, 'value': data_list})
    
    # 绘制小提琴图
    parts = ax.violinplot([data_dict[k] for k in data_dict.keys()],
                          positions=range(len(data_dict)),
                          showmeans=True, showmedians=True)
    
    # 设置颜色
    colors = sns.color_palette("Set2", len(data_dict))
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(data_dict)))
    ax.set_xticklabels(data_dict.keys())
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_vector_plot(fig, filename)
    plt.close()


def plot_histogram(data, xlabel, ylabel, title, filename, bins=30):
    """直方图 + 正态分布拟合"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 直方图
    n, bins_edges, patches = ax.hist(data, bins=bins, density=True, 
                                      alpha=0.7, color='steelblue', 
                                      edgecolor='black', linewidth=1.2)
    
    # 拟合正态分布
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(data.min(), data.max(), 100)
    ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2)),
            'r-', linewidth=2.5, label=f'正态分布\nμ={mu:.2f}, σ={sigma:.2f}')
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_vector_plot(fig, filename)
    plt.close()


def plot_interaction_effect(x, y1, y2, xlabel, ylabel, title, filename, 
                            label1='组1', label2='组2'):
    """交互效应图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, y1, marker='o', linewidth=2.5, markersize=10, 
            label=label1, color='steelblue', alpha=0.8)
    ax.plot(x, y2, marker='s', linewidth=2.5, markersize=10, 
            label=label2, color='coral', alpha=0.8)
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_vector_plot(fig, filename)
    plt.close()


# ============ 示例：生成各类图表 ============

def generate_example_plots():
    """生成示例图表"""
    
    print("开始生成示例矢量图...\n")
    
    # 1. 散点图示例
    np.random.seed(42)
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    plot_scatter_with_regression(
        x, y,
        xlabel='自变量 (X)',
        ylabel='因变量 (Y)',
        title='散点图与回归线示例',
        filename='01_scatter_regression'
    )
    
    # 2. 柱状图示例
    categories = ['外向性', '宜人性', '尽责性', '神经质', '开放性']
    values = [3.5, 4.2, 3.8, 2.9, 4.5]
    plot_bar_chart(
        categories, values,
        xlabel='人格维度',
        ylabel='平均得分',
        title='五大人格维度得分',
        filename='02_bar_chart'
    )
    
    # 3. 折线图示例
    months = np.arange(0, 49, 6)
    y_dict = {
        '高适应组': 4.5 + 0.5 * np.sin(months/10) + np.random.randn(len(months)) * 0.1,
        '中适应组': 3.5 + 0.3 * np.sin(months/10) + np.random.randn(len(months)) * 0.1,
        '低适应组': 2.5 + 0.2 * np.sin(months/10) + np.random.randn(len(months)) * 0.1
    }
    plot_line_chart(
        months, y_dict,
        xlabel='适应时间（月）',
        ylabel='适应水平',
        title='不同组别的适应轨迹',
        filename='03_line_chart'
    )
    
    # 4. 热力图示例
    variables = ['外向性', '宜人性', '尽责性', '神经质', '开放性']
    corr_matrix = np.random.rand(5, 5)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # 对称化
    np.fill_diagonal(corr_matrix, 1)
    plot_heatmap(
        corr_matrix, variables, variables,
        title='人格维度相关矩阵',
        filename='04_heatmap'
    )
    
    # 5. 箱线图示例
    data_dict = {
        '香港': np.random.normal(3.5, 0.8, 100),
        '法国': np.random.normal(3.8, 0.7, 100),
        '美国': np.random.normal(3.6, 0.9, 100)
    }
    plot_boxplot(
        data_dict,
        xlabel='文化群体',
        ylabel='适应得分',
        title='不同文化群体的适应得分分布',
        filename='05_boxplot'
    )
    
    # 6. 小提琴图示例
    plot_violin(
        data_dict,
        xlabel='文化群体',
        ylabel='适应得分',
        title='不同文化群体的适应得分分布（小提琴图）',
        filename='06_violin'
    )
    
    # 7. 直方图示例
    data = np.random.normal(3.5, 0.8, 500)
    plot_histogram(
        data,
        xlabel='适应得分',
        ylabel='概率密度',
        title='适应得分分布',
        filename='07_histogram'
    )
    
    # 8. 交互效应图示例
    x = np.array([1, 2, 3, 4, 5])
    y1 = np.array([2.0, 2.5, 3.2, 4.0, 4.8])
    y2 = np.array([3.0, 3.2, 3.3, 3.3, 3.2])
    plot_interaction_effect(
        x, y1, y2,
        xlabel='时间点',
        ylabel='适应水平',
        title='开放性与适应时间的交互效应',
        filename='08_interaction',
        label1='高开放性',
        label2='低开放性'
    )
    
    print(f"\n✓ 所有示例图表已生成完毕！")
    print(f"✓ 输出目录: {output_dir.absolute()}")
    print(f"✓ 每个图表包含4种格式: SVG, PDF, EPS, PNG(300dpi)")
    print(f"\n使用建议:")
    print(f"  - SVG: 适合在线查看和编辑（推荐用于PPT）")
    print(f"  - PDF: 适合LaTeX论文插入")
    print(f"  - EPS: 适合某些期刊要求")
    print(f"  - PNG: 高分辨率位图备用")


if __name__ == "__main__":
    generate_example_plots()
