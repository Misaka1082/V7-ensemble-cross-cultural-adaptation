"""
生成关键非线性关系的可视化图表
1. 开放性的倒U型关系
2. 自主权的U型关系
3. 来港时长的U型关系
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

print("=" * 80)
print("生成非线性关系可视化图表")
print("=" * 80)

# 加载真实数据
df = pd.read_excel('F:/Project/4.1.9.final/data/processed/real_data_filtered_48months.xlsx')

print(f"\n数据加载完成：{len(df)}个样本")

# ============================================================================
# 1. 开放性的倒U型关系
# ============================================================================
print("\n【1/3】生成开放性的倒U型关系图...")

# 定义倒U型函数（二次函数，开口向下）
def inverted_u(x, a, b, c):
    return a * x**2 + b * x + c

# 准备数据
x_openness = df['开放性'].values
y_adaptation = df['跨文化适应程度'].values

# 拟合倒U型曲线
try:
    popt, _ = curve_fit(inverted_u, x_openness, y_adaptation, p0=[-0.1, 1, 3])
    
    # 生成拟合曲线
    x_smooth = np.linspace(x_openness.min(), x_openness.max(), 100)
    y_smooth = inverted_u(x_smooth, *popt)
    
    # 计算最优点
    optimal_x = -popt[1] / (2 * popt[0])
    optimal_y = inverted_u(optimal_x, *popt)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 散点图
    ax.scatter(x_openness, y_adaptation, alpha=0.6, s=100, 
               edgecolors='black', linewidth=0.5, label='真实数据点')
    
    # 拟合曲线
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='倒U型拟合曲线')
    
    # 标注最优点
    ax.plot(optimal_x, optimal_y, 'g*', markersize=20, 
            label=f'最优点 ({optimal_x:.2f}, {optimal_y:.2f})')
    ax.axvline(optimal_x, color='g', linestyle='--', alpha=0.5)
    
    # 添加置信区间阴影
    residuals = y_adaptation - inverted_u(x_openness, *popt)
    std_residuals = np.std(residuals)
    ax.fill_between(x_smooth, 
                     y_smooth - 1.96*std_residuals, 
                     y_smooth + 1.96*std_residuals,
                     alpha=0.2, color='red', label='95%置信区间')
    
    ax.set_xlabel('开放性', fontsize=14, fontweight='bold')
    ax.set_ylabel('跨文化适应程度', fontsize=14, fontweight='bold')
    ax.set_title('开放性与跨文化适应的倒U型关系\n（过高或过低的开放性都不利于适应）', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 添加文本说明
    textstr = f'拟合方程: y = {popt[0]:.3f}x² + {popt[1]:.3f}x + {popt[2]:.3f}\n'
    textstr += f'最优开放性: {optimal_x:.2f}\n'
    textstr += f'R² = {1 - (np.sum(residuals**2) / np.sum((y_adaptation - y_adaptation.mean())**2)):.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('F:/Project/4.1.9.final/results/openness_inverted_u.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 保存: openness_inverted_u.png")
    plt.close()
    
except Exception as e:
    print(f"  警告: 倒U型拟合失败 - {e}")
    print("  使用散点图和局部回归...")
    
    # 使用LOWESS平滑
    from scipy.signal import savgol_filter
    
    # 排序数据
    sorted_indices = np.argsort(x_openness)
    x_sorted = x_openness[sorted_indices]
    y_sorted = y_adaptation[sorted_indices]
    
    # 平滑
    if len(x_sorted) > 5:
        y_smooth = savgol_filter(y_sorted, window_length=min(11, len(x_sorted)//2*2+1), polyorder=2)
    else:
        y_smooth = y_sorted
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x_openness, y_adaptation, alpha=0.6, s=100, 
               edgecolors='black', linewidth=0.5, label='真实数据点')
    ax.plot(x_sorted, y_smooth, 'r-', linewidth=3, label='平滑曲线')
    
    ax.set_xlabel('开放性', fontsize=14, fontweight='bold')
    ax.set_ylabel('跨文化适应程度', fontsize=14, fontweight='bold')
    ax.set_title('开放性与跨文化适应的关系', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/Project/4.1.9.final/results/openness_inverted_u.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 保存: openness_inverted_u.png")
    plt.close()

# ============================================================================
# 2. 自主权的U型关系
# ============================================================================
print("\n【2/3】生成自主权的U型关系图...")

# 定义U型函数（二次函数，开口向上）
def u_shape(x, a, b, c):
    return a * x**2 + b * x + c

# 准备数据
x_autonomy = df['自主权'].values
y_adaptation = df['跨文化适应程度'].values

# 拟合U型曲线
try:
    popt, _ = curve_fit(u_shape, x_autonomy, y_adaptation, p0=[0.1, -1, 5])
    
    # 生成拟合曲线
    x_smooth = np.linspace(x_autonomy.min(), x_autonomy.max(), 100)
    y_smooth = u_shape(x_smooth, *popt)
    
    # 计算最低点
    lowest_x = -popt[1] / (2 * popt[0])
    lowest_y = u_shape(lowest_x, *popt)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 散点图
    ax.scatter(x_autonomy, y_adaptation, alpha=0.6, s=100,
               edgecolors='black', linewidth=0.5, label='真实数据点')
    
    # 拟合曲线
    ax.plot(x_smooth, y_smooth, 'b-', linewidth=3, label='U型拟合曲线')
    
    # 标注最低点
    ax.plot(lowest_x, lowest_y, 'r*', markersize=20,
            label=f'最低点 ({lowest_x:.2f}, {lowest_y:.2f})')
    ax.axvline(lowest_x, color='r', linestyle='--', alpha=0.5)
    
    # 添加置信区间
    residuals = y_adaptation - u_shape(x_autonomy, *popt)
    std_residuals = np.std(residuals)
    ax.fill_between(x_smooth,
                     y_smooth - 1.96*std_residuals,
                     y_smooth + 1.96*std_residuals,
                     alpha=0.2, color='blue', label='95%置信区间')
    
    ax.set_xlabel('自主权', fontsize=14, fontweight='bold')
    ax.set_ylabel('跨文化适应程度', fontsize=14, fontweight='bold')
    ax.set_title('自主权与跨文化适应的U型关系\n（过低或过高的自主权都有利于适应）',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 添加文本说明
    textstr = f'拟合方程: y = {popt[0]:.3f}x² + {popt[1]:.3f}x + {popt[2]:.3f}\n'
    textstr += f'最低自主权: {lowest_x:.2f}\n'
    textstr += f'R² = {1 - (np.sum(residuals**2) / np.sum((y_adaptation - y_adaptation.mean())**2)):.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('F:/Project/4.1.9.final/results/autonomy_u_shape.png',
                dpi=300, bbox_inches='tight')
    print("✓ 保存: autonomy_u_shape.png")
    plt.close()
    
except Exception as e:
    print(f"  警告: U型拟合失败 - {e}")
    print("  使用散点图和局部回归...")
    
    # 使用平滑
    sorted_indices = np.argsort(x_autonomy)
    x_sorted = x_autonomy[sorted_indices]
    y_sorted = y_adaptation[sorted_indices]
    
    if len(x_sorted) > 5:
        y_smooth = savgol_filter(y_sorted, window_length=min(11, len(x_sorted)//2*2+1), polyorder=2)
    else:
        y_smooth = y_sorted
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x_autonomy, y_adaptation, alpha=0.6, s=100,
               edgecolors='black', linewidth=0.5, label='真实数据点')
    ax.plot(x_sorted, y_smooth, 'b-', linewidth=3, label='平滑曲线')
    
    ax.set_xlabel('自主权', fontsize=14, fontweight='bold')
    ax.set_ylabel('跨文化适应程度', fontsize=14, fontweight='bold')
    ax.set_title('自主权与跨文化适应的关系', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/Project/4.1.9.final/results/autonomy_u_shape.png',
                dpi=300, bbox_inches='tight')
    print("✓ 保存: autonomy_u_shape.png")
    plt.close()

# ============================================================================
# 3. 来港时长的U型关系
# ============================================================================
print("\n【3/3】生成来港时长的U型关系图...")

x_months = df['来港时长'].values
y_adaptation = df['跨文化适应程度'].values

# 拟合U型曲线
try:
    popt, _ = curve_fit(u_shape, x_months, y_adaptation, p0=[0.001, -0.1, 5])
    
    # 生成拟合曲线
    x_smooth = np.linspace(x_months.min(), x_months.max(), 100)
    y_smooth = u_shape(x_smooth, *popt)
    
    # 计算最低点
    lowest_x = -popt[1] / (2 * popt[0])
    lowest_y = u_shape(lowest_x, *popt)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 散点图
    ax.scatter(x_months, y_adaptation, alpha=0.6, s=100,
               edgecolors='black', linewidth=0.5, label='真实数据点')
    
    # 拟合曲线
    ax.plot(x_smooth, y_smooth, 'purple', linewidth=3, label='U型拟合曲线')
    
    # 标注最低点
    ax.plot(lowest_x, lowest_y, 'orange', marker='*', markersize=20,
            label=f'适应低谷 ({lowest_x:.1f}个月, {lowest_y:.2f})')
    ax.axvline(lowest_x, color='orange', linestyle='--', alpha=0.5)
    
    # 添加置信区间
    residuals = y_adaptation - u_shape(x_months, *popt)
    std_residuals = np.std(residuals)
    ax.fill_between(x_smooth,
                     y_smooth - 1.96*std_residuals,
                     y_smooth + 1.96*std_residuals,
                     alpha=0.2, color='purple', label='95%置信区间')
    
    # 标注适应阶段
    ax.axvspan(0, lowest_x, alpha=0.1, color='red', label='文化休克期')
    ax.axvspan(lowest_x, x_months.max(), alpha=0.1, color='green', label='适应恢复期')
    
    ax.set_xlabel('来港时长（月）', fontsize=14, fontweight='bold')
    ax.set_ylabel('跨文化适应程度', fontsize=14, fontweight='bold')
    ax.set_title('来港时长与跨文化适应的U型关系\n（文化休克-适应恢复模式）',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 添加文本说明
    textstr = f'拟合方程: y = {popt[0]:.4f}x² + {popt[1]:.3f}x + {popt[2]:.3f}\n'
    textstr += f'适应低谷: {lowest_x:.1f}个月\n'
    textstr += f'R² = {1 - (np.sum(residuals**2) / np.sum((y_adaptation - y_adaptation.mean())**2)):.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('F:/Project/4.1.9.final/results/months_u_shape.png',
                dpi=300, bbox_inches='tight')
    print("✓ 保存: months_u_shape.png")
    plt.close()
    
except Exception as e:
    print(f"  警告: U型拟合失败 - {e}")
    print("  使用散点图和局部回归...")
    
    # 使用平滑
    sorted_indices = np.argsort(x_months)
    x_sorted = x_months[sorted_indices]
    y_sorted = y_adaptation[sorted_indices]
    
    if len(x_sorted) > 5:
        y_smooth = savgol_filter(y_sorted, window_length=min(11, len(x_sorted)//2*2+1), polyorder=2)
    else:
        y_smooth = y_sorted
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x_months, y_adaptation, alpha=0.6, s=100,
               edgecolors='black', linewidth=0.5, label='真实数据点')
    ax.plot(x_sorted, y_smooth, 'purple', linewidth=3, label='平滑曲线')
    
    ax.set_xlabel('来港时长（月）', fontsize=14, fontweight='bold')
    ax.set_ylabel('跨文化适应程度', fontsize=14, fontweight='bold')
    ax.set_title('来港时长与跨文化适应的关系', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('F:/Project/4.1.9.final/results/months_u_shape.png',
                dpi=300, bbox_inches='tight')
    print("✓ 保存: months_u_shape.png")
    plt.close()

# ============================================================================
# 4. 综合对比图
# ============================================================================
print("\n【额外】生成非线性关系综合对比图...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 开放性
ax = axes[0]
ax.scatter(df['开放性'], df['跨文化适应程度'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
z = np.polyfit(df['开放性'], df['跨文化适应程度'], 2)
p = np.poly1d(z)
x_line = np.linspace(df['开放性'].min(), df['开放性'].max(), 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=2)
ax.set_xlabel('开放性', fontsize=12, fontweight='bold')
ax.set_ylabel('跨文化适应程度', fontsize=12, fontweight='bold')
ax.set_title('倒U型关系', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# 自主权
ax = axes[1]
ax.scatter(df['自主权'], df['跨文化适应程度'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
z = np.polyfit(df['自主权'], df['跨文化适应程度'], 2)
p = np.poly1d(z)
x_line = np.linspace(df['自主权'].min(), df['自主权'].max(), 100)
ax.plot(x_line, p(x_line), 'b-', linewidth=2)
ax.set_xlabel('自主权', fontsize=12, fontweight='bold')
ax.set_ylabel('跨文化适应程度', fontsize=12, fontweight='bold')
ax.set_title('U型关系', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# 来港时长
ax = axes[2]
ax.scatter(df['来港时长'], df['跨文化适应程度'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
z = np.polyfit(df['来港时长'], df['跨文化适应程度'], 2)
p = np.poly1d(z)
x_line = np.linspace(df['来港时长'].min(), df['来港时长'].max(), 100)
ax.plot(x_line, p(x_line), 'purple', linewidth=2)
ax.set_xlabel('来港时长（月）', fontsize=12, fontweight='bold')
ax.set_ylabel('跨文化适应程度', fontsize=12, fontweight='bold')
ax.set_title('U型关系（文化休克）', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('跨文化适应的三种非线性关系模式', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('F:/Project/4.1.9.final/results/nonlinear_comparison.png',
            dpi=300, bbox_inches='tight')
print("✓ 保存: nonlinear_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("所有非线性关系图表已生成！")
print("保存位置: F:/Project/4.1.9.final/results/")
print("=" * 80)
