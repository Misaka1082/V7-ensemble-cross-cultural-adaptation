
"""
Generate Model Performance Comparison Bar Chart (Figure 1 / Figure X).
Output: f:/Project/4_1_9_final/academic_figures/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

OUT_DIR = r"f:\Project\4_1_9_final\academic_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'standard',
    'savefig.pad_inches': 0.15,
})

# ── Data ─────────────────────────────────────────────────────────────────────
metrics = ['R²', 'RMSE', 'MAE']

data = {
    'Hong Kong Linear': [0.674, 2.21,  1.72],
    'Hong Kong V7':     [0.753, 2.00,  1.54],
    'France Linear':    [0.661, 2.54,  1.98],
    'France V7':        [0.785, 2.08,  1.64],
}

colors = {
    'Hong Kong Linear': '#2E86AB',
    'Hong Kong V7':     '#6FB1D0',
    'France Linear':    '#D95F4C',
    'France V7':        '#F4A261',
}

# ── Layout ────────────────────────────────────────────────────────────────────
n_metrics  = len(metrics)       # 3
n_groups   = len(data)          # 4 bars per metric
bar_width  = 0.18
group_gap  = 0.10               # extra gap between metric groups

# x positions for the left edge of each metric group
group_width = n_groups * bar_width + group_gap
group_centers = np.arange(n_metrics) * group_width

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor('white')

labels = list(data.keys())
for k, (label, vals) in enumerate(data.items()):
    offsets = (k - (n_groups - 1) / 2) * bar_width
    x_pos   = group_centers + offsets
    bars = ax.bar(x_pos, vals, width=bar_width,
                  color=colors[label], edgecolor='black', linewidth=0.7,
                  label=label, zorder=3)
    # Value labels on top of bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=7.5, color='#333333')

# ── Axes formatting ──────────────────────────────────────────────────────────
ax.set_xticks(group_centers)
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax.set_xlabel('Performance Metric', fontsize=11, fontweight='bold', labelpad=6)
ax.set_ylabel('Value', fontsize=11, fontweight='bold', labelpad=6)
ax.set_xlim(-group_width * 0.6, group_centers[-1] + group_width * 0.6)
ax.set_ylim(0, 3.2)
ax.tick_params(axis='y', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Light grey dashed y-grid
ax.yaxis.grid(True, linestyle='--', color='#CCCCCC', alpha=0.6, zorder=0)
ax.set_axisbelow(True)

# ── Legend above chart ───────────────────────────────────────────────────────
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22),
          ncol=4, fontsize=9.5,
          frameon=False,
          handlelength=1.4, handletextpad=0.5, columnspacing=1.0)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         'Figure 6.  Model Performance Comparison: Linear Regression vs. V7 Ensemble',
         ha='center', va='top',
         fontsize=11, fontweight='bold', color='#111111',
         transform=fig.transFigure)

# ── Figure note ───────────────────────────────────────────────────────────────
note = (
    "Note. R\u00b2 = coefficient of determination; RMSE = root mean square error; "
    "MAE = mean absolute error.\n"
    "Hong Kong sample: N = 75; France sample: N = 249. "
    "The V7 ensemble consistently outperforms linear regression\n"
    "across both samples, with larger improvements in the France sample (17.8% vs. 11.8%)."
)
fig.text(0.5, 0.005, note, ha='center', va='bottom',
         fontsize=9, style='italic', color='#333333',
         transform=fig.transFigure, multialignment='center')

plt.tight_layout(rect=[0, 0.10, 1, 0.90])

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ('svg', 'pdf', 'png'):
    path = os.path.join(OUT_DIR, f'FigureX_Model_Performance_Comparison.{ext}')
    fig.savefig(path, format=ext)
    print(f'  Saved: {path}')
plt.close(fig)
print('\nDone.')
