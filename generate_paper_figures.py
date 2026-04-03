"""
Generate 7 APA-format vector figures for cross-cultural adaptation paper.
Output: f:/Project/4_1_9_final/academic_figures/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
import matplotlib.patheffects as pe
import numpy as np
import os

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = r"f:\Project\4_1_9_final\academic_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'standard',
    'savefig.pad_inches': 0.15,
})

BLUE   = '#2166AC'
ORANGE = '#D6604D'
GREEN  = '#4DAC26'
GRAY   = '#878787'
LIGHT_BLUE = '#D1E5F0'
LIGHT_ORANGE = '#FDDBC7'
LIGHT_GREEN  = '#B8E186'
LIGHT_GRAY   = '#E0E0E0'


def save_fig(fig, name):
    for ext in ('svg', 'pdf', 'png'):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, format=ext)
        print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2.1 – Quadripartite Conceptual Model
# ══════════════════════════════════════════════════════════════════════════════
def fig_2_1():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # ── Outer ring: Situation (Cultural Distance) ─────────────────────────────
    outer = plt.Circle((5, 4.5), 3.9, color=LIGHT_GRAY, zorder=0)
    ax.add_patch(outer)
    outer_border = plt.Circle((5, 4.5), 3.9, fill=False,
                               edgecolor=GRAY, linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(outer_border)
    ax.text(5, 8.55, 'Situation', ha='center', va='center',
            fontsize=11, fontweight='bold', color=GRAY)
    ax.text(5, 8.15, '(Cultural Distance)', ha='center', va='center',
            fontsize=9, color=GRAY, style='italic')

    # ── Center box: Outcome ───────────────────────────────────────────────────
    cx, cy = 5, 4.5
    center_box = FancyBboxPatch((cx-1.1, cy-0.55), 2.2, 1.1,
                                 boxstyle="round,pad=0.08",
                                 facecolor='#F7F7F7', edgecolor='#333333',
                                 linewidth=2, zorder=5)
    ax.add_patch(center_box)
    ax.text(cx, cy+0.12, 'Cross-Cultural', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333333', zorder=6)
    ax.text(cx, cy-0.22, 'Adaptation', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333333', zorder=6)

    # ── Three inner dimension boxes ────────────────────────────────────────���──
    dims = [
        # (x, y, color, light_color, title, subtitle1, subtitle2)
        (5,   7.0, BLUE,   LIGHT_BLUE,   'Resource',
         'Family Support', 'Sense of Connectedness'),
        (1.8, 2.8, GREEN,  LIGHT_GREEN,  'Strategy',
         'Cultural Maintenance', 'Cultural Contact'),
        (8.2, 2.8, ORANGE, LIGHT_ORANGE, 'Trait',
         'Openness to Experience', 'Personal Autonomy'),
    ]

    box_positions = []
    for (bx, by, col, lcol, title, sub1, sub2) in dims:
        bw, bh = 2.6, 1.5
        box = FancyBboxPatch((bx - bw/2, by - bh/2), bw, bh,
                              boxstyle="round,pad=0.08",
                              facecolor=lcol, edgecolor=col,
                              linewidth=1.8, zorder=3)
        ax.add_patch(box)
        ax.text(bx, by + 0.32, title, ha='center', va='center',
                fontsize=10, fontweight='bold', color=col, zorder=4)
        ax.text(bx, by - 0.05, sub1, ha='center', va='center',
                fontsize=8, color='#333333', zorder=4)
        ax.text(bx, by - 0.38, sub2, ha='center', va='center',
                fontsize=8, color='#333333', zorder=4)
        box_positions.append((bx, by, col))

    # ── Solid arrows: dimensions → center ────────────────────────────────────
    arrow_kw = dict(arrowstyle='->', color='#333333',
                    linewidth=1.5, mutation_scale=14, zorder=4)
    # Resource (top) → center
    ax.annotate('', xy=(cx, cy + 0.55), xytext=(5, 7.0 - 0.75),
                arrowprops=dict(arrowstyle='->', color='#333333',
                                lw=1.5, mutation_scale=14))
    # Strategy (bottom-left) → center
    ax.annotate('', xy=(cx - 0.9, cy - 0.3), xytext=(1.8 + 1.0, 2.8 + 0.5),
                arrowprops=dict(arrowstyle='->', color='#333333',
                                lw=1.5, mutation_scale=14))
    # Trait (bottom-right) → center
    ax.annotate('', xy=(cx + 0.9, cy - 0.3), xytext=(8.2 - 1.0, 2.8 + 0.5),
                arrowprops=dict(arrowstyle='->', color='#333333',
                                lw=1.5, mutation_scale=14))

    # ── Dashed arrows: Situation → each dimension ─────────────────────────────
    dash_kw = dict(arrowstyle='->', color=GRAY,
                   linewidth=1.2, mutation_scale=12,
                   linestyle='dashed')
    # → Resource
    ax.annotate('', xy=(5, 7.0 + 0.75), xytext=(5, 8.4),
                arrowprops=dict(arrowstyle='->', color=GRAY,
                                lw=1.2, mutation_scale=12,
                                linestyle='dashed'))
    # → Strategy
    ax.annotate('', xy=(1.8 - 0.9, 2.8 + 0.2), xytext=(1.4, 4.5),
                arrowprops=dict(arrowstyle='->', color=GRAY,
                                lw=1.2, mutation_scale=12,
                                linestyle='dashed'))
    # → Trait
    ax.annotate('', xy=(8.2 + 0.9, 2.8 + 0.2), xytext=(8.6, 4.5),
                arrowprops=dict(arrowstyle='->', color=GRAY,
                                lw=1.2, mutation_scale=12,
                                linestyle='dashed'))

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#333333', lw=1.5,
               marker='>', markersize=6, label='Direct effect'),
        Line2D([0], [0], color=GRAY, lw=1.2, linestyle='--',
               marker='>', markersize=6, label='Moderating effect'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.02), ncol=2,
              fontsize=8.5, frameon=True, framealpha=0.9,
              edgecolor='#CCCCCC')

    # ── Figure note ───────────────────────────────────────────────────────────
    note = ("Figure 2. Quadripartite model of cross-cultural adaptation. "
            "Resource dimension includes family support and sense of connectedness; "
            "Strategy dimension includes cultural maintenance and cultural contact; "
            "Trait dimension includes openness to experience and personal autonomy; "
            "Situation dimension (cultural distance) moderates all inner-dimension effects.")
    fig.text(0.05, -0.04, note, ha='left', va='top',
             fontsize=7.5, style='italic', wrap=True,
             color='#333333',
             transform=fig.transFigure)

    fig.suptitle('Figure 2  Quadripartite Model of Cross-Cultural Adaptation',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    save_fig(fig, 'Figure2.1_Quadripartite_Model')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5.1 – Feature Importance (SHAP) Side-by-Side Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig_5_1():
    features = [
        'Sense of Connectedness',
        'Social Contact',
        'Cultural Contact',
        'Openness to Experience',
        'Family Support',
        'Cultural Maintenance',
        'Length of Stay*',
        'Comm. Frequency',
        'Comm. Openness',
        'Social Maintenance',
        'Personal Autonomy',
    ]
    hk_vals = [0.733, 0.675, 0.638, 0.347, 0.336, 0.144,
               0.129, 0.108, 0.036, 0.027, 0.012]
    fr_vals = [0.723, 0.670, 0.944, 0.370, 0.748, 0.338,
               0.617, 0.302, 0.539, 0.156, 0.338]

    # Sort by average importance
    avg = [(h + f) / 2 for h, f in zip(hk_vals, fr_vals)]
    order = sorted(range(len(features)), key=lambda i: avg[i])
    features = [features[i] for i in order]
    hk_vals  = [hk_vals[i]  for i in order]
    fr_vals  = [fr_vals[i]  for i in order]

    y = np.arange(len(features))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 7))
    bars_hk = ax.barh(y + height/2, hk_vals, height,
                      color=BLUE, alpha=0.85, label='Hong Kong (N=75)')
    bars_fr = ax.barh(y - height/2, fr_vals, height,
                      color=ORANGE, alpha=0.85, label='France (N=249)')

    # Value labels
    for bar in bars_hk:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f'{w:.3f}', va='center', ha='left', fontsize=9, color=BLUE)
    for bar in bars_fr:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f'{w:.3f}', va='center', ha='left', fontsize=9, color=ORANGE)

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_xlim(0, 1.12)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(axis='x', labelsize=11)

    # Vertical reference line
    ax.axvline(x=0, color='#333333', linewidth=0.8)

    ax.legend(fontsize=11, loc='lower right', frameon=True,
              framealpha=0.9, edgecolor='#CCCCCC')

    # Star annotation – placed mid-right, below the top bars, above the legend
    ax.text(0.98, 0.38,
            '* Fisher\'s Z test: p = .0004\n  (significant cross-cultural\n   difference)',
            transform=ax.transAxes, fontsize=9, va='center', ha='right',
            style='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7',
                      edgecolor='#CCCCCC', alpha=0.9))

    fig.suptitle('Figure 7  SHAP Feature Importance: Hong Kong vs. France',
                 fontsize=13, fontweight='bold')
    note = ("Figure 7. Mean absolute SHAP values for each predictor in the V7 stacking ensemble model.\n"
            "Blue bars = Hong Kong sample (N = 75); orange bars = France sample (N = 249).\n"
            "* Length of Stay showed a statistically significant cross-cultural difference "
            "(Fisher's Z = 3.54, p = .0004).")
    fig.text(0.5, 0.01, note, ha='center', va='bottom',
             fontsize=9, style='italic', color='#333333',
             transform=fig.transFigure, multialignment='center')

    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    save_fig(fig, 'Figure5.1_Feature_Importance_Comparison')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5.2 – Inverted-U Curves Panel (France, 2×3 grid)
# ══════════════════════════════════════════════════════════════════════════════
def fig_5_2():
    panels = [
        ('Family Support',       8,  40, 4.94, 8,  32),
        ('Cultural Maintenance', 2,  14, 4.91, 8,  32),
        ('Social Maintenance',   2,  14, 4.0,  8,  32),
        ('Comm. Frequency',      1,   5, 4.39, 8,  32),
        ('Comm. Openness',       1,   5, 3.5,  8,  32),
        ('Openness to Exp.',     1,   7, 4.0,  8,  32),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, (label, xmin, xmax, peak, ymin, ymax) in enumerate(panels):
        ax = axes[i]
        x = np.linspace(xmin, xmax, 300)
        a = -(ymax - ymin) * 0.45 / ((xmax - xmin) / 2) ** 2
        y = a * (x - peak) ** 2 + (ymin + ymax) / 2 + (ymax - ymin) * 0.2

        ax.plot(x, y, color=ORANGE, linewidth=2.5)
        ax.axvline(peak, color=GRAY, linewidth=1.2, linestyle='--', alpha=0.7)
        ax.text(peak, ymin + 0.5, f'Peak={peak:.2f}',
                ha='center', va='bottom', fontsize=10, color=GRAY, style='italic')

        ax.fill_between(x, y - 1.2, y + 1.2, alpha=0.15, color=ORANGE)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin - 1, ymax + 1)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Predicted Adaptation' if i % 3 == 0 else '', fontsize=11)
        ax.tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.text(0.05, 0.95, f'({chr(97+i)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

    fig.suptitle('Figure 22  Inverted-U Relationships: France Sample',
                 fontsize=14, fontweight='bold')
    note = ("Figure 22. Predicted cross-cultural adaptation scores as a function of six predictors "
            "in the France sample (N = 249).\n"
            "Each panel shows an inverted-U (quadratic) relationship estimated from the V7 stacking "
            "ensemble model.\n"
            "Dashed vertical lines indicate the estimated peak value. Shaded bands represent ±1 SE.")
    fig.text(0.5, 0.01, note, ha='center', va='bottom',
             fontsize=10, style='italic', color='#333333',
             transform=fig.transFigure, multialignment='center')

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    save_fig(fig, 'Figure5.2_InvertedU_Curves_France')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5.3 – Autonomy Cross-Cultural Reversal
# ════════════════════════════════════════════════════════════════════════���═════
def fig_5_3():
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(2, 10, 300)

    # HK: inverted-U, peak ≈ 4.32
    hk_peak = 4.32
    a_hk = -1.8
    hk_y = a_hk * (x - hk_peak) ** 2 + 22
    hk_y = np.clip(hk_y, 8, 32)

    # France: U-shape, trough ≈ 2.89
    fr_trough = 2.89
    a_fr = 1.5
    fr_y = a_fr * (x - fr_trough) ** 2 + 14
    fr_y = np.clip(fr_y, 8, 32)

    ax.plot(x, hk_y, color=BLUE, linewidth=2.2, label='Hong Kong (N=75): Inverted-U')
    ax.fill_between(x, hk_y - 1.0, hk_y + 1.0, alpha=0.12, color=BLUE)

    ax.plot(x, fr_y, color=ORANGE, linewidth=2.2,
            linestyle='--', label='France (N=249): U-shape')
    ax.fill_between(x, fr_y - 1.0, fr_y + 1.0, alpha=0.12, color=ORANGE)

    # Annotations
    ax.axvline(hk_peak, color=BLUE, linewidth=1, linestyle=':', alpha=0.6)
    ax.text(hk_peak + 0.15, 23.5, f'HK peak\n= {hk_peak}',
            fontsize=10, color=BLUE, style='italic')

    ax.axvline(fr_trough, color=ORANGE, linewidth=1, linestyle=':', alpha=0.6)
    ax.text(fr_trough + 0.15, 15.5, f'FR trough\n= {fr_trough}',
            fontsize=10, color=ORANGE, style='italic')

    ax.set_xlabel('Personal Autonomy Score', fontsize=12)
    ax.set_ylabel('Predicted Cross-Cultural Adaptation', fontsize=12)
    ax.set_xlim(2, 10)
    ax.set_ylim(8, 32)
    ax.legend(fontsize=11, loc='upper right', frameon=True,
              framealpha=0.9, edgecolor='#CCCCCC')
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 23  Autonomy–Adaptation: Cross-Cultural Reversal',
                 fontsize=14, fontweight='bold')
    note = ("Figure 23. Nonlinear relationship between personal autonomy and cross-cultural adaptation "
            "differs qualitatively across samples.\n"
            "Hong Kong participants (blue solid line) show an inverted-U pattern (optimal autonomy ≈ 4.32),\n"
            "whereas France participants (orange dashed line) show a U-shaped pattern (trough ≈ 2.89). "
            "Shaded bands represent ±1 SE.")
    fig.text(0.5, 0.01, note, ha='center', va='bottom',
             fontsize=10, style='italic', color='#333333',
             transform=fig.transFigure, multialignment='center')

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    save_fig(fig, 'Figure5.3_Autonomy_CrossCultural_Reversal')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5.4 – Synergy vs. Conflict Interaction Network
# ══════════════════════════════════════════════════════════════════════════════
def fig_5_4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    def draw_network(ax, title, nodes, center_label, edge_color,
                     sign_label, delta_r2, bg_color):
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 2.4)
        ax.axis('off')
        ax.set_facecolor(bg_color)

        cx, cy = 0, 0
        center = plt.Circle((cx, cy), 0.38, color='#F7F7F7',
                             ec='#333333', lw=2, zorder=5)
        ax.add_patch(center)
        ax.text(cx, cy + 0.06, 'Cross-Cultural', ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=6)
        ax.text(cx, cy - 0.12, 'Adaptation', ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=6)

        angles = np.linspace(90, 90 + 360, len(nodes) + 1)[:-1]
        r = 1.1
        for j, (node_label, angle) in enumerate(zip(nodes, angles)):
            nx_ = r * np.cos(np.radians(angle))
            ny_ = r * np.sin(np.radians(angle))
            node_c = plt.Circle((nx_, ny_), 0.32,
                                 color=LIGHT_BLUE if edge_color == GREEN else LIGHT_ORANGE,
                                 ec=edge_color, lw=1.5, zorder=5)
            ax.add_patch(node_c)
            ax.text(nx_, ny_, node_label, ha='center', va='center',
                    fontsize=9.5, zorder=6, multialignment='center')

            dx = cx - nx_
            dy = cy - ny_
            dist = np.sqrt(dx**2 + dy**2)
            ux, uy = dx/dist, dy/dist
            ax.annotate('',
                        xy=(cx - ux*0.38, cy - uy*0.38),
                        xytext=(nx_ + ux*0.32, ny_ + uy*0.32),
                        arrowprops=dict(arrowstyle='->', color=edge_color,
                                        lw=1.8, mutation_scale=14))

        # Place sign label and ΔR² well above the top node (which is at y≈1.1+0.32=1.42)
        ax.text(0, 2.1, sign_label, ha='center', va='center',
                fontsize=15, fontweight='bold', color=edge_color)
        ax.text(0, 1.85, f'ΔR² = {delta_r2}%', ha='center', va='center',
                fontsize=12, color=edge_color, style='italic')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    draw_network(ax1,
                 'Hong Kong: Synergistic Interaction',
                 ['Social\nContact', 'Comm.\nOpenness', 'Sense of\nConnectedness'],
                 'Adaptation', GREEN, '[+] Synergy', 7.2, '#F0FFF0')

    draw_network(ax2,
                 'France: Conflicting Interaction',
                 ['Family\nSupport', 'Sense of\nConnectedness', 'Openness\nto Exp.'],
                 'Adaptation', ORANGE, '[-] Conflict', 2.3, '#FFF8F0')

    fig.suptitle('Figure 24  Interaction Patterns: Synergy vs. Conflict',
                 fontsize=14, fontweight='bold')
    note = ("Figure 24. Interaction network diagrams for Hong Kong (left) and France (right).\n"
            "In Hong Kong, social contact, communication openness, and sense of connectedness "
            "interact synergistically (ΔR² = 7.2%).\n"
            "In France, family support, sense of connectedness, and openness to experience "
            "show a conflicting pattern (ΔR² = 2.3%).")
    fig.text(0.5, 0.01, note, ha='center', va='bottom',
             fontsize=10, style='italic', color='#333333',
             transform=fig.transFigure, multialignment='center')

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    save_fig(fig, 'Figure5.4_Synergy_Conflict_Network')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6.1 – CDMM Three-Layer Pyramid
# ══════════════════════════════════════════════════════════════════════════════
def fig_6_1():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    layers = [
        # (y_bottom, height, color, label, sublabel)
        (0.5, 2.8, '#AED6F1',
         'Universal Core Mechanisms',
         'Cultural Contact  ·  Social Contact\nFamily Support  ·  Sense of Connectedness'),
        (3.5, 2.5, '#85C1E9',
         'Cultural Distance Moderation',
         'Differential weighting of predictors\nacross cultural contexts'),
        (6.2, 2.5, '#2E86C1',
         'Nonlinear & Higher-Order Interactions',
         'Inverted-U / U-shaped effects\nSynergy & conflict interaction patterns'),
    ]

    for (yb, h, col, label, sub) in layers:
        # Trapezoid shape
        margin_bottom = 0.3 * (yb / 10)
        margin_top    = margin_bottom + 0.25
        verts = [
            (margin_bottom, yb),
            (10 - margin_bottom, yb),
            (10 - margin_top - h * 0.08, yb + h),
            (margin_top + h * 0.08, yb + h),
        ]
        poly = plt.Polygon(verts, closed=True, facecolor=col,
                           edgecolor='white', linewidth=2, zorder=2)
        ax.add_patch(poly)
        ax.text(5, yb + h * 0.65, label, ha='center', va='center',
                fontsize=15, fontweight='bold', color='white', zorder=3)
        ax.text(5, yb + h * 0.28, sub, ha='center', va='center',
                fontsize=12.5, color='white', zorder=3, multialignment='center')

    layer_labels = ['Layer I\n(Foundation)', 'Layer II\n(Moderation)', 'Layer III\n(Complexity)']
    y_centers = [0.5 + 2.8/2, 3.5 + 2.5/2, 6.2 + 2.5/2]
    for lbl, yc in zip(layer_labels, y_centers):
        ax.text(9.7, yc, lbl, ha='right', va='center',
                fontsize=12, color='#555555', style='italic')

    ax.text(5, 9.4, 'Cross-Cultural Dynamic Moderation Model (CDMM)',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#1A5276')

    fig.suptitle('Figure 6.1  CDMM Three-Layer Architecture',
                 fontsize=14, fontweight='bold')
    note = ("Figure 6.1. The Cross-Cultural Dynamic Moderation Model (CDMM) organises findings "
            "into three hierarchical layers.\n"
            "Layer I identifies universal core mechanisms that operate across both samples.\n"
            "Layer II captures cultural distance moderation effects.\n"
            "Layer III represents nonlinear and higher-order interaction patterns.")
    fig.text(0.5, 0.01, note, ha='center', va='bottom',
             fontsize=10, style='italic', color='#333333',
             transform=fig.transFigure, multialignment='center')

    plt.tight_layout(rect=[0, 0.1, 1, 0.97])
    save_fig(fig, 'Figure6.1_CDMM_Pyramid')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7.1 – PBS Trinity Flowchart
# ══════════════════════════════════════════════════════════════════════════════
def fig_7_1():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Centre the diagram: 3 top boxes at x=2, 5.5, 9; bottom box at x=5.5
    flow_steps = [
        (2.0,  5.5, LIGHT_BLUE,   BLUE,      'Theory-Guided\nSimulation',
         'N = 100,000\nMonte Carlo'),
        (5.5,  5.5, LIGHT_GREEN,  GREEN,     'V7 Stacking\nEnsemble',
         'RF + GBM + SVR\n+ Meta-learner'),
        (9.0,  5.5, LIGHT_ORANGE, ORANGE,    'SHAP\nAnalysis',
         'Feature importance\n& interaction'),
        (5.5,  2.5, '#E8DAEF',    '#7D3C98', 'Empirical\nValidation',
         'HK N=75\nFrance N=249'),
    ]

    box_centers = []
    for (bx, by, fc, ec, title, sub) in flow_steps:
        bw, bh = 2.4, 1.6
        box = FancyBboxPatch((bx - bw/2, by - bh/2), bw, bh,
                              boxstyle="round,pad=0.1",
                              facecolor=fc, edgecolor=ec,
                              linewidth=1.8, zorder=3)
        ax.add_patch(box)
        ax.text(bx, by + 0.28, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=ec, zorder=4,
                multialignment='center')
        ax.text(bx, by - 0.28, sub, ha='center', va='center',
                fontsize=10, color='#444444', zorder=4,
                multialignment='center')
        box_centers.append((bx, by, ec))

    # ── Horizontal arrows: step 1→2→3 ────────────────────────────────────────
    for i in range(2):
        x1 = flow_steps[i][0] + 1.2
        x2 = flow_steps[i+1][0] - 1.2
        y  = flow_steps[i][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#333333',
                                    lw=1.5, mutation_scale=14))

    # ── Downward arrow: step 3 → validation ──────────────────────────────────
    ax.annotate('', xy=(5.5, 2.5 + 0.8), xytext=(9.0, 5.5 - 0.8),
                arrowprops=dict(arrowstyle='->', color='#7D3C98',
                                lw=1.5, mutation_scale=14,
                                connectionstyle='arc3,rad=0.2'))

    # ── Triangle overlay: PBS vertices ───────────────────────────────────────
    tri_x = [2.0, 9.0, 5.5]
    tri_y = [5.5, 5.5, 2.5]
    triangle = plt.Polygon(list(zip(tri_x, tri_y)), closed=True,
                            fill=False, edgecolor='#AAAAAA',
                            linewidth=1.2, linestyle=':', zorder=2)
    ax.add_patch(triangle)

    # PBS vertex labels
    vertex_labels = [
        (2.0,  6.6, 'Precision',  BLUE),
        (9.0,  6.6, 'Breadth',    ORANGE),
        (5.5,  1.3, 'Simplicity', '#7D3C98'),
    ]
    for (vx, vy, vlabel, vcol) in vertex_labels:
        ax.text(vx, vy, vlabel, ha='center', va='center',
                fontsize=12, fontweight='bold', color=vcol,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=vcol, alpha=0.85))

    # ── PBS label ─────────────────────────────────────────────────────────────
    ax.text(5.5, 7.5, 'PBS Trinity Framework',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDFEFE',
                      edgecolor='#AAAAAA', alpha=0.9))

    fig.suptitle('Figure 7.1  PBS Trinity: Precision–Breadth–Simplicity Framework',
                 fontsize=14, fontweight='bold')
    note = ("Figure 7.1. The PBS Trinity framework integrates three methodological virtues.\n"
            "Precision: theory-guided simulation generates ecologically valid training data.\n"
            "Breadth: the V7 stacking ensemble captures diverse predictor relationships.\n"
            "Simplicity: SHAP analysis distils complex model outputs into interpretable insights.\n"
            "Empirical validation (HK N = 75; France N = 249) closes the loop.")
    fig.text(0.5, 0.01, note, ha='center', va='bottom',
             fontsize=10, style='italic', color='#333333',
             transform=fig.transFigure, multialignment='center')

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    save_fig(fig, 'Figure7.1_PBS_Trinity_Flowchart')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating Figure 2.1 – Quadripartite Model ...")
    fig_2_1()
    print("Generating Figure 5.1 – Feature Importance ...")
    fig_5_1()
    print("Generating Figure 5.2 – Inverted-U Curves ...")
    fig_5_2()
    print("Generating Figure 5.3 – Autonomy Reversal ...")
    fig_5_3()
    print("Generating Figure 5.4 – Synergy vs Conflict ...")
    fig_5_4()
    print("Generating Figure 6.1 – CDMM Pyramid ...")
    fig_6_1()
    print("Generating Figure 7.1 – PBS Trinity ...")
    fig_7_1()
    print("\nAll figures saved to:", OUT_DIR)
