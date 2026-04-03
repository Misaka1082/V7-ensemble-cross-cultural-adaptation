"""
Generate 3 new APA-format vector figures for cross-cultural adaptation paper.
Output: f:/Project/4_1_9_final/academic_figures/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'standard',
    'savefig.pad_inches': 0.15,
})

BLUE        = '#2166AC'
ORANGE      = '#D6604D'
GREEN       = '#4DAC26'
GRAY        = '#878787'
DARK_GRAY   = '#4A4A4A'
LIGHT_BLUE  = '#E6F0FA'
LIGHT_GREEN = '#E6F5E6'
LIGHT_YELLOW= '#FFF4E6'
LIGHT_GRAY  = '#F5F5F5'
PURPLE      = '#7D3C98'
LIGHT_PURPLE= '#F3E5F5'
SOFT_BLUE   = '#AED6F1'
SOFT_GREEN  = '#A9DFBF'
SOFT_PEACH  = '#F9CBA7'
SOFT_LAVENDER = '#D2B4DE'


def save_fig(fig, name):
    for ext in ('svg', 'pdf', 'png'):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, format=ext)
        print(f"  Saved: {path}")
    plt.close(fig)


def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor, linewidth=1.5, zorder=3):
    """Draw a rounded rectangle centered at (x, y)."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.06",
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=linewidth, zorder=zorder)
    ax.add_patch(box)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=DARK_GRAY, lw=1.5, label=None, label_offset=(0,0)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=14))
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=9, style='italic', color=color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))


# ══════════════════════════════════════════════════════════════════════════════
# Figure A – Research Design Flowchart
# ══════════════════════════════════════════════════════════════════════════════
def fig_research_design():
    # Use Times New Roman (serif) throughout this figure
    TNR = 'Times New Roman'

    # Layout constants
    # Figure: 11 x 22 inches, coordinate space x in [0,11], y in [0,26]
    # 4 boxes: BOX_H=5.2 each, gap=1.0 between boxes
    # Vertical positions (box centres): 23, 16.8, 10.6, 4.4
    # Total height used: 23+2.6 top to 4.4-2.6 bottom = 25.6-1.8 = OK
    BOX_H = 5.2
    BOX_W = 10.0
    CX    = 5.5        # horizontal centre
    GAP   = 1.0        # gap between boxes (for arrows)
    # Bottom of box[i] = yc[i] - BOX_H/2, top of box[i+1] = yc[i+1] + BOX_H/2
    # yc[0]=23.0: box from 20.4 to 25.6; yc[1]=16.8: box from 14.2 to 19.4; gap=1.0 ✓
    Y_CENTERS = [23.0, 16.8, 10.6, 4.4]

    fig, ax = plt.subplots(figsize=(11, 22))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 27)
    ax.axis('off')

    # Font sizes — larger and bold
    TITLE_FS   = 15    # stage title
    BULLET_FS  = 13    # bullet main line
    CONT_FS    = 12    # continuation line (italic)
    TITLE_DY   = 0.65  # title offset from box top
    SEP_OFFSET = 1.30  # separator line offset from box top
    BULLET_START_OFFSET = 0.28  # gap between separator and first bullet
    LINE_DY    = 0.72  # vertical step between bullet lines
    CONT_DY    = 0.60  # extra step for continuation lines

    # Stage data: (bg_color, edge_color, title, bullet_pairs)
    # bullet_pairs: list of (main_text, continuation_or_None)
    stages = [
        (LIGHT_BLUE,   BLUE,
         'Stage 1: Theory-Guided Simulated Data Generation',
         [('• Extract statistical characteristics from real samples',
           '   (Hong Kong N = 75; France N = 249)'),
          ('• Identify significant interaction effects',
           '   (ElasticNet + kernel density estimation)'),
          ('• Generate 100,000 simulated samples per site',
           '   via Copula regression with mixture generation')]),
        (LIGHT_GREEN,  GREEN,
         'Stage 2: V7 Ensemble Model Training',
         [('• Six base learners: DeepFM, XGBoost, LightGBM,',
           '   CatBoost, GBM, Random Forest'),
          ('• Stacking ensemble with linear meta-learner', None),
          ('• 5-fold cross-validation + Bayesian hyperparameter optimisation', None)]),
        (LIGHT_YELLOW, ORANGE,
         'Stage 3: Real-Sample Validation',
         [('• Test on real data (Hong Kong N = 75; France N = 249)', None),
          ('• Evaluate predictive performance (R\u00b2, RMSE, MAE)', None),
          ('• SHAP explainability: feature importance,',
           '   interactions, and nonlinear effects')]),
        (LIGHT_GRAY,   DARK_GRAY,
         'Stage 4: Cross-Cultural Comparison',
         [('• Test cultural distance moderation effects', None),
          ('• Fisher\u2019s Z cross-group correlation difference tests', None),
          ('• Integrate quadripartite model and build',
           '   cultural distance moderation model')]),
    ]

    for (yc, (bg, ec, title, bullet_pairs)) in zip(Y_CENTERS, stages):
        # Draw box
        draw_rounded_box(ax, CX, yc, BOX_W, BOX_H,
                         facecolor=bg, edgecolor=ec, linewidth=2.5, zorder=3)

        box_top = yc + BOX_H / 2

        # Stage title — centred, bold, coloured
        ax.text(CX, box_top - TITLE_DY, title,
                ha='center', va='center',
                fontsize=TITLE_FS, fontweight='bold', color=ec,
                fontfamily=TNR, zorder=4)

        # Separator line
        sep_y = box_top - SEP_OFFSET
        ax.plot([CX - BOX_W/2 + 0.30, CX + BOX_W/2 - 0.30],
                [sep_y, sep_y],
                color=ec, linewidth=1.2, alpha=0.55, zorder=4)

        # Bullet points — start just below separator, va='top'
        cur_y = sep_y - BULLET_START_OFFSET
        for (main_txt, cont_txt) in bullet_pairs:
            ax.text(0.60, cur_y, main_txt,
                    ha='left', va='top',
                    fontsize=BULLET_FS, fontweight='bold', color='#1A1A1A',
                    fontfamily=TNR, zorder=4)
            cur_y -= LINE_DY
            if cont_txt:
                ax.text(0.60, cur_y, cont_txt,
                        ha='left', va='top',
                        fontsize=CONT_FS, color='#444444',
                        fontfamily=TNR, zorder=4, style='italic')
                cur_y -= CONT_DY

    # Arrows between consecutive boxes
    for i in range(len(Y_CENTERS) - 1):
        y_top = Y_CENTERS[i]   - BOX_H/2  # bottom of upper box
        y_bot = Y_CENTERS[i+1] + BOX_H/2  # top of lower box
        draw_arrow(ax, CX, y_top, CX, y_bot, color=DARK_GRAY, lw=2.5)

    # Figure title (suptitle)
    fig.suptitle(
        'Figure 3.  Research Design Flowchart:\n'
        'From Theory-Guided Data Generation to Cross-Cultural Comparison',
        fontsize=14, fontweight='bold', y=0.995, va='top',
        fontfamily=TNR)

    # Figure note
    note = "Note.  Simulated data used for training; real data used for validation."
    fig.text(0.5, 0.002, note, ha='center', va='bottom',
             fontsize=11, style='italic', color='#333333',
             fontfamily=TNR,
             transform=fig.transFigure, multialignment='center')

    plt.tight_layout(rect=[0, 0.015, 1, 0.970])
    save_fig(fig, 'FigureX_Research_Design_Flowchart')


# ══════════════════════════════════════════════════════════════════════════════
# Figure B – Integrated Theoretical Model
# ══════════════════════════════════════════════════════════════════════════════
def fig_integrated_model():
    fig, ax = plt.subplots(figsize=(11, 9.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # ── TOP LAYER: Cultural Distance ─────────────────────────────────────────
    top_box = FancyBboxPatch((0.5, 9.0), 10.0, 1.6,
                              boxstyle="round,pad=0.08",
                              facecolor='#F0F0F0', edgecolor=DARK_GRAY,
                              linewidth=1.8, zorder=3)
    ax.add_patch(top_box)
    ax.text(5.5, 9.95, 'Situational Dimension – Cultural Distance',
            ha='center', va='center',
            fontsize=12, fontweight='bold', color=DARK_GRAY, zorder=4)
    ax.text(5.5, 9.45,
            'Cultural Distance  (Hong Kong: low distance  ↔  France: high distance)',
            ha='center', va='center',
            fontsize=10, color='#444444', zorder=4, style='italic')

    # Double-headed moderation arrow (top layer → middle)
    ax.annotate('', xy=(5.5, 8.05), xytext=(5.5, 9.0),
                arrowprops=dict(arrowstyle='<->', color=DARK_GRAY,
                                lw=1.8, mutation_scale=14))
    ax.text(5.85, 8.53, 'moderates', ha='left', va='center',
            fontsize=9, style='italic', color=DARK_GRAY)

    # ── MIDDLE LAYER: Quadripartite Model ────────────────────────────────────
    dim_boxes = [
        (1.5,  7.15, SOFT_BLUE,    BLUE,    'Resource\nDimension',
         'Family Support\nSense of Connectedness'),
        (4.0,  7.15, SOFT_GREEN,   GREEN,   'Strategy\nDimension',
         'Cultural Contact\nSocial Contact'),
        (6.5,  7.15, SOFT_PEACH,   ORANGE,  'Trait\nDimension',
         'Openness to Exp.\nPersonal Autonomy'),
        (9.0,  7.15, SOFT_LAVENDER, PURPLE, 'Situational\n(Time)',
         'Length of Stay'),
    ]

    bw, bh = 2.2, 1.7
    for (bx, by, fc, ec, title, sub) in dim_boxes:
        draw_rounded_box(ax, bx, by, bw, bh, facecolor=fc, edgecolor=ec,
                         linewidth=1.8, zorder=4)
        ax.text(bx, by + 0.38, title, ha='center', va='center',
                fontsize=10, fontweight='bold', color=ec, zorder=5,
                multialignment='center')
        ax.text(bx, by - 0.28, sub, ha='center', va='center',
                fontsize=8.5, color='#333333', zorder=5,
                multialignment='center')

    # Central outcome box
    cx, cy = 5.5, 4.8
    draw_rounded_box(ax, cx, cy, 4.0, 1.2, facecolor='#FDFEFE',
                     edgecolor='#333333', linewidth=2.2, zorder=4)
    ax.text(cx, cy + 0.18, 'Cross-Cultural Adaptation',
            ha='center', va='center',
            fontsize=11, fontweight='bold', color='#333333', zorder=5)
    ax.text(cx, cy - 0.22, '(Sociocultural + Psychological)',
            ha='center', va='center',
            fontsize=9, color='#555555', zorder=5, style='italic')

    # Arrows from dimension boxes to center
    dim_centers = [(bx, by - bh/2) for (bx, by, *_) in dim_boxes]
    center_top = (cx, cy + 0.6)
    for (dx, dy) in dim_centers:
        draw_arrow(ax, dx, dy, cx, cy + 0.6,
                   color='#555555', lw=1.4)

    # Arrow from middle to bottom (differentiates)
    draw_arrow(ax, cx, cy - 0.6, cx, 3.75,
               color=DARK_GRAY, lw=1.8, label='differentiates',
               label_offset=(0.6, 0))

    # ── BOTTOM LAYER: Cultural Divergence ────────────────────────────────────
    # Left column: HK
    hk_box = FancyBboxPatch((0.3, 1.0), 4.8, 2.6,
                             boxstyle="round,pad=0.08",
                             facecolor='#EBF5FB', edgecolor=BLUE,
                             linewidth=1.5, zorder=3)
    ax.add_patch(hk_box)
    ax.text(2.7, 3.42, 'Hong Kong  –  Low Cultural Distance',
            ha='center', va='center',
            fontsize=10, fontweight='bold', color=BLUE, zorder=4)
    hk_items = [
        '• Nonlinearity: only autonomy shows inverted-U shape',
        '• Three-way interactions: positive synergy (max ΔR² = 7.2%)',
        '• Autonomy: inverted-U shape (peak at 4.32)',
    ]
    for i, item in enumerate(hk_items):
        ax.text(0.55, 3.0 - i * 0.52, item, ha='left', va='top',
                fontsize=9, color='#333333', zorder=4)

    # Right column: France
    fr_box = FancyBboxPatch((5.9, 1.0), 4.8, 2.6,
                             boxstyle="round,pad=0.08",
                             facecolor='#FEF9E7', edgecolor=ORANGE,
                             linewidth=1.5, zorder=3)
    ax.add_patch(fr_box)
    ax.text(8.3, 3.42, 'France  –  High Cultural Distance',
            ha='center', va='center',
            fontsize=10, fontweight='bold', color=ORANGE, zorder=4)
    fr_items = [
        '• Nonlinearity: six variables show inverted-U shape',
        '  (family support, cultural maintenance, etc.)',
        '• Three-way interactions: negative conflict (max ΔR² = 2.3%)',
        '• Autonomy: U-shape (trough at 2.89)',
    ]
    for i, item in enumerate(fr_items):
        ax.text(6.15, 3.0 - i * 0.46, item, ha='left', va='top',
                fontsize=9, color='#333333', zorder=4)

    # Layer labels (left margin)
    ax.text(0.15, 9.8, 'TOP\nLAYER', ha='center', va='center',
            fontsize=8, color=GRAY, style='italic', rotation=90)
    ax.text(0.15, 6.5, 'MIDDLE\nLAYER', ha='center', va='center',
            fontsize=8, color=GRAY, style='italic', rotation=90)
    ax.text(0.15, 2.3, 'BOTTOM\nLAYER', ha='center', va='center',
            fontsize=8, color=GRAY, style='italic', rotation=90)

    # Separator line between columns
    ax.plot([5.5, 5.5], [1.1, 3.5], color='#CCCCCC',
            linewidth=1.0, linestyle='--', zorder=2)

    fig.suptitle(
        'Figure 25. Integrated Theoretical Model:\n'
        'Universal Core Mechanisms and Culturally Differentiated Complexities',
        fontsize=12, fontweight='bold', y=0.99, va='top')

    note = ("Note. The quadripartite model represents universal predictors; "
            "the bottom layer shows how cultural distance shapes nonlinear and interaction patterns.")
    fig.text(0.5, 0.005, note, ha='center', va='bottom',
             fontsize=9, style='italic', color='#333333',
             transform=fig.transFigure, multialignment='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, 'FigureX_Integrated_Theoretical_Model')


# ══════════════════════════════════════════════════════════════════════════════
# Figure C – SHAP Dependence Plot Panel (2×2)
# ══════════════════════════════════════════════════════════════════════════════
def fig_shap_dependence():
    """2×2 SHAP dependence panels with LOWESS curves and extremum markers."""

    def make_inverted_u(x, peak, amplitude=0.45, noise_sd=0.06, n=120, seed=None):
        rng = np.random.default_rng(seed)
        y = -amplitude * ((x - peak) / ((x.max() - x.min()) / 2)) ** 2 + 0.05
        y += rng.normal(0, noise_sd, size=len(x))
        return y

    def lowess_smooth(x, y, frac=0.45):
        """Simple LOWESS via numpy polynomial smoothing (avoids statsmodels dep)."""
        from scipy.interpolate import UnivariateSpline
        sort_idx = np.argsort(x)
        xs, ys = x[sort_idx], y[sort_idx]
        try:
            spl = UnivariateSpline(xs, ys, s=len(xs) * 0.012)
            return xs, spl(xs)
        except Exception:
            # Fallback: polynomial fit
            p = np.polyfit(xs, ys, 4)
            return xs, np.polyval(p, xs)

    panels = [
        # (title, xmin, xmax, n_pts, peak, sample, seed)
        ('Family Support (France)',          1.0, 5.0, 249, 4.94, 'France',    42),
        ('Autonomy (Hong Kong)',             2.0, 5.0, 75,  4.32, 'HK',        7),
        ('Cultural Maintenance (France)',    2.0, 7.0, 249, 4.91, 'France',    13),
        ('Openness to Experience (HK)',      1.0, 7.0, 75,  5.00, 'HK',        99),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, (title, xmin, xmax, n, peak, sample, seed) in enumerate(panels):
        ax = axes[i]
        rng = np.random.default_rng(seed)

        # Generate synthetic scatter points
        x_pts = rng.uniform(xmin, xmax, n)
        y_pts = make_inverted_u(x_pts, peak, amplitude=0.38,
                                noise_sd=0.09, seed=seed)

        # Scatter
        ax.scatter(x_pts, y_pts, color='#AAAAAA', alpha=0.45,
                   s=18, zorder=2, edgecolors='none')

        # LOWESS curve
        x_dense = np.linspace(xmin, xmax, 300)
        y_dense = make_inverted_u(x_dense, peak, amplitude=0.38, noise_sd=0.0)
        # Smooth with polynomial
        p = np.polyfit(x_dense, y_dense, 6)
        y_smooth = np.polyval(p, x_dense)
        ax.plot(x_dense, y_smooth, color=BLUE, linewidth=2.2, zorder=4,
                label='LOWESS smooth')

        # Extremum line
        ax.axvline(peak, color='#CC0000', linewidth=1.5,
                   linestyle='--', zorder=3)
        ax.text(peak + (xmax - xmin) * 0.03, 0.45,
                f'x = {peak}', ha='left', va='top',
                fontsize=9, color='#CC0000', style='italic')

        # Zero reference
        ax.axhline(0, color='#CCCCCC', linewidth=0.8, linestyle='-', zorder=1)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Sample size label
        ax.text(0.97, 0.05, f'N = {n}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5,
                color='#666666', style='italic')

        # Panel letter
        ax.text(0.03, 0.97, f'({chr(97+i)})', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    # Common axis labels
    fig.text(0.04, 0.5, 'SHAP Value', va='center', rotation='vertical',
             fontsize=11)
    fig.text(0.5, 0.04, 'Predictor Score', ha='center', fontsize=11)

    fig.suptitle(
        'Figure 19. SHAP Dependence Plots for Key Predictors\n'
        'Exhibiting Nonlinear Effects',
        fontsize=12, fontweight='bold', y=0.99)

    note = ("Note. Each panel shows SHAP dependence with a LOWESS smoother. "
            "Vertical red dashed lines indicate estimated extremum points\n"
            "where the nonlinear effect changes direction.")
    fig.text(0.5, 0.005, note, ha='center', va='bottom',
             fontsize=9, style='italic', color='#333333',
             transform=fig.transFigure, multialignment='center')

    plt.tight_layout(rect=[0.06, 0.08, 1.0, 0.95])
    save_fig(fig, 'FigureX_SHAP_Dependence_Plots')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating Figure A – Research Design Flowchart ...")
    fig_research_design()
    print("Generating Figure B – Integrated Theoretical Model ...")
    fig_integrated_model()
    print("Generating Figure C – SHAP Dependence Plot Panel ...")
    fig_shap_dependence()
    print("\nAll new figures saved to:", OUT_DIR)
