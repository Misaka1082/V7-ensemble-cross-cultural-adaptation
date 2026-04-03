"""
Generate APA-formatted English variable tables for HK and France samples.
Outputs SVG and PDF vector graphics.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

output_dir = r"f:\Project\4_1_9_final\academic_figures"
os.makedirs(output_dir, exist_ok=True)

# ── Table data ──────────────────────────────────────────────────────────────
hk_data = [
    ("months_in_hk",       "Length of Stay in HK",        "Continuous", "0–48",  "Demographic information"),
    ("family_support",     "Family Support",               "Continuous", "8–40",  "Family Support Scale"),
    ("social_connection",  "Sense of Connectedness",       "Continuous", "3–18",  "Social Connectedness Scale"),
    ("cultural_maintenance","Cultural Maintenance",        "Continuous", "2–14",  "Acculturation Strategy Scale"),
    ("social_maintenance", "Social Maintenance",           "Continuous", "2–14",  "Acculturation Strategy Scale"),
    ("cultural_contact",   "Cultural Contact",             "Continuous", "2–14",  "Acculturation Strategy Scale"),
    ("social_contact",     "Social Contact",               "Continuous", "2–14",  "Acculturation Strategy Scale"),
    ("openness",           "Openness to Experience",       "Continuous", "1–7",   "Openness Scale"),
    ("autonomy",           "Personal Autonomy",            "Continuous", "2–10",  "Autonomy Scale"),
    ("comm_frequency",     "Communication Frequency",      "Continuous", "1–5",   "Single-item measure"),
    ("comm_openness",      "Communication Openness",       "Continuous", "1–5",   "Single-item measure"),
]

fr_data = [
    ("months_in_fr",       "Length of Stay in France",     "Continuous", "0–48",  "Demographic information"),
    ("family_support",     "Family Support",               "Continuous", "8–40",  "Family Support Scale"),
    ("social_connection",  "Sense of Connectedness",       "Continuous", "3–18",  "Social Connectedness Scale"),
    ("cultural_maintenance","Cultural Maintenance",        "Continuous", "2–14",  "Acculturation Strategy Scale (FR)"),
    ("social_maintenance", "Social Maintenance",           "Continuous", "2–14",  "Acculturation Strategy Scale (FR)"),
    ("cultural_contact",   "Cultural Contact",             "Continuous", "2–14",  "Acculturation Strategy Scale (FR)"),
    ("social_contact",     "Social Contact",               "Continuous", "2–14",  "Acculturation Strategy Scale (FR)"),
    ("openness",           "Openness to Experience",       "Continuous", "1–7",   "Openness Scale"),
    ("autonomy",           "Personal Autonomy",            "Continuous", "2–10",  "Autonomy Scale"),
    ("comm_frequency",     "Communication Frequency",      "Continuous", "1–5",   "Single-item measure"),
    ("comm_openness",      "Communication Openness",       "Continuous", "1–5",   "Single-item measure"),
]

headers = ["Variable Name", "Definition", "Type", "Range", "Measurement Source"]
col_widths = [0.18, 0.24, 0.12, 0.10, 0.36]   # proportions (sum=1)

def draw_table(data, title, note, filename_stem):
    n_rows = len(data)
    n_cols = len(headers)
    row_h = 0.38          # inches per row
    header_h = 0.45
    top_margin = 0.55     # for title
    bottom_margin = 0.65  # for note
    fig_w = 10.0
    fig_h = top_margin + header_h + n_rows * row_h + bottom_margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    # ── APA title ──
    ax.text(0.0, fig_h - 0.18, title,
            fontsize=11, fontfamily='Arial', fontstyle='italic',
            va='top', ha='left', transform=ax.transData)

    # ── top rule (thick) ──
    y_top = fig_h - top_margin
    ax.axhline(y=y_top, xmin=0, xmax=1, color='black', linewidth=1.5)

    # ── header row ──
    x = 0.0
    for i, (h, w) in enumerate(zip(headers, col_widths)):
        ax.text(x + 0.01, y_top - header_h / 2, h,
                fontsize=9.5, fontfamily='Arial', fontweight='bold',
                va='center', ha='left')
        x += w

    # ── mid rule (thin) ──
    y_mid = y_top - header_h
    ax.axhline(y=y_mid, xmin=0, xmax=1, color='black', linewidth=0.8)

    # ── data rows ──
    for r, row in enumerate(data):
        y_row = y_mid - (r + 0.5) * row_h
        bg_color = '#F7F7F7' if r % 2 == 0 else 'white'
        rect = FancyBboxPatch((0, y_mid - (r + 1) * row_h), 1, row_h,
                              boxstyle="square,pad=0", linewidth=0,
                              facecolor=bg_color)
        ax.add_patch(rect)
        x = 0.0
        for val, w in zip(row, col_widths):
            ax.text(x + 0.01, y_row, val,
                    fontsize=8.5, fontfamily='Arial',
                    va='center', ha='left')
            x += w

    # ── bottom rule (thick) ──
    y_bot = y_mid - n_rows * row_h
    ax.axhline(y=y_bot, xmin=0, xmax=1, color='black', linewidth=1.5)

    # ── note ──
    ax.text(0.0, y_bot - 0.12,
            f"Note. {note}",
            fontsize=8, fontfamily='Arial', fontstyle='italic',
            va='top', ha='left', wrap=True)

    fig.tight_layout(pad=0)
    for ext in ('svg', 'pdf', 'png'):
        path = os.path.join(output_dir, f"{filename_stem}.{ext}")
        dpi = 300 if ext == 'png' else None
        fig.savefig(path, format=ext, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close(fig)


# ── Table 1: Hong Kong ───────────────────────────────────────────────────────
draw_table(
    hk_data,
    title="Table 1\nVariable Definitions and Measurement Sources: Hong Kong Sample (N = 75)",
    note=("All variables are continuous. months_in_hk = length of stay in Hong Kong (months); "
          "family_support = perceived family support; social_connection = sense of social connectedness; "
          "cultural_maintenance / social_maintenance = acculturation strategy subscales; "
          "cultural_contact / social_contact = contact with host culture subscales; "
          "openness = openness to experience (Big Five); autonomy = personal autonomy; "
          "comm_frequency / comm_openness = single-item communication measures."),
    filename_stem="Table1_HK_Variables_English"
)

# ── Table 2: France ──────────────────────────────────────────────────────────
draw_table(
    fr_data,
    title="Table 2\nVariable Definitions and Measurement Sources: France Sample (N = 249)",
    note=("All variables are continuous. months_in_fr = length of stay in France (months); "
          "family_support = perceived family support; social_connection = sense of social connectedness; "
          "cultural_maintenance / social_maintenance / cultural_contact / social_contact = "
          "French version of the Acculturation Strategy Scale; "
          "openness = openness to experience (Big Five); autonomy = personal autonomy; "
          "comm_frequency / comm_openness = single-item communication measures."),
    filename_stem="Table2_France_Variables_English"
)

print("Done.")
