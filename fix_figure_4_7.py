"""
Fix Figure 4.7: Hong Kong Two-Way Interaction Heatmap
Add white grid lines and fix note position
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set publication-quality parameters
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 11
rcParams['svg.fonttype'] = 'none'
rcParams['pdf.fonttype'] = 42

# Variables in order
vars_en = ['Social\nConnectedness', 'Family\nSupport', 'Cultural\nContact', 
           'Social\nContact', 'Openness', 'Cultural\nMaintenance', 
           'Social\nMaintenance', 'Comm.\nFrequency', 'Comm.\nHonesty', 
           'Autonomy', 'Months\nin HK']

# Create interaction matrix (11x11) based on two_way_interactions.csv
n = len(vars_en)
interaction_matrix = np.zeros((n, n))

# Key interactions from the data (standardized coefficients)
interactions = [
    (2, 4, 0.643),   # Cultural Contact × Openness
    (2, 7, -0.635),  # Cultural Contact × Comm Frequency
    (1, 4, 0.721),   # Family Support × Openness
    (2, 8, -0.548),  # Cultural Contact × Comm Honesty
    (7, 9, -0.527),  # Comm Frequency × Autonomy
    (5, 0, -0.498),  # Cultural Maintenance × Social Connectedness
    (2, 10, -0.587), # Cultural Contact × Months
    (7, 10, -0.575), # Comm Frequency × Months
    (3, 9, -0.486),  # Social Contact × Autonomy
    (8, 9, -0.334),  # Comm Honesty × Autonomy
    (9, 4, 0.452),   # Autonomy × Openness
    (5, 9, -0.322),  # Cultural Maintenance × Autonomy
    (5, 4, 0.399),   # Cultural Maintenance × Openness
    (3, 0, 0.439),   # Social Contact × Social Connectedness
    (3, 10, -0.539), # Social Contact × Months
]

for i, j, coef in interactions:
    interaction_matrix[i, j] = coef
    interaction_matrix[j, i] = coef  # Symmetric

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap
im = ax.imshow(interaction_matrix, cmap='RdBu_r', aspect='auto', 
               vmin=-0.8, vmax=0.8)

# Set ticks
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(vars_en, fontsize=9)
ax.set_yticklabels(vars_en, fontsize=9)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add white grid lines between cells
for i in range(n + 1):
    ax.axhline(i - 0.5, color='white', linewidth=1.5)
    ax.axvline(i - 0.5, color='white', linewidth=1.5)

# Add text annotations for non-zero values
for i in range(n):
    for j in range(n):
        if abs(interaction_matrix[i, j]) > 0.01:
            text = ax.text(j, i, f'{interaction_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

ax.set_title('Figure 8 Hong Kong Sample Two-way Interaction Effects Heatmap',
            fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Standardized Coefficient (β)', rotation=270, labelpad=20, fontsize=11)

# Adjust layout to prevent label overlap
plt.tight_layout(rect=[0, 0.03, 1, 1])

# Add note - positioned to avoid label overlap
fig.text(0.5, 0.01, 'Note: Red = positive interaction, Blue = negative interaction',
        ha='center', fontsize=9, style='italic')

# Save in multiple formats
output_dir = 'academic_figures'
import os
os.makedirs(output_dir, exist_ok=True)

formats = ['png', 'pdf', 'eps', 'svg']
for fmt in formats:
    output_path = f'{output_dir}/Figure_4.7_HK_Interaction_Heatmap.{fmt}'
    if fmt == 'png':
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    else:
        plt.savefig(output_path, format=fmt, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'✓ Saved: {output_path}')

plt.close()

print("\n" + "="*70)
print("Figure 4.7 regeneration complete with white grid lines!")
print("="*70)
