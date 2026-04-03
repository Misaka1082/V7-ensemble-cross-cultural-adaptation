"""
Fix Figure 4.8: France Two-Way Interaction Heatmap
Generate complete 11×11 interaction matrix based on Part4 data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Set publication-quality parameters
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Define all 11 features in V7 model
features = [
    'Cultural\nMaintenance',
    'Family\nSupport', 
    'Communication\nFrequency',
    'Social\nMaintenance',
    'Social\nContact',
    'Social\nConnectedness',
    'Cultural\nContact',
    'Communication\nHonesty',
    'Months in\nFrance',
    'Openness',
    'Autonomy'
]

# Create 11×11 interaction matrix
n = len(features)
france_matrix = np.zeros((n, n))

# Fill in the 9 significant two-way interactions from Part4 Table 14.1
# All interactions are symmetric (β_ij = β_ji)

# Rank 1: Cultural Maintenance × Family Support: β=-0.1447, p=0.0001
france_matrix[0, 1] = france_matrix[1, 0] = -0.1447

# Rank 2: Cultural Maintenance × Communication Frequency: β=-0.1425, p=0.0002
france_matrix[0, 2] = france_matrix[2, 0] = -0.1425

# Rank 3: Cultural Maintenance × Social Maintenance: β=-0.1250, p=0.0015
france_matrix[0, 3] = france_matrix[3, 0] = -0.1250

# Rank 4: Cultural Maintenance × Social Contact: β=-0.1100, p=0.0128
france_matrix[0, 4] = france_matrix[4, 0] = -0.1100

# Rank 5: Cultural Maintenance × Social Connectedness: β=-0.0945, p=0.0134
france_matrix[0, 5] = france_matrix[5, 0] = -0.0945

# Rank 6: Social Maintenance × Family Support: β=-0.0868, p=0.0151
france_matrix[3, 1] = france_matrix[1, 3] = -0.0868

# Rank 7: Months in France × Openness: β=+0.0822, p=0.0330
france_matrix[8, 9] = france_matrix[9, 8] = 0.0822

# Rank 8: Cultural Contact × Communication Frequency: β=+0.0818, p=0.0495
france_matrix[6, 2] = france_matrix[2, 6] = 0.0818

# Rank 9: Social Maintenance × Communication Frequency: β=-0.0712, p=0.0454
france_matrix[3, 2] = france_matrix[2, 3] = -0.0712

# Create figure - match Figure 4.7 style
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap - same style as Figure 4.7
im = ax.imshow(france_matrix, cmap='RdBu_r', aspect='auto',
               vmin=-0.8, vmax=0.8)

# Set ticks and labels
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(features, fontsize=9)
ax.set_yticklabels(features, fontsize=9)

# Rotate x labels - same as Figure 4.7
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add grid lines - white lines between cells like Figure 4.7
for i in range(n + 1):
    ax.axhline(i - 0.5, color='white', linewidth=1.5)
    ax.axvline(i - 0.5, color='white', linewidth=1.5)

# Add text annotations for non-zero values - same style as Figure 4.7
for i in range(n):
    for j in range(n):
        if abs(france_matrix[i, j]) > 0.01:
            text = ax.text(j, i, f'{france_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

# Add title - same style as Figure 4.7
ax.set_title('Figure 9 France Sample Two-way Interaction Effects Heatmap',
            fontsize=14, fontweight='bold', pad=20)

# Add colorbar - same style as Figure 4.7
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
    output_path = f'{output_dir}/Figure_4.8_France_Interaction_Heatmap.{fmt}'
    plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')

plt.close()

# Generate summary statistics
print("\n" + "="*70)
print("FRANCE TWO-WAY INTERACTION MATRIX SUMMARY")
print("="*70)

# Count interactions
non_zero = np.sum(np.abs(france_matrix) > 0.001) // 2  # Divide by 2 for symmetry
print(f"\nTotal significant interactions: {non_zero}")

# Analyze by feature
print("\nInteractions by feature:")
for i, feature in enumerate(features):
    feature_clean = feature.replace('\n', ' ')
    count = np.sum(np.abs(france_matrix[i, :]) > 0.001) - (1 if france_matrix[i, i] != 0 else 0)
    if count > 0:
        interactions = []
        for j in range(n):
            if i != j and abs(france_matrix[i, j]) > 0.001:
                interactions.append(f"{features[j].replace(chr(10), ' ')} (β={france_matrix[i, j]:.3f})")
        print(f"  {feature_clean}: {count} interactions")
        for interaction in interactions:
            print(f"    - {interaction}")

# Direction analysis
positive = np.sum(france_matrix > 0.001) // 2
negative = np.sum(france_matrix < -0.001) // 2
print(f"\nDirection distribution:")
print(f"  Positive interactions: {positive} ({positive/non_zero*100:.1f}%)")
print(f"  Negative interactions: {negative} ({negative/non_zero*100:.1f}%)")

# Strongest interactions
print(f"\nTop 5 strongest interactions:")
flat_indices = np.argsort(np.abs(france_matrix.flatten()))[::-1]
shown = 0
for idx in flat_indices:
    i, j = idx // n, idx % n
    if i < j and abs(france_matrix[i, j]) > 0.001:  # Upper triangle only, non-zero
        print(f"  {shown+1}. {features[i].replace(chr(10), ' ')} × {features[j].replace(chr(10), ' ')}: "
              f"β={france_matrix[i, j]:.4f}")
        shown += 1
        if shown >= 5:
            break

print("\n" + "="*70)
print("Figure 4.8 regeneration complete!")
print("="*70)
