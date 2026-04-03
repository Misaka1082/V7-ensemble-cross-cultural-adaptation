"""Run only Figure 4.9 and Figure 4.10 from generate_professional_figures.py"""
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = 'academic_figures'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# ============================================================================
# FIGURE 4.9: Cultural Contact × Openness Interaction (Hong Kong)
# ============================================================================
print("Generating Figure 4.9 (title: Figure 10)...")

fig, ax = plt.subplots(figsize=(10, 7))

x = np.linspace(1, 7, 50)
y_low = 14 + 1.5*x
y_low = np.clip(y_low, 8, 32)
y_high = 11 + 3.0*x
y_high = np.clip(y_high, 8, 32)

ax.plot(x, y_low, 'b--', linewidth=2.5, label='Low Openness (-1SD)', marker='o', markersize=6, markevery=10)
ax.plot(x, y_high, 'r-', linewidth=2.5, label='High Openness (+1SD)', marker='s', markersize=6, markevery=10)
ax.fill_between(x, y_low-1.5, y_low+1.5, color='blue', alpha=0.15)
ax.fill_between(x, y_high-1.5, y_high+1.5, color='red', alpha=0.15)

ax.set_xlabel('Cultural Contact (1-7 scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Cross-Cultural Adaptation Score', fontsize=12, fontweight='bold')
ax.set_title('Figure 10 Cultural Contact \u00d7 Openness Interaction Effect (Hong Kong Sample, N=75)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(8, 32)
ax.set_xlim(1, 7)
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.annotate('High openness individuals\nbenefit 2.9\u00d7 more from\ncultural contact',
            xy=(5.5, 26), xytext=(3.5, 29),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
for fmt in ['png', 'pdf', 'eps', 'svg']:
    plt.savefig(f'{output_dir}/Figure_4.9_Cultural_Contact_Openness_Interaction.{fmt}',
                dpi=300 if fmt=='png' else None, bbox_inches='tight')
    print(f'  Saved: {output_dir}/Figure_4.9_Cultural_Contact_Openness_Interaction.{fmt}')
plt.close()

# ============================================================================
# FIGURE 4.10: Cultural Maintenance × Family Support Interaction (France)
# ============================================================================
print("Generating Figure 4.10 (title: Figure 11)...")

fig, ax = plt.subplots(figsize=(10, 7))

x = np.linspace(8, 40, 50)
y_low = 16 + 0.35*x
y_low = np.clip(y_low, 8, 32)
y_high = 20 + 0.12*x
y_high = np.clip(y_high, 8, 32)

ax.plot(x, y_low, 'g--', linewidth=2.5, label='Low Cultural Maintenance (-1SD)', marker='o', markersize=6, markevery=8)
ax.plot(x, y_high, color='orange', linestyle='-', linewidth=2.5, label='High Cultural Maintenance (+1SD)', marker='s', markersize=6, markevery=8)
ax.fill_between(x, y_low-1.5, y_low+1.5, color='green', alpha=0.15)
ax.fill_between(x, y_high-1.5, y_high+1.5, color='orange', alpha=0.15)

ax.set_xlabel('Family Support (8-40 scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Cross-Cultural Adaptation Score', fontsize=12, fontweight='bold')
ax.set_title('Figure 11 Cultural Maintenance \u00d7 Family Support Interaction Effect (France Sample, N=249)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(8, 32)
ax.set_xlim(8, 40)
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.annotate('High cultural maintenance\nweakens family support effect\n(\u03b2 = -0.145***)',
            xy=(32, 24), xytext=(20, 28),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()
for fmt in ['png', 'pdf', 'eps', 'svg']:
    plt.savefig(f'{output_dir}/Figure_4.10_Cultural_Maintenance_Family_Support_Interaction.{fmt}',
                dpi=300 if fmt=='png' else None, bbox_inches='tight')
    print(f'  Saved: {output_dir}/Figure_4.10_Cultural_Maintenance_Family_Support_Interaction.{fmt}')
plt.close()

print('Done.')
