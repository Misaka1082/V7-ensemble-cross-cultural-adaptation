"""
Generate all academic figures in ENGLISH for psychology journal publication
High-resolution vector graphics (PNG 300dpi, PDF, EPS, SVG)
Figures 4.3.2, 4.4, 4.5, 4.6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# Set English font and style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-paper')

# Create output directory
output_dir = 'academic_figures'
os.makedirs(output_dir, exist_ok=True)

# Feature name mapping (English)
feature_names_en = {
    'social_connectedness': 'Social Connectedness',
    'social_contact': 'Social Contact',
    'cultural_contact': 'Cultural Contact',
    'openness': 'Openness',
    'family_support': 'Family Support',
    'cultural_maintenance': 'Cultural Maintenance',
    'months_in_hk': 'Months in HK',
    'months_in_france': 'Months in France',
    'family_communication_frequency': 'Communication Frequency',
    'communication_honesty': 'Communication Honesty',
    'social_maintenance': 'Social Maintenance',
    'autonomy': 'Autonomy'
}

print("="*80)
print("GENERATING FIGURE 4.3.2: V7 Model Prediction Scatter Plot")
print("="*80)

# Read data
hk_data = pd.read_csv('results/final_robust/real_data_predictions.csv')
hk_data = hk_data.head(75)  # N=75
hk_true = hk_data['真实跨文化适应得分'].values
hk_pred = hk_data['预测跨文化适应得分'].values

france_data = pd.read_csv('france_models/predictions.csv')
france_true = france_data['真实值'].values
france_pred = france_data['V7预测'].values

print(f"Hong Kong sample: N={len(hk_true)}")
print(f"France sample: N={len(france_true)}")

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

hk_r2, hk_rmse, hk_mae = calculate_metrics(hk_true, hk_pred)
france_r2, france_rmse, france_mae = calculate_metrics(france_true, france_pred)

print(f"HK: R²={hk_r2:.3f}, RMSE={hk_rmse:.2f}, MAE={hk_mae:.2f}")
print(f"France: R²={france_r2:.3f}, RMSE={france_rmse:.2f}, MAE={france_mae:.2f}")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def plot_scatter(ax, y_true, y_pred, r2, rmse, mae, n, title, color):
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color=color, edgecolors='white', linewidth=0.5)
    
    # Diagonal line (y=x ideal prediction)
    min_val, max_val = 8, 32
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect Prediction (y=x)')
    
    # Linear fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    x_fit = np.array([min_val, max_val])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', linewidth=1.5, label='Linear Fit')
    
    # 95% confidence interval
    predict = slope * y_true + intercept
    residuals = y_pred - predict
    std_residuals = np.std(residuals)
    
    x_dense = np.linspace(min_val, max_val, 100)
    y_dense = slope * x_dense + intercept
    ci = 1.96 * std_residuals
    ax.fill_between(x_dense, y_dense - ci, y_dense + ci, 
                     color='red', alpha=0.15, label='95% CI')
    
    # Axes settings
    ax.set_xlim(8, 32)
    ax.set_ylim(8, 32)
    ax.set_xlabel('Actual Cross-Cultural Adaptation Score', fontsize=11)
    ax.set_ylabel('Predicted Cross-Cultural Adaptation Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Statistics text box
    textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nN = {n}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # Ticks
    ax.set_xticks(range(8, 33, 4))
    ax.set_yticks(range(8, 33, 4))
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

# Plot Hong Kong sample
plot_scatter(axes[0], hk_true, hk_pred, hk_r2, hk_rmse, hk_mae, 75, 
             'Hong Kong Sample (N=75)', 'steelblue')

# Plot France sample
plot_scatter(axes[1], france_true, france_pred, france_r2, france_rmse, france_mae, 249,
             'France Sample (N=249)', 'seagreen')

plt.tight_layout()

# Save in multiple formats
for fmt in ['png', 'pdf', 'eps', 'svg']:
    output_path = f'{output_dir}/Figure_4.3.2_V7_Prediction_Scatter.{fmt}'
    if fmt == 'png':
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_path, format=fmt, bbox_inches='tight')
    print(f"Saved: {output_path}")

plt.close()

print("\n" + "="*80)
print("GENERATING FIGURE 4.4: Hong Kong Feature Importance")
print("="*80)

# Read Hong Kong SHAP data
hk_shap = pd.read_csv('results/feature_importance_75samples_cv.csv')
hk_shap['Feature_EN'] = hk_shap['特征'].map(feature_names_en)
hk_shap = hk_shap.sort_values('SHAP重要性', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(hk_shap)))
bars = ax.barh(range(len(hk_shap)), hk_shap['SHAP重要性'], color=colors, edgecolor='black', linewidth=0.5)

# Add value labels
for i, (idx, row) in enumerate(hk_shap.iterrows()):
    ax.text(row['SHAP重要性'] + 0.02, i, f"{row['SHAP重要性']:.3f}", 
            va='center', fontsize=9)

ax.set_yticks(range(len(hk_shap)))
ax.set_yticklabels(hk_shap['Feature_EN'], fontsize=10)
ax.set_xlabel('SHAP Importance', fontsize=11)
ax.set_title('Figure 4.4 Hong Kong Sample V7 Model Feature Importance (N=75)', 
             fontsize=12, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, 0.8)

plt.tight_layout()
for fmt in ['png', 'pdf', 'eps', 'svg']:
    output_path = f'{output_dir}/Figure_4.4_HK_Feature_Importance.{fmt}'
    if fmt == 'png':
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_path, format=fmt, bbox_inches='tight')
    print(f"Saved: {output_path}")
plt.close()

print("\n" + "="*80)
print("GENERATING FIGURE 4.5: France Feature Importance")
print("="*80)

# Read France SHAP data
france_shap = pd.read_csv('france_models/feature_importance_shap.csv')
france_shap['Feature_EN'] = france_shap['特征'].map(feature_names_en)
france_shap = france_shap.sort_values('SHAP重要性', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(france_shap)))
bars = ax.barh(range(len(france_shap)), france_shap['SHAP重要性'], color=colors, edgecolor='black', linewidth=0.5)

# Add value labels
for i, (idx, row) in enumerate(france_shap.iterrows()):
    ax.text(row['SHAP重要性'] + 0.02, i, f"{row['SHAP重要性']:.3f}", 
            va='center', fontsize=9)

ax.set_yticks(range(len(france_shap)))
ax.set_yticklabels(france_shap['Feature_EN'], fontsize=10)
ax.set_xlabel('SHAP Importance', fontsize=11)
ax.set_title('Figure 4.5 France Sample V7 Model Feature Importance (N=249)', 
             fontsize=12, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, 1.0)

plt.tight_layout()
for fmt in ['png', 'pdf', 'eps', 'svg']:
    output_path = f'{output_dir}/Figure_4.5_France_Feature_Importance.{fmt}'
    if fmt == 'png':
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_path, format=fmt, bbox_inches='tight')
    print(f"Saved: {output_path}")
plt.close()

print("\n" + "="*80)
print("GENERATING FIGURE 4.6: Feature Importance Cross-Cultural Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 8))

# Prepare comparison data
hk_dict = dict(zip(hk_shap['特征'], hk_shap['SHAP重要性']))
france_dict = dict(zip(france_shap['特征'], france_shap['SHAP重要性']))

# Get all features
all_features = list(set(hk_dict.keys()) | set(france_dict.keys()))
all_features = [f for f in all_features if f in feature_names_en]

# Prepare plot data
hk_values = [hk_dict.get(f, 0) for f in all_features]
france_values = [france_dict.get(f, 0) for f in all_features]
feature_labels = [feature_names_en[f] for f in all_features]

# Sort by Hong Kong importance
sorted_indices = np.argsort(hk_values)[::-1]
hk_values = [hk_values[i] for i in sorted_indices]
france_values = [france_values[i] for i in sorted_indices]
feature_labels = [feature_labels[i] for i in sorted_indices]

x = np.arange(len(feature_labels))
width = 0.35

bars1 = ax.bar(x - width/2, hk_values, width, label='Hong Kong (N=75)', 
               color='steelblue', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, france_values, width, label='France (N=249)', 
               color='seagreen', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Features', fontsize=11)
ax.set_ylabel('SHAP Importance', fontsize=11)
ax.set_title('Figure 4.6 V7 Model Feature Importance Cross-Cultural Comparison', 
             fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

plt.tight_layout()
for fmt in ['png', 'pdf', 'eps', 'svg']:
    output_path = f'{output_dir}/Figure_4.6_Feature_Importance_Comparison.{fmt}'
    if fmt == 'png':
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_path, format=fmt, bbox_inches='tight')
    print(f"Saved: {output_path}")
plt.close()

print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*80)
print(f"\nAll figures saved to: {output_dir}/")
print("Formats: PNG (300dpi), PDF, EPS, SVG")
print("Language: English")
print("Style: Psychology journal publication quality")
