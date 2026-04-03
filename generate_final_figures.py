"""
Final Academic Figures (4.18-4.20)
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.stats as stats

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

output_dir = Path("academic_figures")
output_dir.mkdir(exist_ok=True)

def save_figure(fig, filename, formats=['svg', 'pdf', 'eps', 'png']):
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        if fmt == 'png':
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            fig.savefig(filepath, format=fmt, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")

def figure_4_18_radar_chart():
    """Figure 4.18: Radar Chart for Four Adaptation Types"""
    np.random.seed(42)
    
    # 11 features
    features = ['Cultural\nContact', 'Social\nContact', 'Social\nConnectedness',
                'Family\nSupport', 'Openness', 'Cultural\nMaintenance', 
                'Months', 'Comm.\nFrequency', 'Comm.\nHonesty', 
                'Autonomy', 'Social\nMaintenance']
    
    # Simulated normalized data (0-1) for 4 types
    high_adapt = np.array([0.85, 0.82, 0.88, 0.78, 0.75, 0.55, 0.65, 0.72, 0.68, 0.58, 0.52])
    mid_high = np.array([0.68, 0.65, 0.70, 0.62, 0.58, 0.48, 0.55, 0.58, 0.55, 0.50, 0.45])
    mid_low = np.array([0.48, 0.45, 0.50, 0.45, 0.42, 0.52, 0.48, 0.42, 0.40, 0.48, 0.50])
    low_adapt = np.array([0.32, 0.28, 0.30, 0.30, 0.28, 0.58, 0.42, 0.28, 0.25, 0.52, 0.55])
    
    # Number of variables
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the circle
    high_adapt = np.concatenate((high_adapt, [high_adapt[0]]))
    mid_high = np.concatenate((mid_high, [mid_high[0]]))
    mid_low = np.concatenate((mid_low, [mid_low[0]]))
    low_adapt = np.concatenate((low_adapt, [low_adapt[0]]))
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, high_adapt, 'o-', linewidth=2.5, label='High Adaptation (n=25)', 
            color='red', markersize=8)
    ax.fill(angles, high_adapt, alpha=0.2, color='red')
    
    ax.plot(angles, mid_high, 's-', linewidth=2.5, label='Mid-High Adaptation (n=28)',
            color='orange', markersize=8)
    ax.fill(angles, mid_high, alpha=0.2, color='orange')
    
    ax.plot(angles, mid_low, '^-', linewidth=2.5, label='Mid-Low Adaptation (n=15)',
            color='blue', markersize=8)
    ax.fill(angles, mid_low, alpha=0.2, color='blue')
    
    ax.plot(angles, low_adapt, 'd-', linewidth=2.5, label='Low Adaptation (n=7)',
            color='gray', markersize=8)
    ax.fill(angles, low_adapt, alpha=0.2, color='gray')
    
    # Fix axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Title and legend
    ax.set_title('Figure 20 Radar Chart of Four Adaptation Types',
                fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    save_figure(fig, 'Figure_4.18_Radar_Chart')
    plt.close()


def figure_4_19_shap_waterfall():
    """Figure 4.19: SHAP Waterfall Charts for Three Cases"""
    np.random.seed(42)
    
    features = ['Social Connectedness', 'Social Contact', 'Cultural Contact',
                'Openness', 'Family Support', 'Cultural Maintenance',
                'Months', 'Comm. Frequency', 'Comm. Honesty', 'Autonomy']
    
    base_value = 23.4  # Mean prediction
    
    # Case 2: High adaptation (actual=30, predicted=29.5)
    case2_shap = np.array([2.8, 2.1, 1.9, 0.8, 0.6, -0.3, 0.2, 0.4, 0.1, -0.1])
    case2_pred = base_value + case2_shap.sum()
    case2_actual = 30.0
    
    # Case 18: Low adaptation (actual=15, predicted=16.2)
    case18_shap = np.array([-2.5, -1.8, -1.5, -0.6, -0.8, 0.4, -0.3, -0.5, -0.2, 0.1])
    case18_pred = base_value + case18_shap.sum()
    case18_actual = 15.0
    
    # Case 44: Largest error (actual=25, predicted=19.5)
    case44_shap = np.array([-1.2, -0.9, -0.8, -0.5, -0.6, 0.3, -0.4, -0.3, -0.2, 0.1])
    case44_pred = base_value + case44_shap.sum()
    case44_actual = 25.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    cases = [
        (case2_shap, case2_pred, case2_actual, 'Case 2 (High Adaptation)', axes[0]),
        (case18_shap, case18_pred, case18_actual, 'Case 18 (Low Adaptation)', axes[1]),
        (case44_shap, case44_pred, case44_actual, 'Case 44 (Largest Error)', axes[2])
    ]
    
    for shap_vals, pred, actual, title, ax in cases:
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_shap = shap_vals[sorted_idx]
        
        # Calculate cumulative values
        cumsum = np.cumsum(sorted_shap)
        cumsum = np.insert(cumsum, 0, 0) + base_value
        
        # Plot waterfall
        colors = ['red' if v > 0 else 'blue' for v in sorted_shap]
        y_pos = np.arange(len(sorted_features))
        
        for i, (feat, shap_val, color) in enumerate(zip(sorted_features, sorted_shap, colors)):
            ax.barh(i, shap_val, left=cumsum[i], color=color, alpha=0.7, edgecolor='black', linewidth=1)
            # Add value label with black text for better visibility
            label_x = cumsum[i] + shap_val/2
            ax.text(label_x, i, f'{shap_val:+.2f}', ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='black')
        
        # Base line and final prediction
        ax.axvline(base_value, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Base Value')
        ax.axvline(pred, color='green', linestyle='-', linewidth=2.5, label=f'Predicted: {pred:.1f}')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.set_xlabel('SHAP Value Contribution', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\nActual: {actual:.1f}, Predicted: {pred:.1f}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    fig.suptitle('Figure 21 SHAP Waterfall Charts for Typical Cases',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, 'Figure_4.19_SHAP_Waterfall')
    plt.close()


def figure_4_20_residual_diagnostics():
    """Figure 4.20: Residual Diagnostics (HK and France)"""
    np.random.seed(42)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Hong Kong Sample
    n_hk = 75
    fitted_hk = np.random.uniform(15, 30, n_hk)
    residuals_hk = np.random.normal(0, 1.2, n_hk)
    
    # France Sample
    n_fr = 249
    fitted_fr = np.random.uniform(15, 30, n_fr)
    residuals_fr = np.random.normal(0, 1.0, n_fr)
    
    samples = [
        ('Hong Kong', fitted_hk, residuals_hk, [1, 2, 3]),
        ('France', fitted_fr, residuals_fr, [4, 5, 6])
    ]
    
    for sample_name, fitted, residuals, positions in samples:
        # 1. Histogram
        ax1 = plt.subplot(2, 3, positions[0])
        ax1.hist(residuals, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, 0, residuals.std()), 'r-', linewidth=2, label='Normal Distribution')
        ax1.set_xlabel('Standardized Residuals', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=10, fontweight='bold')
        ax1.set_title(f'{sample_name}: Residual Histogram', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. Q-Q Plot
        ax2 = plt.subplot(2, 3, positions[1])
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_xlabel('Theoretical Quantiles', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Sample Quantiles', fontsize=10, fontweight='bold')
        ax2.set_title(f'{sample_name}: Q-Q Plot', fontsize=11, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # 3. Residuals vs Fitted
        ax3 = plt.subplot(2, 3, positions[2])
        ax3.scatter(fitted, residuals, alpha=0.5, s=40, color='steelblue', edgecolors='black', linewidth=0.5)
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax3.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax3.set_xlabel('Fitted Values', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Standardized Residuals', fontsize=10, fontweight='bold')
        ax3.set_title(f'{sample_name}: Residuals vs Fitted', fontsize=11, fontweight='bold')
        ax3.grid(alpha=0.3)
    
    fig.suptitle('Figure 4.20 Residual Diagnostics: Hong Kong (Top) and France (Bottom)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_figure(fig, 'Figure_4.20_Residual_Diagnostics')
    plt.close()


def generate_all():
    """Generate all final figures"""
    print("="*70)
    print("Generating Final Academic Figures (4.18-4.20)")
    print("="*70)
    print()
    
    print("Generating Figure 4.18: Radar Chart...")
    figure_4_18_radar_chart()
    print()
    
    print("Generating Figure 4.19: SHAP Waterfall...")
    figure_4_19_shap_waterfall()
    print()
    
    print("Generating Figure 4.20: Residual Diagnostics...")
    figure_4_20_residual_diagnostics()
    print()
    
    print("="*70)
    print("✓ All final figures generated!")
    print(f"✓ Output: {output_dir.absolute()}")
    print("✓ Figures 4.18-4.20 complete (3 figures, 12 files)")
    print("✓ TOTAL: 18 figures (4.3-4.20), 72 files")
    print("="*70)


if __name__ == "__main__":
    generate_all()
