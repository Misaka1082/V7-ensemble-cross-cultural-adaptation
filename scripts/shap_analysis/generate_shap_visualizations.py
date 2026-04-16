"""Generate SHAP visualizations for V7 ensemble model."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path


def plot_summary(shap_values, X, output_path=None, max_display=11, show=False):
    """
    Beeswarm summary plot of SHAP values.

    Parameters
    ----------
    shap_values : np.ndarray  shape (n_samples, n_features)
    X : pd.DataFrame
    output_path : str or Path, optional
    max_display : int
    show : bool  Whether to call plt.show()
    """
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to {output_path}")
    if show:
        plt.show()
    plt.close()


def plot_bar(shap_values, X, output_path=None, max_display=11, show=False):
    """Bar chart of mean absolute SHAP values (feature importance)."""
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, plot_type='bar',
                      max_display=max_display, show=False)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Bar plot saved to {output_path}")
    if show:
        plt.show()
    plt.close()


def plot_waterfall(shap_values, explainer, X, sample_idx=0,
                   output_path=None, show=False):
    """Waterfall plot for a single prediction."""
    plt.figure(figsize=(10, 6))
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X.iloc[sample_idx].values,
        feature_names=list(X.columns)
    )
    # shap.plots.waterfall is the current API (shap >= 0.40);
    # fall back to legacy shap.waterfall_plot for older versions
    if hasattr(shap.plots, 'waterfall'):
        shap.plots.waterfall(explanation, show=False)
    else:
        shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Waterfall plot saved to {output_path}")
    if show:
        plt.show()
    plt.close()


def plot_dependence(shap_values, X, feature, interaction_feature='auto',
                    output_path=None, show=False):
    """Dependence plot for a single feature."""
    feature_names = list(X.columns)
    feat_idx = feature_names.index(feature) if isinstance(feature, str) else feature

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx, shap_values, X,
        interaction_index=interaction_feature,
        show=False
    )
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dependence plot saved to {output_path}")
    if show:
        plt.show()
    plt.close()


def generate_all_shap_plots(shap_values, X, explainer=None,
                             output_dir='results/shap_plots/', show=False):
    """
    Generate the full set of standard SHAP plots.

    Parameters
    ----------
    shap_values : np.ndarray
    X : pd.DataFrame
    explainer : shap.Explainer, optional  Required for waterfall plot.
    output_dir : str
    show : bool
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_summary(shap_values, X, out / 'shap_summary.png', show=show)
    plot_bar(shap_values, X, out / 'shap_bar.png', show=show)

    if explainer is not None:
        plot_waterfall(shap_values, explainer, X, sample_idx=0,
                       output_path=out / 'shap_waterfall.png', show=show)

    # Dependence plots for top-3 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[::-1][:3]
    feature_names = list(X.columns)
    for idx in top3_idx:
        fname = feature_names[idx].replace(' ', '_')
        plot_dependence(shap_values, X, idx,
                        output_path=out / f'shap_dependence_{fname}.png',
                        show=show)

    print(f"All SHAP plots saved to {out}")
