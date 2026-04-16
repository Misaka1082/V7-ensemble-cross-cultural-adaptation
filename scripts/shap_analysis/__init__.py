"""SHAP analysis module for V7 ensemble model interpretability."""

from .compute_shap_values import compute_shap_values, compute_ensemble_shap, save_shap_values
from .generate_shap_visualizations import (
    plot_summary, plot_bar, plot_waterfall, plot_dependence, generate_all_shap_plots
)

__all__ = [
    'compute_shap_values',
    'compute_ensemble_shap',
    'save_shap_values',
    'plot_summary',
    'plot_bar',
    'plot_waterfall',
    'plot_dependence',
    'generate_all_shap_plots',
]
