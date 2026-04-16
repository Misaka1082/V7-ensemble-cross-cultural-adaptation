"""Model evaluation utilities for V7 ensemble."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_metrics(y_true, y_pred, prefix=''):
    """
    Compute standard regression metrics.

    Returns
    -------
    dict with keys: rmse, mae, r2, mape
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE — guard against zero division
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    metrics = {f'{prefix}rmse': rmse, f'{prefix}mae': mae,
               f'{prefix}r2': r2, f'{prefix}mape': mape}
    return metrics


def evaluate_cv_folds(cv_results, y_col='y_true', pred_col='y_pred', fold_col='fold'):
    """
    Evaluate per-fold and aggregate metrics from cross-validation results DataFrame.

    Parameters
    ----------
    cv_results : pd.DataFrame  with columns [fold, y_true, y_pred]

    Returns
    -------
    fold_metrics : pd.DataFrame  per-fold metrics
    agg_metrics : dict  mean ± std across folds
    """
    fold_metrics = []
    for fold, grp in cv_results.groupby(fold_col):
        m = regression_metrics(grp[y_col], grp[pred_col], prefix='')
        m['fold'] = fold
        fold_metrics.append(m)

    fold_df = pd.DataFrame(fold_metrics).set_index('fold')

    agg = {}
    for col in ['rmse', 'mae', 'r2', 'mape']:
        agg[f'{col}_mean'] = fold_df[col].mean()
        agg[f'{col}_std'] = fold_df[col].std()

    return fold_df, agg


def compare_models(results_dict, y_true):
    """
    Compare multiple models given a dict of {model_name: y_pred}.

    Returns
    -------
    pd.DataFrame  sorted by R²
    """
    rows = []
    for name, y_pred in results_dict.items():
        m = regression_metrics(y_true, y_pred)
        m['model'] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index('model').sort_values('r2', ascending=False)
    return df


def print_metrics(metrics, title='Model Performance'):
    """Pretty-print a metrics dict."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")
    print(f"{'='*50}\n")
