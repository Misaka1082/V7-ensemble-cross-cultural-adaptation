"""Cross-cultural result validation for HK and France samples."""

import numpy as np
import pandas as pd
from scipy import stats


def fisher_z_test(r1, n1, r2, n2):
    """
    Fisher Z-transformation test for comparing two correlations.

    Returns
    -------
    z_stat : float
    p_value : float
    """
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value


def compare_feature_importance(fi_hk, fi_france, feature_col='feature',
                                importance_col='importance'):
    """
    Compare feature importance rankings between two samples.

    Parameters
    ----------
    fi_hk, fi_france : pd.DataFrame  with feature and importance columns

    Returns
    -------
    pd.DataFrame  merged comparison with rank difference
    """
    hk = fi_hk[[feature_col, importance_col]].copy()
    hk['rank_hk'] = hk[importance_col].rank(ascending=False).astype(int)
    hk = hk.rename(columns={importance_col: 'importance_hk'})

    fr = fi_france[[feature_col, importance_col]].copy()
    fr['rank_france'] = fr[importance_col].rank(ascending=False).astype(int)
    fr = fr.rename(columns={importance_col: 'importance_france'})

    merged = hk.merge(fr, on=feature_col, how='outer')
    merged['rank_diff'] = (merged['rank_hk'] - merged['rank_france']).abs()

    rho, p = stats.spearmanr(
        merged['rank_hk'].fillna(merged['rank_hk'].max() + 1),
        merged['rank_france'].fillna(merged['rank_france'].max() + 1)
    )
    print(f"Spearman rank correlation (HK vs France): rho={rho:.3f}, p={p:.4f}")
    return merged.sort_values('rank_hk')


def validate_cross_cultural_consistency(metrics_hk, metrics_france, threshold_r2=0.5):
    """
    Check whether both samples meet minimum performance thresholds
    and whether their R² values are within an acceptable range.

    Parameters
    ----------
    metrics_hk, metrics_france : dict  with 'r2', 'rmse', 'mae' keys
    threshold_r2 : float  minimum acceptable R²

    Returns
    -------
    report : dict
    """
    report = {
        'hk_r2': metrics_hk['r2'],
        'france_r2': metrics_france['r2'],
        'hk_meets_threshold': metrics_hk['r2'] >= threshold_r2,
        'france_meets_threshold': metrics_france['r2'] >= threshold_r2,
        'r2_difference': abs(metrics_hk['r2'] - metrics_france['r2']),
        'consistent': abs(metrics_hk['r2'] - metrics_france['r2']) < 0.15,
    }

    print("\nCross-Cultural Validation Report")
    print("=" * 40)
    for k, v in report.items():
        print(f"  {k:<30}: {v}")
    print("=" * 40)
    return report


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000,
                 ci=0.95, random_seed=42):
    """
    Bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    metric_fn : callable  f(y_true, y_pred) -> float

    Returns
    -------
    (lower, upper, mean)
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(scores, [alpha, 1 - alpha])
    return lower, upper, scores.mean()
