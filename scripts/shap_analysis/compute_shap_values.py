"""Compute SHAP values for V7 ensemble model."""

import numpy as np
import pandas as pd
import pickle
import shap
from pathlib import Path


def load_model(model_path):
    """Load a trained model from pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def compute_shap_values(model_path, X, model_type='xgboost', background_samples=100,
                        random_seed=42):
    """
    Compute SHAP values for a given model and dataset.

    Parameters
    ----------
    model_path : str or Path
        Path to the trained model pickle file.
    X : pd.DataFrame
        Feature matrix.
    model_type : str
        One of 'xgboost', 'lightgbm', 'catboost', 'gbm', 'randomforest', 'linear'.
    background_samples : int
        Number of background samples for KernelExplainer (tree models use TreeExplainer).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features).
    explainer : shap.Explainer
        The fitted SHAP explainer object.
    """
    np.random.seed(random_seed)
    model = load_model(model_path)

    tree_types = ('xgboost', 'lightgbm', 'catboost', 'gbm', 'randomforest')

    if model_type in tree_types:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        # Linear or unknown: use LinearExplainer or KernelExplainer
        np.random.seed(random_seed)
        background = shap.sample(X, min(background_samples, len(X)))
        try:
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X)
        except Exception:
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X, nsamples=200)

    return shap_values, explainer


def compute_ensemble_shap(models_path, X, feature_names=None, random_seed=42):
    """
    Compute averaged SHAP values across all base learners in the V7 ensemble.

    Parameters
    ----------
    models_path : str or Path
        Path to the ensemble models pickle file (dict of {name: model}).
    X : pd.DataFrame
        Feature matrix.
    feature_names : list, optional
        Feature names. Defaults to X.columns.
    random_seed : int
        Random seed.

    Returns
    -------
    mean_shap : np.ndarray  shape (n_samples, n_features)
    shap_dict : dict  {model_name: shap_values}
    """
    np.random.seed(random_seed)

    if feature_names is None:
        feature_names = list(X.columns)

    with open(models_path, 'rb') as f:
        models = pickle.load(f)

    tree_model_names = {'xgboost', 'lightgbm', 'catboost', 'gbm', 'randomforest',
                        'XGBoost', 'LightGBM', 'CatBoost', 'GBM', 'RandomForest',
                        'GradientBoosting', 'gradient_boosting'}

    shap_dict = {}
    for name, model in models.items():
        try:
            mtype = 'tree' if any(t.lower() in name.lower()
                                  for t in tree_model_names) else 'linear'
            if mtype == 'tree':
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X)
            else:
                np.random.seed(random_seed)
                background = shap.sample(X, min(100, len(X)))
                explainer = shap.LinearExplainer(model, background)
                sv = explainer.shap_values(X)
            shap_dict[name] = sv
            print(f"  Computed SHAP for {name}: shape {np.array(sv).shape}")
        except Exception as e:
            print(f"  Warning: Could not compute SHAP for {name}: {e}")

    if not shap_dict:
        raise RuntimeError("No SHAP values could be computed for any model.")

    arrays = [np.array(v) for v in shap_dict.values()]
    mean_shap = np.mean(arrays, axis=0)
    return mean_shap, shap_dict


def save_shap_values(shap_values, output_path, feature_names=None):
    """Save SHAP values to .npy and optionally a CSV summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_path), shap_values)
    print(f"SHAP values saved to {output_path}")

    if feature_names is not None:
        mean_abs = np.abs(shap_values).mean(axis=0)
        summary = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs
        }).sort_values('mean_abs_shap', ascending=False)
        csv_path = output_path.with_suffix('.csv')
        summary.to_csv(csv_path, index=False)
        print(f"SHAP summary saved to {csv_path}")
        return summary

    return None
