"""
Optimized Data Generation Module for Cross-Cultural Adaptation Study

This module generates synthetic data that preserves interaction effects from real data
using a Gaussian Copula + Interaction Regression hybrid approach.

Key Features:
- Preserves 2-way, 3-way, and 4-way interaction effects
- Maintains marginal distributions and correlations
- Includes quadratic (non-linear) terms
- Fully reproducible with random seed control

Usage:
    from scripts.data_generation import generate_data
    from scripts.utils import set_all_seeds
    
    set_all_seeds(42)
    df_synthetic = generate_data(
        real_data_path='data/processed/real_data.xlsx',
        n_samples=100000,
        output_path='data/processed/synthetic_100k.csv'
    )
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from pathlib import Path
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

from ..utils import set_all_seeds


# Column mappings
COLUMN_MAPPING = {
    '序号': 'sample_id',
    '跨文化适应程度': 'cross_cultural_adaptation',
    '文化保持': 'cultural_maintenance',
    '社会保持': 'social_maintenance',
    '文化接触': 'cultural_contact',
    '社会接触': 'social_contact',
    '家庭支持': 'family_support',
    '家庭沟通频率': 'comm_frequency_feeling',
    '沟通坦诚度': 'comm_openness',
    '自主权': 'personal_autonomy',
    '社会联结感': 'social_connection',
    '开放性': 'openness',
    '来港时长': 'months_in_hk'
}

FEATURES_CN = ['文化保持', '社会保持', '文化接触', '社会接触', '家庭支持',
               '家庭沟通频率', '沟通坦诚度', '自主权', '社会联结感', '开放性', '来港时长']
TARGET_CN = '跨文化适应程度'

# Significant interactions (from statistical analysis)
SIGNIFICANT_2WAY_CN = [
    ('文化接触', '开放性'), ('家庭支持', '开放性'), ('社会接触', '开放性'),
    ('家庭沟通频率', '自主权'), ('家庭支持', '来港时长'), ('文化保持', '开放性'),
    ('社会接触', '来港时长'), ('沟通坦诚度', '自主权'),
]

SIGNIFICANT_3WAY_CN = [
    ('社会接触', '家庭支持', '社会联结感'), ('社会接触', '自主权', '来港时长'),
    ('文化接触', '家庭支持', '社会联结感'), ('文化接触', '社会接触', '来港时长'),
    ('文化接触', '家庭支持', '家庭沟通频率'), ('社会接触', '沟通坦诚度', '社会联结感'),
    ('沟通坦诚度', '自主权', '开放性'), ('社会接触', '家庭支持', '来港时长'),
    ('文化接触', '自主权', '社会联结感'), ('沟通坦诚度', '社会联结感', '开放性'),
]

SIGNIFICANT_4WAY_CN = [
    ('文化保持', '家庭支持', '家庭沟通频率', '社会联结感'),
    ('文化保持', '自主权', '社会联结感', '来港时长'),
    ('文化保持', '社会保持', '家庭支持', '社会联结感'),
    ('社会保持', '家庭支持', '家庭沟通频率', '自主权'),
    ('社会接触', '家庭沟通频率', '自主权', '开放性'),
    ('社会接触', '家庭支持', '家庭沟通频率', '开放性'),
    ('社会接触', '沟通坦诚度', '自主权', '开放性'),
    ('文化保持', '家庭支持', '自主权', '开放性'),
    ('文化保持', '社会保持', '家庭支持', '自主权'),
    ('社会保持', '家庭支持', '家庭沟通频率', '社会联结感'),
    ('社会接触', '自主权', '开放性', '来港时长'),
]

QUADRATIC_CN = ['开放性']


class InteractionRegressionModel:
    """Regression model with interaction terms for target generation."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.residual_kde = None
        self.residual_min = None
        self.residual_max = None
        self.y_mean = None
        self.y_std = None
        
    def _create_interaction_features(self, X_scaled, feature_names):
        """Create interaction feature matrix."""
        feat_idx = {name: i for i, name in enumerate(feature_names)}
        interaction_cols = []
        interaction_names = []
        
        # 2-way interactions
        for f1, f2 in SIGNIFICANT_2WAY_CN:
            col = X_scaled[:, feat_idx[f1]] * X_scaled[:, feat_idx[f2]]
            interaction_cols.append(col)
            interaction_names.append(f"{f1}×{f2}")
        
        # 3-way interactions
        for combo in SIGNIFICANT_3WAY_CN:
            col = np.ones(len(X_scaled))
            for f in combo:
                col = col * X_scaled[:, feat_idx[f]]
            interaction_cols.append(col)
            interaction_names.append('×'.join(combo))
        
        # 4-way interactions
        for combo in SIGNIFICANT_4WAY_CN:
            col = np.ones(len(X_scaled))
            for f in combo:
                col = col * X_scaled[:, feat_idx[f]]
            interaction_cols.append(col)
            interaction_names.append('×'.join(combo))
        
        # Quadratic terms
        for f in QUADRATIC_CN:
            col = X_scaled[:, feat_idx[f]] ** 2
            interaction_cols.append(col)
            interaction_names.append(f"{f}²")
        
        if interaction_cols:
            interactions = np.column_stack(interaction_cols)
            X_full = np.hstack([X_scaled, interactions])
            all_names = list(feature_names) + interaction_names
        else:
            X_full = X_scaled
            all_names = list(feature_names)
        
        return X_full, all_names
    
    def fit(self, df_real, verbose=True):
        """Fit interaction regression model on real data."""
        if verbose:
            print("\n" + "=" * 70)
            print("Stage 1: Fitting Interaction Regression Model")
            print("=" * 70)
        
        X = df_real[FEATURES_CN].values.astype(float)
        y = df_real[TARGET_CN].values.astype(float)
        
        self.y_mean = y.mean()
        self.y_std = y.std()
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Create interaction features
        X_full, self.feature_names = self._create_interaction_features(X_scaled, FEATURES_CN)
        
        if verbose:
            print(f"  Base features: {len(FEATURES_CN)}")
            print(f"  2-way interactions: {len(SIGNIFICANT_2WAY_CN)}")
            print(f"  3-way interactions: {len(SIGNIFICANT_3WAY_CN)}")
            print(f"  4-way interactions: {len(SIGNIFICANT_4WAY_CN)}")
            print(f"  Quadratic terms: {len(QUADRATIC_CN)}")
            print(f"  Total features: {X_full.shape[1]}")
        
        # ElasticNet with cross-validation
        alphas = np.logspace(-4, 1, 50)
        self.model = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.3, 0.5, 0.7], 
                                   cv=5, max_iter=10000, random_state=42)
        self.model.fit(X_full, y)
        
        y_pred = self.model.predict(X_full)
        r2 = r2_score(y, y_pred)
        residuals = y - y_pred
        
        # Fit residual distribution
        self.residual_kde = stats.gaussian_kde(residuals)
        self.residual_min = residuals.min()
        self.residual_max = residuals.max()
        
        if verbose:
            print(f"\n  Model Results:")
            print(f"    Best alpha: {self.model.alpha_:.4f}")
            print(f"    Training R²: {r2:.4f}")
            print(f"    Residual std: {residuals.std():.4f}")
            print(f"    Residual range: [{self.residual_min:.2f}, {self.residual_max:.2f}]")
        
        return r2
    
    def predict(self, X_raw):
        """Predict with interaction terms."""
        X_scaled = self.scaler.transform(X_raw)
        X_full, _ = self._create_interaction_features(X_scaled, FEATURES_CN)
        return self.model.predict(X_full)
    
    def predict_with_noise(self, X_raw, noise_scale=1.0):
        """Predict with calibrated noise."""
        y_pred = self.predict(X_raw)
        noise = self.residual_kde.resample(len(y_pred)).flatten()
        noise = np.clip(noise, self.residual_min * 1.5, self.residual_max * 1.5)
        return y_pred + noise * noise_scale


class CopulaFeatureGenerator:
    """Generate features using Gaussian Copula to preserve correlations."""
    
    def __init__(self):
        self.marginals = {}
        self.normal_corr = None
        
    def fit(self, df_real, verbose=True):
        """Fit marginal distributions and correlation structure."""
        if verbose:
            print("\n" + "=" * 70)
            print("Stage 2: Fitting Copula Model (Features Only)")
            print("=" * 70)
        
        for col in FEATURES_CN:
            data = df_real[col].dropna().values.astype(float)
            
            # Fit multiple distributions
            best_dist = None
            best_params = None
            best_ks = 1.0
            
            for dist_name, dist_obj in [('norm', stats.norm), ('lognorm', stats.lognorm),
                                         ('gamma', stats.gamma), ('beta', stats.beta)]:
                try:
                    params = dist_obj.fit(data)
                    ks_stat, _ = stats.kstest(data, dist_name, args=params)
                    if ks_stat < best_ks:
                        best_ks = ks_stat
                        best_dist = dist_name
                        best_params = params
                except:
                    continue
            
            if best_dist is None:
                best_dist = 'norm'
                best_params = stats.norm.fit(data)
            
            self.marginals[col] = {
                'distribution': best_dist,
                'params': best_params,
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'ks_stat': best_ks,
                'is_integer': bool(np.all(data == data.astype(int))),
            }
            
            if verbose:
                print(f"  {col}: {best_dist}, KS={best_ks:.4f}")
        
        # Spearman correlation matrix
        spearman_corr = df_real[FEATURES_CN].corr(method='spearman').values
        
        # Convert to normal copula Pearson correlation
        n = len(FEATURES_CN)
        self.normal_corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.normal_corr[i, j] = 1.0
                else:
                    self.normal_corr[i, j] = 2 * np.sin(np.pi / 6 * spearman_corr[i, j])
        
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(self.normal_corr)
        if np.min(eigvals) < 0:
            if verbose:
                print("  Correcting correlation matrix to positive definite...")
            self.normal_corr = self._nearest_positive_definite(self.normal_corr)
        
        if verbose:
            print(f"\n  Spearman correlation matrix fitted ({n}x{n})")
    
    def generate(self, n_samples, random_state=42, verbose=True):
        """Generate n_samples feature samples."""
        if verbose:
            print(f"\n  Generating {n_samples} feature samples...")
        
        np.random.seed(random_state)
        n_features = len(FEATURES_CN)
        
        # Multivariate normal sampling
        z_samples = np.random.multivariate_normal(
            np.zeros(n_features), self.normal_corr, size=n_samples
        )
        
        # Transform to uniform
        u_samples = stats.norm.cdf(z_samples)
        
        # Inverse CDF transform
        data = np.zeros((n_samples, n_features))
        for i, col in enumerate(FEATURES_CN):
            m = self.marginals[col]
            dist_obj = getattr(stats, m['distribution'])
            raw_values = dist_obj.ppf(u_samples[:, i], *m['params'])
            
            # Clip to range
            range_ext = 0.05 * (m['max'] - m['min'])
            raw_values = np.clip(raw_values, m['min'] - range_ext, m['max'] + range_ext)
            
            # Round if integer
            if m['is_integer']:
                raw_values = np.round(raw_values).astype(float)
                raw_values = np.clip(raw_values, m['min'], m['max'])
            
            data[:, i] = raw_values
        
        df_gen = pd.DataFrame(data, columns=FEATURES_CN)
        
        if verbose:
            print(f"  Feature generation complete: {df_gen.shape}")
        
        return df_gen
    
    @staticmethod
    def _nearest_positive_definite(A):
        """Find nearest positive definite matrix."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while k <= 100:
            try:
                np.linalg.cholesky(A3)
                return A3
            except np.linalg.LinAlgError:
                mineig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += I * (-mineig * k**2 + spacing)
                k += 1
        return A3


def generate_data(real_data_path, n_samples=100000, output_path=None, 
                  random_seed=42, noise_scale=1.0, verbose=True):
    """
    Generate synthetic data preserving interaction effects.
    
    Parameters
    ----------
    real_data_path : str or Path
        Path to real data Excel file
    n_samples : int, default=100000
        Number of samples to generate
    output_path : str or Path, optional
        Path to save generated data (CSV format)
    random_seed : int, default=42
        Random seed for reproducibility
    noise_scale : float, default=1.0
        Scale factor for residual noise
    verbose : bool, default=True
        Print progress information
        
    Returns
    -------
    pd.DataFrame
        Generated synthetic data with English column names
    """
    set_all_seeds(random_seed)
    
    if verbose:
        print("=" * 70)
        print("Synthetic Data Generation with Interaction Preservation")
        print("=" * 70)
        print(f"Target: {n_samples} samples")
        print(f"Method: Copula + Interaction Regression + Residual Calibration")
    
    # Load real data
    real_path = Path(real_data_path)
    if not real_path.exists():
        raise FileNotFoundError(f"Real data file not found: {real_path}")
    
    df_real = pd.read_excel(real_path)
    if verbose:
        print(f"\nReal data loaded: {len(df_real)} samples")
    
    # Stage 1: Fit interaction regression model
    reg_model = InteractionRegressionModel()
    r2_train = reg_model.fit(df_real, verbose=verbose)
    
    # Stage 2: Fit copula and generate features
    copula_gen = CopulaFeatureGenerator()
    copula_gen.fit(df_real, verbose=verbose)
    df_features = copula_gen.generate(n_samples, random_state=random_seed, verbose=verbose)
    
    # Stage 3: Generate target variable
    if verbose:
        print("\n" + "=" * 70)
        print("Stage 3: Generating Target Variable")
        print("=" * 70)
    
    X_raw = df_features[FEATURES_CN].values.astype(float)
    y_pred = reg_model.predict(X_raw)
    
    # Add calibrated noise
    noise = reg_model.residual_kde.resample(len(y_pred)).flatten()
    noise = np.clip(noise, reg_model.residual_min * 1.5, reg_model.residual_max * 1.5)
    y_gen = y_pred + noise * noise_scale
    
    # Clip and round
    y_min, y_max = 8, 32
    y_gen = np.clip(y_gen, y_min - 1, y_max + 1)
    y_gen = np.round(y_gen).astype(float)
    y_gen = np.clip(y_gen, y_min, y_max)
    
    if verbose:
        print(f"  Target mean: {y_gen.mean():.2f} (real: {reg_model.y_mean:.2f})")
        print(f"  Target std: {y_gen.std():.2f} (real: {reg_model.y_std:.2f})")
        print(f"  Target range: [{y_gen.min():.0f}, {y_gen.max():.0f}]")
    
    # Combine data
    df_gen = df_features.copy()
    df_gen[TARGET_CN] = y_gen
    
    # Convert to English column names
    df_gen_en = df_gen.rename(columns=COLUMN_MAPPING)
    df_gen_en['sample_id'] = range(n_samples)
    
    # Save if output path provided
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_gen_en.to_csv(out_path, index=False)
        if verbose:
            print(f"\n  Data saved to: {out_path}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("✅ Data generation complete!")
        print("=" * 70)
    
    return df_gen_en
