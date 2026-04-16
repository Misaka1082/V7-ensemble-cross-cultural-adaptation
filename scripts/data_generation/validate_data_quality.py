"""
Data Quality Validation Module

Validates generated synthetic data quality by checking:
- Distribution similarity (KS tests)
- Correlation preservation
- Interaction effect preservation
- Statistical properties

Usage:
    from scripts.data_generation import validate_data_quality
    
    results = validate_data_quality(
        generated_data='data/processed/synthetic_100k.csv',
        real_data='data/processed/real_data.xlsx',
        output_report='validation_report.json'
    )
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from pathlib import Path
import json


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

COLUMN_MAPPING_REV = {v: k for k, v in COLUMN_MAPPING.items()}

FEATURES_CN = ['文化保持', '社会保持', '文化接触', '社会接触', '家庭支持',
               '家庭沟通频率', '沟通坦诚度', '自主权', '社会联结感', '开放性', '来港时长']
TARGET_CN = '跨文化适应程度'

FEATURES_EN = [COLUMN_MAPPING[f] for f in FEATURES_CN]
TARGET_EN = 'cross_cultural_adaptation'


def validate_distributions(df_gen, df_real, verbose=True):
    """
    Validate distribution similarity using KS tests.
    
    Parameters
    ----------
    df_gen : pd.DataFrame
        Generated data (English column names)
    df_real : pd.DataFrame
        Real data (Chinese column names)
    verbose : bool
        Print results
        
    Returns
    -------
    dict
        KS test results for each variable
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Distribution Validation (KS Tests)")
        print("=" * 70)
        print(f"{'Variable':<25} {'KS Statistic':>12} {'p-value':>10} {'Status':>8}")
        print("-" * 70)
    
    results = {}
    
    # Convert generated data to Chinese names for comparison
    df_gen_cn = df_gen.rename(columns=COLUMN_MAPPING_REV)
    
    for col_cn in FEATURES_CN + [TARGET_CN]:
        real_data = df_real[col_cn].dropna().values
        gen_data = df_gen_cn[col_cn].dropna().values
        
        ks_stat, p_val = stats.ks_2samp(real_data, gen_data)
        
        # Status: ✓ if p > 0.01 (distributions similar)
        status = "✓" if p_val > 0.01 else "⚠" if p_val > 0.001 else "✗"
        
        results[col_cn] = {
            'ks_statistic': float(ks_stat),
            'p_value': float(p_val),
            'passed': p_val > 0.01
        }
        
        if verbose:
            print(f"{col_cn:<25} {ks_stat:12.4f} {p_val:10.4f} {status:>8}")
    
    pass_rate = sum(1 for r in results.values() if r['passed']) / len(results)
    
    if verbose:
        print("-" * 70)
        print(f"Pass rate: {pass_rate:.1%} ({sum(1 for r in results.values() if r['passed'])}/{len(results)})")
    
    return results


def validate_correlations(df_gen, df_real, verbose=True):
    """
    Validate feature-target correlation preservation.
    
    Parameters
    ----------
    df_gen : pd.DataFrame
        Generated data (English column names)
    df_real : pd.DataFrame
        Real data (Chinese column names)
    verbose : bool
        Print results
        
    Returns
    -------
    dict
        Correlation comparison results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Correlation Validation (Spearman)")
        print("=" * 70)
        print(f"{'Feature':<25} {'Real':>10} {'Generated':>10} {'Diff':>10} {'Status':>8}")
        print("-" * 70)
    
    results = {}
    df_gen_cn = df_gen.rename(columns=COLUMN_MAPPING_REV)
    
    for col_cn in FEATURES_CN:
        real_corr = df_real[col_cn].corr(df_real[TARGET_CN], method='spearman')
        gen_corr = df_gen_cn[col_cn].corr(df_gen_cn[TARGET_CN], method='spearman')
        diff = abs(real_corr - gen_corr)
        
        # Status: ✓ if diff < 0.1
        status = "✓" if diff < 0.1 else "⚠" if diff < 0.2 else "✗"
        
        results[col_cn] = {
            'real_correlation': float(real_corr),
            'generated_correlation': float(gen_corr),
            'difference': float(diff),
            'passed': diff < 0.1
        }
        
        if verbose:
            print(f"{col_cn:<25} {real_corr:10.4f} {gen_corr:10.4f} {diff:10.4f} {status:>8}")
    
    avg_diff = np.mean([r['difference'] for r in results.values()])
    pass_rate = sum(1 for r in results.values() if r['passed']) / len(results)
    
    if verbose:
        print("-" * 70)
        print(f"Average difference: {avg_diff:.4f}")
        print(f"Pass rate: {pass_rate:.1%}")
    
    return results


def validate_interactions(df_gen, df_real, sample_size=5000, verbose=True):
    """
    Validate key 2-way interaction preservation.
    
    Parameters
    ----------
    df_gen : pd.DataFrame
        Generated data (English column names)
    df_real : pd.DataFrame
        Real data (Chinese column names)
    sample_size : int
        Sample size for generated data testing
    verbose : bool
        Print results
        
    Returns
    -------
    dict
        Interaction validation results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Interaction Effect Validation (Top 5 2-way)")
        print("=" * 70)
    
    # Top 5 most significant 2-way interactions
    key_interactions = [
        ('文化接触', '开放性'),
        ('家庭支持', '开放性'),
        ('社会接触', '开放性'),
        ('家庭沟通频率', '自主权'),
        ('家庭支持', '来港时长'),
    ]
    
    df_gen_cn = df_gen.rename(columns=COLUMN_MAPPING_REV)
    
    # Standardize
    scaler_real = StandardScaler()
    X_real = pd.DataFrame(scaler_real.fit_transform(df_real[FEATURES_CN]), columns=FEATURES_CN)
    y_real = df_real[TARGET_CN].values
    
    scaler_gen = StandardScaler()
    X_gen = pd.DataFrame(scaler_gen.fit_transform(df_gen_cn[FEATURES_CN]), columns=FEATURES_CN)
    y_gen = df_gen_cn[TARGET_CN].values
    
    results = []
    
    if verbose:
        print(f"{'Interaction':<30} {'Real p':>10} {'Gen p':>10} {'Direction':>10} {'Status':>8}")
        print("-" * 70)
    
    for f1, f2 in key_interactions:
        # Real data
        X_test_r = X_real.copy()
        X_test_r[f'{f1}×{f2}'] = X_real[f1] * X_real[f2]
        X_test_r = sm.add_constant(X_test_r)
        model_r = sm.OLS(y_real, X_test_r).fit()
        coef_r = model_r.params[f'{f1}×{f2}']
        pval_r = model_r.pvalues[f'{f1}×{f2}']
        
        # Generated data (sample)
        sample_idx = np.random.choice(len(X_gen), min(sample_size, len(X_gen)), replace=False)
        X_test_g = X_gen.iloc[sample_idx].copy()
        X_test_g[f'{f1}×{f2}'] = X_gen.iloc[sample_idx][f1] * X_gen.iloc[sample_idx][f2]
        X_test_g = sm.add_constant(X_test_g)
        model_g = sm.OLS(y_gen[sample_idx], X_test_g).fit()
        coef_g = model_g.params[f'{f1}×{f2}']
        pval_g = model_g.pvalues[f'{f1}×{f2}']
        
        direction_match = (coef_r > 0) == (coef_g > 0)
        passed = direction_match and pval_g < 0.1
        status = "✓" if passed else "⚠" if direction_match else "✗"
        
        results.append({
            'interaction': f'{f1}×{f2}',
            'real_p': float(pval_r),
            'gen_p': float(pval_g),
            'real_coef': float(coef_r),
            'gen_coef': float(coef_g),
            'direction_match': bool(direction_match),
            'passed': bool(passed)
        })
        
        if verbose:
            interaction_name = f"{f1}×{f2}"
            dir_str = "Match" if direction_match else "Diff"
            print(f"{interaction_name:<30} {pval_r:10.4f} {pval_g:10.4f} {dir_str:>10} {status:>8}")
    
    pass_rate = sum(1 for r in results if r['passed']) / len(results)
    
    if verbose:
        print("-" * 70)
        print(f"Pass rate: {pass_rate:.1%}")
    
    return results


def validate_data_quality(generated_data, real_data, output_report=None, verbose=True):
    """
    Comprehensive data quality validation.
    
    Parameters
    ----------
    generated_data : str or Path or pd.DataFrame
        Generated synthetic data (CSV file or DataFrame)
    real_data : str or Path or pd.DataFrame
        Real data (Excel/CSV file or DataFrame)
    output_report : str or Path, optional
        Path to save validation report (JSON)
    verbose : bool
        Print validation results
        
    Returns
    -------
    dict
        Comprehensive validation results
    """
    if verbose:
        print("=" * 70)
        print("DATA QUALITY VALIDATION")
        print("=" * 70)
    
    # Load data
    if isinstance(generated_data, (str, Path)):
        df_gen = pd.read_csv(generated_data)
    else:
        df_gen = generated_data
    
    if isinstance(real_data, (str, Path)):
        real_path = Path(real_data)
        if real_path.suffix == '.xlsx':
            df_real = pd.read_excel(real_path)
        else:
            df_real = pd.read_csv(real_path)
    else:
        df_real = real_data
    
    if verbose:
        print(f"\nGenerated data: {len(df_gen)} samples")
        print(f"Real data: {len(df_real)} samples")
    
    # Run validations
    validation_results = {
        'distributions': validate_distributions(df_gen, df_real, verbose=verbose),
        'correlations': validate_correlations(df_gen, df_real, verbose=verbose),
        'interactions': validate_interactions(df_gen, df_real, verbose=verbose),
    }
    
    # Summary
    dist_pass = sum(1 for r in validation_results['distributions'].values() if r['passed'])
    corr_pass = sum(1 for r in validation_results['correlations'].values() if r['passed'])
    int_pass = sum(1 for r in validation_results['interactions'] if r['passed'])
    
    validation_results['summary'] = {
        'distribution_pass_rate': dist_pass / len(validation_results['distributions']),
        'correlation_pass_rate': corr_pass / len(validation_results['correlations']),
        'interaction_pass_rate': int_pass / len(validation_results['interactions']),
        'overall_quality': 'Good' if all([
            dist_pass / len(validation_results['distributions']) > 0.7,
            corr_pass / len(validation_results['correlations']) > 0.7,
            int_pass / len(validation_results['interactions']) > 0.6
        ]) else 'Acceptable' if all([
            dist_pass / len(validation_results['distributions']) > 0.5,
            corr_pass / len(validation_results['correlations']) > 0.5,
            int_pass / len(validation_results['interactions']) > 0.4
        ]) else 'Needs Improvement'
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Distribution validation: {validation_results['summary']['distribution_pass_rate']:.1%}")
        print(f"Correlation validation: {validation_results['summary']['correlation_pass_rate']:.1%}")
        print(f"Interaction validation: {validation_results['summary']['interaction_pass_rate']:.1%}")
        print(f"\nOverall quality: {validation_results['summary']['overall_quality']}")
        print("=" * 70)
    
    # Save report
    if output_report:
        report_path = Path(output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\nValidation report saved to: {report_path}")
    
    return validation_results
