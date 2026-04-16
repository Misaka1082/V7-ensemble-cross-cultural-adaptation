"""Log environment information for reproducibility."""

import sys
import platform
import numpy as np
import pandas as pd
import sklearn
from datetime import datetime


def log_environment_info(output_file=None):
    """
    Log system and package version information for reproducibility.
    
    Parameters
    ----------
    output_file : str, optional
        Path to save the environment log. If None, prints to console.
        
    Returns
    -------
    dict
        Dictionary containing environment information
        
    Examples
    --------
    >>> from scripts.utils import log_environment_info
    >>> env_info = log_environment_info('logs/environment.txt')
    """
    env_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'sklearn_version': sklearn.__version__,
    }
    
    # Try to get additional package versions
    try:
        import xgboost
        env_info['xgboost_version'] = xgboost.__version__
    except ImportError:
        env_info['xgboost_version'] = 'Not installed'
    
    try:
        import lightgbm
        env_info['lightgbm_version'] = lightgbm.__version__
    except ImportError:
        env_info['lightgbm_version'] = 'Not installed'
    
    try:
        import catboost
        env_info['catboost_version'] = catboost.__version__
    except ImportError:
        env_info['catboost_version'] = 'Not installed'
    
    try:
        import shap
        env_info['shap_version'] = shap.__version__
    except ImportError:
        env_info['shap_version'] = 'Not installed'
    
    # Format output
    output_lines = ["=" * 60]
    output_lines.append("ENVIRONMENT INFORMATION FOR REPRODUCIBILITY")
    output_lines.append("=" * 60)
    for key, value in env_info.items():
        output_lines.append(f"{key}: {value}")
    output_lines.append("=" * 60)
    
    output_text = "\n".join(output_lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Environment information saved to {output_file}")
    else:
        print(output_text)
    
    return env_info
