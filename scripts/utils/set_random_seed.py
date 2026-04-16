"""Set random seeds for reproducibility across all libraries."""

import random
import numpy as np
import os


def set_all_seeds(seed=42):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value
        
    Notes
    -----
    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - Environment variables for hash randomization
    
    Examples
    --------
    >>> from scripts.utils import set_all_seeds
    >>> set_all_seeds(42)
    Random seeds set to 42 for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seeds set to {seed} for reproducibility")
    
    return seed
