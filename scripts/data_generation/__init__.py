"""Data generation module for synthetic data with interaction preservation."""

from .generate_data import generate_data, InteractionRegressionModel, CopulaFeatureGenerator
from .validate_data_quality import validate_data_quality
from .generate_sample_data import generate_sample_dataset

__all__ = [
    'generate_data',
    'InteractionRegressionModel',
    'CopulaFeatureGenerator',
    'validate_data_quality',
    'generate_sample_dataset',
]
