"""
Generate Small Sample Dataset for Testing and Demonstration

This script generates a small sample dataset (1000 samples by default) for:
- Quick testing of analysis pipelines
- Demonstration purposes
- Tutorial examples
- CI/CD testing

Usage:
    python -m scripts.data_generation.generate_sample_data
    
    Or programmatically:
    from scripts.data_generation import generate_sample_dataset
    
    df_sample = generate_sample_dataset(
        real_data_path='data/processed/real_data.xlsx',
        n_samples=1000,
        output_dir='data/sample/'
    )
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.data_generation.generate_data import generate_data
from scripts.data_generation.validate_data_quality import validate_data_quality
from scripts.utils import set_all_seeds, log_environment_info


def generate_sample_dataset(real_data_path, n_samples=1000, output_dir='data/sample/', 
                            random_seed=42, validate=True, verbose=True):
    """
    Generate a small sample dataset for testing and demonstration.
    
    Parameters
    ----------
    real_data_path : str or Path
        Path to real data Excel file
    n_samples : int, default=1000
        Number of samples to generate (recommended: 1000-5000)
    output_dir : str or Path, default='data/sample/'
        Directory to save sample data
    random_seed : int, default=42
        Random seed for reproducibility
    validate : bool, default=True
        Run validation after generation
    verbose : bool, default=True
        Print progress information
        
    Returns
    -------
    pd.DataFrame
        Generated sample dataset
    """
    if verbose:
        print("=" * 70)
        print("SAMPLE DATA GENERATION FOR TESTING & DEMONSTRATION")
        print("=" * 70)
        print(f"Generating {n_samples} samples...")
        print(f"Random seed: {random_seed}")
    
    # Set random seeds
    set_all_seeds(random_seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    sample_file = output_path / f'sample_data_{n_samples}.csv'
    df_sample = generate_data(
        real_data_path=real_data_path,
        n_samples=n_samples,
        output_path=sample_file,
        random_seed=random_seed,
        noise_scale=1.0,
        verbose=verbose
    )
    
    # Validate if requested
    if validate:
        if verbose:
            print("\n" + "=" * 70)
            print("VALIDATING SAMPLE DATA QUALITY")
            print("=" * 70)
        
        validation_report = output_path / f'sample_data_{n_samples}_validation.json'
        validation_results = validate_data_quality(
            generated_data=df_sample,
            real_data=real_data_path,
            output_report=validation_report,
            verbose=verbose
        )
    
    if verbose:
        print("\n" + "=" * 70)
        print("✅ SAMPLE DATA GENERATION COMPLETE")
        print("=" * 70)
        print(f"Sample data saved to: {sample_file}")
        print(f"Number of samples: {len(df_sample)}")
        print(f"Number of features: {len(df_sample.columns)}")
        if validate:
            print(f"Validation report: {validation_report}")
            print(f"Overall quality: {validation_results['summary']['overall_quality']}")
        print("=" * 70)
    
    return df_sample


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate sample dataset for testing and demonstration'
    )
    parser.add_argument(
        '--real-data',
        type=str,
        default='data/processed/real_data_filtered_48months.xlsx',
        help='Path to real data file (default: data/processed/real_data_filtered_48months.xlsx)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples to generate (default: 1000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/sample/',
        help='Output directory (default: data/sample/)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation step'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )
    
    args = parser.parse_args()
    
    # Log environment
    if not args.quiet:
        log_environment_info()
    
    # Generate sample data
    df_sample = generate_sample_dataset(
        real_data_path=args.real_data,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        random_seed=args.seed,
        validate=not args.no_validate,
        verbose=not args.quiet
    )
    
    return df_sample


if __name__ == '__main__':
    main()
