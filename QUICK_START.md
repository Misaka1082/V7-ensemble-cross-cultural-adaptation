# Quick Start Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

## 1. Generate Synthetic Data

```bash
# Generate a small sample dataset for testing (1,000 rows)
python scripts/data_generation/generate_sample_data.py \
    --real_data CFA/HKN=75.xlsx \
    --n_samples 1000 \
    --output_dir data/sample/

# Generate the full synthetic dataset (100,000 rows, France)
python scripts/data_generation/generate_sample_data.py \
    --real_data CFA/Franchn=249.xlsx \
    --n_samples 100000 \
    --output_dir data/france_synthetic/
```

## 2. Train the V7 Ensemble Model

```bash
# Hong Kong sample (N=75)
python train_v7_complete_with_cv.py

# France sample (N=249)
python train_france_v7_complete_with_cv.py
```

Both scripts use **random seed = 42** and **5-fold cross-validation**.

## 3. Run SHAP Analysis

```python
from scripts.shap_analysis import compute_ensemble_shap, generate_all_shap_plots
import pandas as pd

X = pd.read_csv('data/sample/sample_data.csv').drop(columns=['adaptation_score'])
mean_shap, shap_dict = compute_ensemble_shap('results/hk_v7_models.pkl', X)
generate_all_shap_plots(mean_shap, X, output_dir='results/shap_plots/')
```

## 4. Evaluate Model Performance

```python
from scripts.evaluation import regression_metrics, evaluate_cv_folds
import pandas as pd

cv = pd.read_csv('results/cv_results_75samples.csv')
fold_df, agg = evaluate_cv_folds(cv)
print(agg)
```

## 5. Validate Cross-Cultural Results

```python
from scripts.evaluation import validate_cross_cultural_consistency

metrics_hk     = {'r2': 0.82, 'rmse': 0.41, 'mae': 0.33}
metrics_france = {'r2': 0.79, 'rmse': 0.44, 'mae': 0.35}
validate_cross_cultural_consistency(metrics_hk, metrics_france)
```

## Project Structure

```
├── scripts/
│   ├── data_generation/   # Synthetic data generation (Copula + Interaction Regression)
│   ├── shap_analysis/     # SHAP value computation and visualization
│   ├── evaluation/        # Model metrics and cross-cultural validation
│   └── utils/             # Reproducibility helpers (seeds, environment logging)
├── train_v7_complete_with_cv.py        # HK model training
├── train_france_v7_complete_with_cv.py # France model training
├── CFA/                   # Raw data files
├── results/               # Model outputs and figures
└── requirements.txt
```

## Reproducibility

All experiments use `random_seed=42`. See `REPRODUCIBILITY_CHECKLIST.md` for full details.
