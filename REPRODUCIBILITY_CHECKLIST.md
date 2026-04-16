# Reproducibility Checklist

This checklist ensures all results in the study can be fully reproduced.

## Environment

- [ ] Python 3.8+ installed
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Verify environment: `python -c "import scripts.utils; scripts.utils.log_environment_info()"`

## Random Seeds

All stochastic operations use **seed = 42**:

| Component | Seed Setting |
|---|---|
| NumPy | `np.random.seed(42)` |
| Python random | `random.seed(42)` |
| XGBoost | `random_state=42` |
| LightGBM | `random_state=42` |
| CatBoost | `random_seed=42` |
| Scikit-learn | `random_state=42` |
| Data generation | `random_seed=42` |

Use `scripts.utils.set_all_seeds(42)` at the top of any script to set all seeds at once.

## Data

- [ ] Raw HK data: `CFA/HKN=75.xlsx` (N=75, SHA-256 should match repository)
- [ ] Raw France data: `CFA/Franchn=249.xlsx` (N=249)
- [ ] Synthetic France data generated with `n_samples=100000`, `months_filter=48`
- [ ] KS test p-values > 0.05 for all 11 features (see `appendices/Appendix_D_Simulated_Data_Validation.md`)

## Model Training

- [ ] HK model: run `python train_v7_complete_with_cv.py`
  - Expected CV R²: ~0.82 (±0.05)
  - Expected CV RMSE: ~0.41 (±0.03)
- [ ] France model: run `python train_france_v7_complete_with_cv.py`
  - Expected CV R²: ~0.79 (±0.04)
  - Expected CV RMSE: ~0.44 (±0.03)

## Cross-Validation

- 5-fold stratified CV
- Fold splits are deterministic given seed=42
- Results saved to `results/cv_results_75samples.csv` and `france_models/cv_results_france.csv`

## SHAP Analysis

- [ ] SHAP values computed with `shap.TreeExplainer` for tree-based models
- [ ] SHAP values averaged across all 6 base learners
- [ ] Top feature: `cultural_openness` (both samples)
- [ ] SHAP values saved to `france_models/shap_values_france.npy`

## Figures

All figures are generated deterministically. To regenerate:

```bash
python generate_academic_figures.py
python generate_figures_4_11_to_4_17.py
python generate_figures_4_18_to_4_20.py
```

Output: `academic_figures/` directory (PNG, PDF, SVG, EPS formats).

## Key Results to Verify

| Metric | HK | France |
|---|---|---|
| CV R² (mean) | ~0.82 | ~0.79 |
| CV RMSE (mean) | ~0.41 | ~0.44 |
| Top feature | cultural_openness | cultural_openness |
| 2nd feature | social_support | language_proficiency |

## Software Versions (used in original study)

See `requirements.txt` for pinned versions. Key packages:

- `xgboost >= 1.7`
- `lightgbm >= 3.3`
- `catboost >= 1.1`
- `shap >= 0.41`
- `scikit-learn >= 1.1`
- `scipy >= 1.9`
