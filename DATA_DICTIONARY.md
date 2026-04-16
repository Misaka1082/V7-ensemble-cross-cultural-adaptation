# Data Dictionary

## Study Overview

Cross-cultural adaptation study comparing Hong Kong (HK, N=75) and France (N=249) samples.  
Target variable: **Psychological adaptation score** (continuous, higher = better adaptation).

---

## Predictor Variables (11 features)

| Variable | Description | Scale | Type |
|---|---|---|---|
| `cultural_openness` | Openness to host culture norms and values | 1–7 Likert | Continuous |
| `language_proficiency` | Self-rated proficiency in host language | 1–7 Likert | Continuous |
| `social_support` | Perceived social support from host nationals | 1–7 Likert | Continuous |
| `family_support` | Support from family (local or remote) | 1–7 Likert | Continuous |
| `autonomy` | Sense of personal autonomy and self-direction | 1–7 Likert | Continuous |
| `identity_clarity` | Clarity of personal/cultural identity | 1–7 Likert | Continuous |
| `discrimination_experience` | Frequency of perceived discrimination | 1–7 Likert | Continuous (reverse-coded) |
| `host_contact_quality` | Quality of contact with host nationals | 1–7 Likert | Continuous |
| `heritage_maintenance` | Maintenance of heritage cultural practices | 1–7 Likert | Continuous |
| `months_in_country` | Duration of residence in host country | Months | Continuous |
| `age` | Participant age | Years | Continuous |

---

## Target Variable

| Variable | Description | Range |
|---|---|---|
| `adaptation_score` | Composite psychological adaptation score | 1–7 |

---

## Synthetic Data

The synthetic datasets (`france_data/france_100k_48months.csv`) were generated using a **Gaussian Copula + Interaction Regression** hybrid method:

1. **Gaussian Copula** preserves marginal distributions and pairwise correlations.
2. **Interaction Regression** (ElasticNetCV) captures 2-way, 3-way, 4-way interactions and quadratic terms.
3. Samples filtered to `months_in_country <= 48`.
4. KS tests confirm distribution similarity (all p > 0.05).

See `scripts/data_generation/generate_data.py` for implementation details.

---

## Raw Data Files

| File | Sample | N | Format |
|---|---|---|---|
| `CFA/HKN=75.xlsx` | Hong Kong | 75 | Excel |
| `CFA/Franchn=249.xlsx` | France | 249 | Excel |
| `CFA/FranchN=249.csv` | France | 249 | CSV |

> Raw data files contain anonymized participant responses. No PII is included.

---

## Model Output Files

| File | Description |
|---|---|
| `results/cv_results_75samples.csv` | 5-fold CV predictions for HK sample |
| `france_models/cv_results_france.csv` | 5-fold CV predictions for France sample |
| `france_models/feature_importance.csv` | Feature importance scores (France) |
| `results/feature_importance_75samples_cv.csv` | Feature importance scores (HK) |
| `france_models/shap_values_france.npy` | SHAP values array (France, shape: N×11) |
