# Cross-Cultural Adaptation Study: Hong Kong & France

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Predicting Cross-Cultural Adaptation Outcomes Using Machine Learning: A Comparative Study of Hong Kong (N = 75) and France (N = 249)**

This repository contains the core training code for the V7 ensemble model used in a cross-cultural psychology study. The V7 model applies stacked gradient boosting (6 base learners + linear meta-learner) with 5-fold cross-validation to predict cross-cultural adaptation scores.

---

## 📁 Repository Structure

```
├── train_v7_complete_with_cv.py          # Train HK V7 ensemble model (5-fold CV)
├── train_france_v7_complete_with_cv.py   # Train France V7 ensemble model (5-fold CV)
├── archive/
│   └── generate_interaction_preserved_dataset.py  # Synthetic data generation (Copula + Interaction Regression)
├── requirements.txt                      # Python dependencies
├── LICENSE
└── README.md
```

---

## 🔬 Study Overview

| | Hong Kong | France |
|---|---|---|
| **N** | 75 | 249 |
| **Outcome** | Cross-cultural adaptation score | Cross-cultural adaptation score |
| **Predictors** | 11 psychological/social variables | 11 psychological/social variables |
| **CV R² (5-fold mean ± SD)** | .727 ± .001 | .742 ± .004 |
| **Test set R²** | .753 | .785 |
| **Algorithm** | V7 stacked ensemble (6 base learners + LinearRegression meta-learner) | Same |

### V7 Model Architecture

The V7 ensemble uses **stacking** with the following base learners:
- DeepFM (neural network with factorization machines)
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting (sklearn)
- Random Forest

A **LinearRegression** meta-learner combines the out-of-fold predictions from all 6 base learners. Feature importance is computed as a SHAP-weighted average across the tree-based models.

### Key Variables (11 predictors)

| Abbreviation | English | Chinese |
|---|---|---|
| CC | Cultural Contact | 文化接触 |
| SC | Social Contact | 社会接触 |
| SCn | Social Connectedness | 社会联结感 |
| FS | Family Support | 家庭支持 |
| Op | Openness | 开放性 |
| CM | Cultural Maintenance | 文化保持 |
| MHK | Months in HK/France | 来港/法时长 |
| CF | Communication Frequency | 家庭沟通频率 |
| CH | Communication Honesty | 沟通坦诚度 |
| Au | Autonomy | 自主权 |
| SM | Social Maintenance | 社会保持 |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare data

The training scripts expect real survey data files (not included in this repository due to participant privacy):

- **HK model** (`train_v7_complete_with_cv.py`): requires `CFA/HKN=75.xlsx`
- **France model** (`train_france_v7_complete_with_cv.py`): requires `CFA/Franchn=249.xlsx`

To generate synthetic training data (100k samples) from the real data using a Gaussian Copula + Interaction Regression approach:

```bash
python archive/generate_interaction_preserved_dataset.py
```

This saves the synthetic dataset to `data/processed/interaction_preserved_100k.csv`.

### 3. Train models

```bash
# Hong Kong sample (N=75), 5-fold cross-validation
python train_v7_complete_with_cv.py

# France sample (N=249), 5-fold cross-validation
python train_france_v7_complete_with_cv.py
```

Results are saved to `results/` (HK) and `france_models/` (France).

---

## 📊 Main Results

### Model Performance (V7, 5-Fold Cross-Validation on Synthetic Data + Evaluation on Real Data)

| Metric | Hong Kong | France |
|--------|-----------|--------|
| CV R² (mean ± SD) | .727 ± .001 | .742 ± .004 |
| Test R² (real data) | .753 | .785 |
| Test RMSE | 1.998 | 2.076 |
| Test MAE | 1.537 | 1.644 |

### Top Interaction Effects

**Hong Kong:** Cultural Contact × Openness (ΔR² = .029, *p* = .010)

**France:** Family Support × Cultural Maintenance (ΔR² = .021, *p* = .032)

---

## 📂 Data Availability

- **Raw data** (`CFA/HKN=75.xlsx`, `CFA/Franchn=249.xlsx`): **Not included** in the repository to protect participant privacy.
- **Simulated data** (`data/processed/interaction_preserved_100k.csv`): **Not included** due to file size. Regenerate using `archive/generate_interaction_preserved_dataset.py`.
- **Trained model files** (`*.pkl`): **Not included** due to file size. Retrain using the provided scripts.

---

## 🛠 Requirements

See `requirements.txt`. Key packages:

```
catboost>=1.2
xgboost>=2.0
lightgbm>=4.0
scikit-learn>=1.3
shap>=0.44
numpy>=1.24
pandas>=2.0
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
openpyxl>=3.1
```

Python 3.10+ recommended.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 📧 Citation

If you use this code in your research, please cite:

```bibtex
@misc{crosscultural2026,
  title   = {Cross-Cultural Adaptation Study: Hong Kong \& France — V7 Ensemble Model},
  year    = {2026},
  url     = {https://github.com/Misaka1082/V7-ensemble-cross-cultural-adaptation},
  note    = {V7 stacked ensemble training code for cross-cultural adaptation prediction}
}
```

---

## 🙏 Acknowledgements

Data collection was conducted with participants from Hong Kong (N = 75) and France (N = 249). All data are anonymized in accordance with ethical guidelines.
