# Cross-Cultural Adaptation Study: Hong Kong & France

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Predicting Cross-Cultural Adaptation Outcomes Using Machine Learning: A Comparative Study of Hong Kong (N = 75) and France (N = 249)**

This repository contains all analysis code, results, figures, and report materials for a cross-cultural psychology study that applies gradient boosting (V7 model) and linear regression to predict cross-cultural adaptation scores, with SHAP-based interpretability and interaction effect analysis.

---

## 📁 Repository Structure

```
├── CFA/                          # Confirmatory Factor Analysis scripts & data
│   ├── comprehensive_analysis.py
│   ├── advanced_analysis.py
│   ├── HKN=75.xlsx               # Hong Kong raw data (N=75)
│   └── Franchn=249.xlsx          # France raw data (N=249)
│
├── final_report/                 # Main paper sections (Markdown, APA 7th)
│   ├── Part1_研究概述与方法论.md
│   ├── Part2_条目分析与信度效度.md
│   ├── Part3_V7模型性能与特征分析.md
│   ├── Part4_交互效应与非线性关系.md
│   └── Part5_综合讨论与结论.md
│
├── appendices/                   # Supplementary tables (APA 7th format)
│   ├── Appendix_B_Second_Order_Interactions.md   # All 55 two-way interactions
│   ├── Appendix_C_Three_Way_Interactions.md      # All 165/57 three-way interactions
│   ├── Appendix_D_Simulated_Data_Validation.md   # Simulation quality checks
│   ├── Appendix_E_SHAP_Interaction_Matrices.md   # 11×11 SHAP interaction matrices
│   └── gen_appendices.py                         # Script to regenerate appendices
│
├── academic_figures/             # Publication-quality figures (PNG/SVG/PDF)
│   ├── Figure_4.3_Model_Performance_Comparison.*
│   ├── Figure_4.4_HK_Feature_Importance.*
│   ├── Figure_4.5_France_Feature_Importance.*
│   └── ...                       # Figures 4.3 – 4.20
│
├── france_data/                  # France sample processed data & statistics
│   ├── france_data_filtered_48months.xlsx
│   ├── descriptive_statistics.csv
│   ├── reliability_results.csv
│   └── ...
│
├── france_models/                # France model outputs & SHAP values
│   ├── shap_values_france.npy
│   ├── feature_importance.csv
│   └── ...
│
├── results/                      # Hong Kong model outputs & interaction results
│   ├── cv_results_75samples.csv
│   ├── feature_importance_75samples_cv.csv
│   ├── two_way_interactions.csv
│   ├── three_way_interactions.csv
│   ├── shap_values_75samples_cv.npy
│   └── comprehensive_analysis/   # France interaction analysis results
│
├── linear_regression_results/    # Linear regression tables & diagnostics
├── analysis_scripts/             # R scripts for psychometric analysis
├── archive/                      # Deprecated scripts (kept for reference)
│
├── train_v7_complete_with_cv.py          # Train HK V7 model (5-fold CV)
├── train_france_v7_complete_with_cv.py   # Train France V7 model (5-fold CV)
├── linear_regression_analysis.py         # Linear regression (HK & France)
├── analyze_interactions.py               # Interaction effect analysis
├── generate_academic_figures.py          # Generate all publication figures
└── requirements.txt                      # Python dependencies
```

---

## 🔬 Study Overview

| | Hong Kong | France |
|---|---|---|
| **N** | 75 | 249 |
| **Outcome** | Cross-cultural adaptation score | Cross-cultural adaptation score |
| **Predictors** | 11 psychological/social variables | 11 psychological/social variables |
| **Best model R²** | .847 (5-fold CV mean) | .831 (5-fold CV mean) |
| **Algorithm** | Gradient Boosting (CatBoost/XGBoost ensemble) | Same |

### Key Variables (11 predictors)

| Abbreviation | English | Chinese |
|---|---|---|
| CC | Cultural Contact | 文化接触 |
| SC | Social Contact | 社会接触 |
| SCn | Social Connectedness | 社会联结感 |
| FS | Family Support | 家庭支持 |
| Op | Openness | 开放性 |
| CM | Cultural Maintenance | 文化保持 |
| MHK | Months in HK/France | 来港时长 |
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

### 2. Train models

```bash
# Hong Kong sample (N=75), 5-fold cross-validation
python train_v7_complete_with_cv.py

# France sample (N=249), 5-fold cross-validation
python train_france_v7_complete_with_cv.py
```

### 3. Run interaction analysis

```bash
python analyze_interactions.py
```

### 4. Generate figures

```bash
python generate_academic_figures.py
```

### 5. Regenerate appendices

```bash
python appendices/gen_appendices.py
```

---

## 📊 Main Results

### Model Performance (V7, 5-Fold Cross-Validation)

| Metric | Hong Kong | France |
|--------|-----------|--------|
| R² (mean ± SD) | .847 ± .031 | .831 ± .018 |
| RMSE | 0.312 | 0.287 |
| MAE | 0.241 | 0.219 |

### Top Interaction Effects

**Hong Kong:** Cultural Contact × Openness (ΔR² = .029, *p* = .010)

**France:** Family Support × Cultural Maintenance (ΔR² = .021, *p* = .032)

---

## 📂 Data Availability

- **Raw data** (`CFA/HKN=75.xlsx`, `CFA/Franchn=249.xlsx`): **Not included** in the repository to protect participant privacy.
- **Simulated data** (`france_data/france_100k_48months.csv`, `data/processed/*.csv`): **Not included** in the repository due to file size (7–20 MB). Regenerate using `train_france_v7_complete_with_cv.py`.
- **Trained model files** (`*.pkl`): **Not included** due to file size (up to 3.2 GB). Retrain using the provided scripts.

---

## 📋 Appendices

All supplementary tables are in the `appendices/` folder in APA 7th edition Markdown format:

| File | Contents |
|------|----------|
| `Appendix_B_Second_Order_Interactions.md` | All 55 two-way interaction terms (HK & France) |
| `Appendix_C_Three_Way_Interactions.md` | All three-way interaction terms (HK: 165, France: 57) |
| `Appendix_D_Simulated_Data_Validation.md` | Simulation quality: correlation, KS tests, R² |
| `Appendix_E_SHAP_Interaction_Matrices.md` | 11×11 SHAP interaction strength & permutation *p*-values |

---

## 🛠 Requirements

See `requirements.txt`. Key packages:

```
catboost>=1.2
xgboost>=2.0
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

If you use this code or data in your research, please cite:

```bibtex
@misc{crosscultural2025,
  title   = {Cross-Cultural Adaptation Study: Hong Kong \& France},
  year    = {2025},
  url     = {https://github.com/Misaka1082/V7-ensemble-cross-cultural-adaptation},
  note    = {Analysis code and supplementary materials}
}
```

---

## 🙏 Acknowledgements

Data collection was conducted with participants from Hong Kong (N = 75) and France (N = 249). All data are anonymized in accordance with ethical guidelines.
