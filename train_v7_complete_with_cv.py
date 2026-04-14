"""
V7 Ensemble Model — Full Training Pipeline with 5-Fold Cross-Validation
=======================================================================
Dataset : 100k simulated samples derived from 75 real HK samples
          (months_in_hk filtered to 1-48 months)
Runtime : ~1-2 hours (CPU); ~20-40 min (GPU)

Usage
-----
1. Generate the training data first:
       python archive/generate_interaction_preserved_dataset.py
   This writes  data/processed/interaction_preserved_100k_48months.csv
   and          data/processed/real_data_filtered_48months.xlsx

2. Run this script:
       python train_v7_complete_with_cv.py

Outputs (saved to results/)
---------------------------
  cv_results_75samples.csv          – per-fold R2 for every base learner
  shap_values_75samples_cv.npy      – ensemble SHAP values on real test set
  feature_importance_75samples_cv.csv
  v7_final_model_hk.pkl             – final model bundle (scaler + 6 learners + meta)
  v7_models_75samples_5fold.pkl     – all 5 fold model bundles
  training_report_with_cv.txt       – summary report
"""

import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============================================================================
# Step 1 – Load data
# ============================================================================
print("=" * 80)
print("V7 Ensemble Model — Full Training Pipeline (5-Fold CV)")
print("=" * 80)

start_time = time.time()

print(f"\n[Step 1/12] Loading data  ({time.strftime('%Y-%m-%d %H:%M:%S')})")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "data", "processed", "interaction_preserved_100k_48months.csv")
REAL_XLSX = os.path.join(BASE_DIR, "data", "processed", "real_data_filtered_48months.xlsx")

# Validate paths before loading
for path in (TRAIN_CSV, REAL_XLSX):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired data file not found: {path}\n"
            "Please run  archive/generate_interaction_preserved_dataset.py  first."
        )

df_train = pd.read_csv(TRAIN_CSV)
df_real = pd.read_excel(REAL_XLSX)

# Rename Chinese columns to English
column_mapping = {
    "跨文化适应程度": "cross_cultural_adaptation",
    "文化保持": "cultural_maintenance",
    "社会保持": "social_maintenance",
    "文化接触": "cultural_contact",
    "社会接触": "social_contact",
    "家庭支持": "family_support",
    "家庭沟通频率": "family_communication_frequency",
    "沟通坦诚度": "communication_honesty",
    "自主权": "autonomy",
    "社会联结感": "social_connectedness",
    "开放性": "openness",
    "来港时长": "months_in_hk",
}
df_real = df_real.rename(columns=column_mapping)

feature_cols = [
    "cultural_maintenance", "social_maintenance", "cultural_contact",
    "social_contact", "family_support", "family_communication_frequency",
    "communication_honesty", "autonomy", "social_connectedness",
    "openness", "months_in_hk",
]

# Validate that all required columns exist
missing = [c for c in feature_cols + ["cross_cultural_adaptation"] if c not in df_real.columns]
if missing:
    raise ValueError(
        f"The following columns are missing from the real data after renaming: {missing}\n"
        "Check that column_mapping covers all Chinese column names in the xlsx file."
    )

print(f"  Training samples : {len(df_train):,}")
print(f"  Real test samples: {len(df_real)}")
print(f"  Features         : {len(feature_cols)}")

# ============================================================================
# Step 2 – Prepare 5-fold CV
# ============================================================================
print("\n[Step 2/12] Setting up 5-fold cross-validation")

X_full = df_train[feature_cols].values
y_full = df_train["cross_cultural_adaptation"].values
X_test = df_real[feature_cols].values
y_test = df_real["cross_cultural_adaptation"].values

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {k: [] for k in ("deepfm", "xgb", "lgb", "cb", "gbm", "rf", "v7")}

print(f"  Train per fold : ~{int(len(X_full) * 0.8):,}")
print(f"  Val   per fold : ~{int(len(X_full) * 0.2):,}")


# ── DeepFM definition ────────────────────────────────────────────────────────
class DeepFM(nn.Module):
    """Factorisation Machine + Deep Neural Network hybrid regressor."""

    def __init__(self, n_features: int, embedding_dim: int = 16):
        super().__init__()
        self.fm_linear = nn.Linear(n_features, 1)
        self.fm_embeddings = nn.Parameter(torch.randn(n_features, embedding_dim))
        self.dnn = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.fm_linear(x)
        x_exp = x.unsqueeze(2)
        emb_exp = self.fm_embeddings.unsqueeze(0).expand(x.size(0), -1, -1)
        we = x_exp * emb_exp
        fm_part = 0.5 * torch.sum(
            torch.sum(we, dim=1) ** 2 - torch.sum(we ** 2, dim=1), dim=1, keepdim=True
        )
        return linear_part + fm_part + self.dnn(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ============================================================================
# Steps 3-7 – 5-fold cross-validation
# ============================================================================
print("\n[Steps 3-7/12] 5-fold cross-validation training (~1 hour on CPU)")

fold_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/5  —  {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y_full[train_idx], y_full[val_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── 1. DeepFM ────────────────────────────────────────────
    print(f"  [{fold}/5] Training DeepFM ...")
    deepfm = DeepFM(len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(deepfm.parameters(), lr=1e-3)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train).unsqueeze(1)),
        batch_size=256, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val).unsqueeze(1)),
        batch_size=256,
    )

    best_val_loss, patience_counter, best_state = float("inf"), 0, None
    for epoch in range(100):
        deepfm.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(deepfm(bx), by).backward()
            optimizer.step()

        deepfm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                val_loss += criterion(deepfm(bx.to(device)), by.to(device)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in deepfm.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    deepfm.load_state_dict(best_state)
    deepfm.eval()
    with torch.no_grad():
        deepfm_val_pred  = deepfm(torch.FloatTensor(X_val_s).to(device)).cpu().numpy().flatten()
        deepfm_test_pred = deepfm(torch.FloatTensor(X_test_s).to(device)).cpu().numpy().flatten()

    cv_results["deepfm"].append(r2_score(y_val, deepfm_val_pred))
    print(f"    DeepFM      R2: {cv_results['deepfm'][-1]:.4f}")

    # ── 2. XGBoost ───────────────────────────────────────────
    print(f"  [{fold}/5] Training XGBoost ...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1, early_stopping_rounds=50,
    )
    xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    xgb_val_pred  = xgb_model.predict(X_val_s)
    xgb_test_pred = xgb_model.predict(X_test_s)
    cv_results["xgb"].append(r2_score(y_val, xgb_val_pred))
    print(f"    XGBoost     R2: {cv_results['xgb'][-1]:.4f}")

    # ── 3. LightGBM ──────────────────────────────────────────
    print(f"  [{fold}/5] Training LightGBM ...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.1, num_leaves=31,
        feature_fraction=0.7, bagging_fraction=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(
        X_train_s, y_train, eval_set=[(X_val_s, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    lgb_val_pred  = lgb_model.predict(X_val_s)
    lgb_test_pred = lgb_model.predict(X_test_s)
    cv_results["lgb"].append(r2_score(y_val, lgb_val_pred))
    print(f"    LightGBM    R2: {cv_results['lgb'][-1]:.4f}")

    # ── 4. CatBoost ──────────────────────────────────────────
    print(f"  [{fold}/5] Training CatBoost ...")
    cb_model = cb.CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, random_state=42, verbose=False,
    )
    cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val), early_stopping_rounds=50)
    cb_val_pred  = cb_model.predict(X_val_s)
    cb_test_pred = cb_model.predict(X_test_s)
    cv_results["cb"].append(r2_score(y_val, cb_val_pred))
    print(f"    CatBoost    R2: {cv_results['cb'][-1]:.4f}")

    # ── 5. GBM ───────────────────────────────────────────────
    print(f"  [{fold}/5] Training GBM ...")
    gbm_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=42,
    )
    gbm_model.fit(X_train_s, y_train)
    gbm_val_pred  = gbm_model.predict(X_val_s)
    gbm_test_pred = gbm_model.predict(X_test_s)
    cv_results["gbm"].append(r2_score(y_val, gbm_val_pred))
    print(f"    GBM         R2: {cv_results['gbm'][-1]:.4f}")

    # ── 6. RandomForest ──────────────────────────────────────
    print(f"  [{fold}/5] Training RandomForest ...")
    rf_model = RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_split=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf_model.fit(X_train_s, y_train)
    rf_val_pred  = rf_model.predict(X_val_s)
    rf_test_pred = rf_model.predict(X_test_s)
    cv_results["rf"].append(r2_score(y_val, rf_val_pred))
    print(f"    RandomForest R2: {cv_results['rf'][-1]:.4f}")

    # ── 7. Stacking meta-learner ─────────────────────────────
    print(f"  [{fold}/5] Stacking ...")
    meta_train = np.column_stack([deepfm_val_pred, xgb_val_pred, lgb_val_pred,
                                   cb_val_pred, gbm_val_pred, rf_val_pred])
    meta_test  = np.column_stack([deepfm_test_pred, xgb_test_pred, lgb_test_pred,
                                   cb_test_pred, gbm_test_pred, rf_test_pred])
    meta_model = LinearRegression()
    meta_model.fit(meta_train, y_val)
    cv_results["v7"].append(r2_score(y_val, meta_model.predict(meta_train)))
    print(f"    V7 ensemble  R2: {cv_results['v7'][-1]:.4f}")

    fold_models.append({
        "scaler": scaler, "deepfm": deepfm,
        "xgb": xgb_model, "lgb": lgb_model, "cb": cb_model,
        "gbm": gbm_model, "rf": rf_model, "meta": meta_model,
    })
    print(f"\nFold {fold}/5 done — elapsed: {(time.time() - start_time)/60:.1f} min")

# ============================================================================
# Step 8 – CV summary
# ============================================================================
print("\n" + "=" * 80)
print("[Step 8/12] Cross-validation summary")
print("=" * 80)

cv_summary = pd.DataFrame(cv_results)
print("\nPer-fold R2:")
print(cv_summary.to_string(index=False))
print("\nMean ± Std:")
for name, vals in cv_results.items():
    print(f"  {name:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
cv_summary.to_csv(os.path.join(BASE_DIR, "results", "cv_results_75samples.csv"), index=False)

# ============================================================================
# Step 9 – Train final model on full data
# ============================================================================
print("\n[Step 9/12] Training final model on full 100k dataset")

scaler_final = StandardScaler()
X_full_s = scaler_final.fit_transform(X_full)
X_test_s_final = scaler_final.transform(X_test)

print("  Training final DeepFM ...")
deepfm_final = DeepFM(len(feature_cols)).to(device)
opt_f = torch.optim.Adam(deepfm_final.parameters(), lr=1e-3)
crit_f = nn.MSELoss()
ld_f = DataLoader(
    TensorDataset(torch.FloatTensor(X_full_s), torch.FloatTensor(y_full).unsqueeze(1)),
    batch_size=256, shuffle=True,
)
for _ in range(50):
    deepfm_final.train()
    for bx, by in ld_f:
        bx, by = bx.to(device), by.to(device)
        opt_f.zero_grad()
        crit_f(deepfm_final(bx), by).backward()
        opt_f.step()
deepfm_final.eval()

print("  Training final XGBoost ...")
xgb_final = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)
xgb_final.fit(X_full_s, y_full, verbose=False)

print("  Training final LightGBM ...")
lgb_final = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.1, num_leaves=31,
    feature_fraction=0.7, bagging_fraction=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
)
lgb_final.fit(X_full_s, y_full)

print("  Training final CatBoost ...")
cb_final = cb.CatBoostRegressor(
    iterations=500, learning_rate=0.03, depth=6,
    l2_leaf_reg=3.0, random_state=42, verbose=False,
)
cb_final.fit(X_full_s, y_full)

print("  Training final GBM ...")
gbm_final = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.1, max_depth=5,
    subsample=0.8, random_state=42,
)
gbm_final.fit(X_full_s, y_full)

print("  Training final RandomForest ...")
rf_final = RandomForestRegressor(
    n_estimators=500, max_depth=15, min_samples_split=5,
    max_features="sqrt", random_state=42, n_jobs=-1,
)
rf_final.fit(X_full_s, y_full)

# Build meta-learner from out-of-fold (OOF) predictions
print("  Building meta-learner from OOF predictions ...")
oof_deepfm = np.zeros(len(X_full))
oof_xgb    = np.zeros(len(X_full))
oof_lgb    = np.zeros(len(X_full))
oof_cb     = np.zeros(len(X_full))
oof_gbm    = np.zeros(len(X_full))
oof_rf     = np.zeros(len(X_full))

for i, (_, val_idx) in enumerate(kfold.split(X_full)):
    fm = fold_models[i]
    Xv = fm["scaler"].transform(X_full[val_idx])
    with torch.no_grad():
        oof_deepfm[val_idx] = fm["deepfm"](torch.FloatTensor(Xv).to(device)).cpu().numpy().flatten()
    oof_xgb[val_idx] = fm["xgb"].predict(Xv)
    oof_lgb[val_idx] = fm["lgb"].predict(Xv)
    oof_cb[val_idx]  = fm["cb"].predict(Xv)
    oof_gbm[val_idx] = fm["gbm"].predict(Xv)
    oof_rf[val_idx]  = fm["rf"].predict(Xv)

meta_oof = np.column_stack([oof_deepfm, oof_xgb, oof_lgb, oof_cb, oof_gbm, oof_rf])
meta_final = LinearRegression()
meta_final.fit(meta_oof, y_full)
print(f"  Meta-learner coefficients: {meta_final.coef_}")

# Save final model bundle
final_bundle = {
    "scaler": scaler_final, "deepfm": deepfm_final,
    "xgb": xgb_final, "lgb": lgb_final, "cb": cb_final,
    "gbm": gbm_final, "rf": rf_final, "meta": meta_final,
}
joblib.dump(final_bundle, os.path.join(BASE_DIR, "results", "v7_final_model_hk.pkl"))
print("  Final model bundle saved.")

# ============================================================================
# Step 10 – SHAP analysis
# ============================================================================
print("\n[Step 10/12] SHAP explainability analysis")

# Use fold-0 model with its own scaler (avoids leaked scaler from loop variable)
model_for_shap = fold_models[0]
X_test_s_shap = model_for_shap["scaler"].transform(X_test)

shap_values_dict = {}
for name, mdl in [
    ("xgb", model_for_shap["xgb"]),
    ("lgb", model_for_shap["lgb"]),
    ("cb",  model_for_shap["cb"]),
    ("gbm", model_for_shap["gbm"]),
    ("rf",  model_for_shap["rf"]),
]:
    try:
        explainer = shap.TreeExplainer(mdl)
        shap_values_dict[name] = explainer.shap_values(X_test_s_shap)
        print(f"  {name} SHAP values computed.")
    except Exception as exc:
        print(f"  {name} SHAP failed: {exc}")

if shap_values_dict:
    w = 1.0 / len(shap_values_dict)
    shap_values_ensemble = sum(w * sv for sv in shap_values_dict.values())

    feature_importance = np.abs(shap_values_ensemble).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": feature_importance,
    }).sort_values("shap_importance", ascending=False)

    print("\nFeature importance (SHAP):")
    print(importance_df.to_string(index=False))

    np.save(os.path.join(BASE_DIR, "results", "shap_values_75samples_cv.npy"), shap_values_ensemble)
    importance_df.to_csv(
        os.path.join(BASE_DIR, "results", "feature_importance_75samples_cv.csv"), index=False
    )
else:
    print("  WARNING: SHAP computation failed for all models — skipping SHAP outputs.")
    importance_df = pd.DataFrame({"feature": feature_cols, "shap_importance": [float("nan")] * len(feature_cols)})

# ============================================================================
# Step 11 – Evaluate on test set
# ============================================================================
print("\n[Step 11/12] Evaluating on real test set (average of 5 fold models)")

test_preds = []
for fm in fold_models:
    Xt = fm["scaler"].transform(X_test)
    with torch.no_grad():
        dp = fm["deepfm"](torch.FloatTensor(Xt).to(device)).cpu().numpy().flatten()
    meta_feat = np.column_stack([
        dp,
        fm["xgb"].predict(Xt),
        fm["lgb"].predict(Xt),
        fm["cb"].predict(Xt),
        fm["gbm"].predict(Xt),
        fm["rf"].predict(Xt),
    ])
    test_preds.append(fm["meta"].predict(meta_feat))

final_pred = np.mean(test_preds, axis=0)
final_r2   = r2_score(y_test, final_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
final_mae  = mean_absolute_error(y_test, final_pred)

print(f"\n  R2   = {final_r2:.4f}")
print(f"  RMSE = {final_rmse:.4f}")
print(f"  MAE  = {final_mae:.4f}")

# ============================================================================
# Step 12 – Save report and fold models
# ============================================================================
print("\n[Step 12/12] Saving report and fold models")

elapsed = time.time() - start_time

report = f"""
{'='*80}
V7 Ensemble Model — Training Report (5-Fold CV)
{'='*80}

Completed : {time.strftime('%Y-%m-%d %H:%M:%S')}
Elapsed   : {elapsed/60:.1f} min

5-Fold Cross-Validation R2 (mean ± std):
  DeepFM      : {np.mean(cv_results['deepfm']):.4f} ± {np.std(cv_results['deepfm']):.4f}
  XGBoost     : {np.mean(cv_results['xgb']):.4f} ± {np.std(cv_results['xgb']):.4f}
  LightGBM    : {np.mean(cv_results['lgb']):.4f} ± {np.std(cv_results['lgb']):.4f}
  CatBoost    : {np.mean(cv_results['cb']):.4f} ± {np.std(cv_results['cb']):.4f}
  GBM         : {np.mean(cv_results['gbm']):.4f} ± {np.std(cv_results['gbm']):.4f}
  RandomForest: {np.mean(cv_results['rf']):.4f} ± {np.std(cv_results['rf']):.4f}
  V7 Ensemble : {np.mean(cv_results['v7']):.4f} ± {np.std(cv_results['v7']):.4f}

Final Test-Set Performance (5-fold average, N=75 real HK samples):
  R2   = {final_r2:.4f}
  RMSE = {final_rmse:.4f}
  MAE  = {final_mae:.4f}

Saved files (results/):
  cv_results_75samples.csv
  shap_values_75samples_cv.npy
  feature_importance_75samples_cv.csv
  v7_final_model_hk.pkl
  v7_models_75samples_5fold.pkl
  training_report_with_cv.txt
{'='*80}
"""

print(report)

with open(os.path.join(BASE_DIR, "results", "training_report_with_cv.txt"), "w", encoding="utf-8") as f:
    f.write(report)

joblib.dump(fold_models, os.path.join(BASE_DIR, "results", "v7_models_75samples_5fold.pkl"))

print("All outputs saved to results/")
print("=" * 80)
