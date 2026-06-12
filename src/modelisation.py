from features_eng import features_enginering

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # New import
from sklearn.tree import DecisionTreeClassifier # New import
import time


def pre_training():
    X, y = features_enginering()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Explicitly convert all numeric columns to float32 to ensure compatibility with SMOTE
    print('⚙️ Converting all numeric columns to float32 for SMOTE compatibility...')
    numeric_cols_train = X_train.select_dtypes(include=np.number).columns
    X_train[numeric_cols_train] = X_train[numeric_cols_train].astype(np.float32)

    numeric_cols_test = X_test.select_dtypes(include=np.number).columns
    X_test[numeric_cols_test] = X_test[numeric_cols_test].astype(np.float32)
    print('  ✅ Numeric columns converted to float32.')


    print(f'Train : {X_train.shape[0]:,} lignes  (retours: {y_train.mean():.2%})')
    print(f'Test  : {X_test.shape[0]:,} lignes  (retours: {y_test.mean():.2%})')

    # SMOTE sur le train uniquement
    print('🔁 Application SMOTE sur train...')
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f'Train après SMOTE : {X_train_res.shape[0]:,} lignes  (retours: {y_train_res.mean():.2%})')
    print('✅ Split + SMOTE OK')
    return (X_train_res, y_train_res, X_test, y_test)



models = {}
def training(X_train_res, y_train_res):
    train_times = {}

    # Removed scale_pos calculation as it was only for LightGBM/XGBoost.

    # ── Logistic Regression ───────────────────────────
    print('⚡ Logistic Regression...')
    t0 = time.time()
    lr_model = LogisticRegression(
        solver='liblinear', # good for small datasets, and also handles L1/L2 regularization
        random_state=42,
        class_weight='balanced', # handle imbalanced classes
        max_iter=1000
    )
    lr_model.fit(X_train_res, y_train_res)
    models['LogisticRegression'] = lr_model
    train_times['LogisticRegression'] = time.time() - t0
    print(f'  ✅ Logistic Regression — {train_times["LogisticRegression"]:.1f}s')

    # ── Decision Tree ───────────────────────────
    print('⚡ Decision Tree...')
    t0 = time.time()
    dt_model = DecisionTreeClassifier(
        max_depth=10, # Limiting depth to avoid overfitting and achieve AUC < 0.95
        random_state=42,
        class_weight='balanced' # handle imbalanced classes
    )
    dt_model.fit(X_train_res, y_train_res)
    models['DecisionTree'] = dt_model
    train_times['DecisionTree'] = time.time() - t0
    print(f'  ✅ Decision Tree  — {train_times["DecisionTree"]:.1f}s')

    # ── Random Forest (kept as it was already in the desired AUC range) ─────────────────────────────
    print('⚡ Random Forest...')
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_res, y_train_res)
    print("------------- here is the model is features-------\n")
    print(list(rf_model.feature_names_in_))
    models['RandomForest'] = rf_model
    train_times['RandomForest'] = time.time() - t0
    print(f'  ✅ RandomForest — {train_times["RandomForest"]:.1f}s')

    print('\n✅ Tous les modèles entraînés')

# ═══════════════════════════════════════════════
# 5.3 — Prédictions sur le test set
# ═══════════════════════════════════════════════

predictions = {}
def prediction(X_test):
    for name, model in models.items():
        predictions[name] = {
            'proba' : model.predict_proba(X_test)[:, 1],
            'label' : model.predict(X_test)
        }
        print(f'  ✅ {name} — prédictions OK')
        print('\n✅ Prédictions terminées')

import pickle as pk

def main():
    X_train_res, y_train_res, X_test, y_test = pre_training()
    training(X_train_res, y_train_res)
    try:
        with open("all_trained_models.pkl", "wb") as f:
            pk.dump(models, f)
    except Exception as e:
        print(f"error: {e}")
    prediction(X_test)
    
main()