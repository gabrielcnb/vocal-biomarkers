"""Treinamento dos 3 classificadores: SVM, Random Forest, XGBoost.

Todos usam GridSearchCV com StratifiedKFold(10) e scoring='roc_auc'.
class_weight='balanced' handles the imbalance (147 PD vs 48 healthy).
"""
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from config import (
    SVM_MODEL,
    RF_MODEL,
    XGB_MODEL,
    BEST_MODEL,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    CV_FOLDS,
)


def train_all(X_train, y_train, scaler, feature_names):
    """Treina SVM, RF e XGBoost com GridSearchCV.

    Returns:
        dict: {nome: (modelo, best_score)} para cada classificador
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    # --- SVM ---
    print("\n" + "=" * 60)
    print("Treinando SVM (RBF kernel)...")
    print("=" * 60)
    svm_params = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
    }
    svm = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        svm_params,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    svm.fit(X_train, y_train)
    print(f"SVM melhor AUC (CV): {svm.best_score_:.4f}")
    print(f"SVM melhores params: {svm.best_params_}")
    joblib.dump(svm.best_estimator_, SVM_MODEL)
    results["SVM"] = (svm.best_estimator_, svm.best_score_)

    # --- Random Forest ---
    print("\n" + "=" * 60)
    print("Treinando Random Forest...")
    print("=" * 60)
    rf_params = {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, None],
    }
    rf = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        rf_params,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    rf.fit(X_train, y_train)
    print(f"RF melhor AUC (CV): {rf.best_score_:.4f}")
    print(f"RF melhores params: {rf.best_params_}")
    joblib.dump(rf.best_estimator_, RF_MODEL)
    results["Random Forest"] = (rf.best_estimator_, rf.best_score_)

    # --- XGBoost ---
    print("\n" + "=" * 60)
    print("Treinando XGBoost...")
    print("=" * 60)
    # scale_pos_weight for the imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
    }
    xgb = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        xgb_params,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    xgb.fit(X_train, y_train)
    print(f"XGBoost melhor AUC (CV): {xgb.best_score_:.4f}")
    print(f"XGBoost melhores params: {xgb.best_params_}")
    joblib.dump(xgb.best_estimator_, XGB_MODEL)
    results["XGBoost"] = (xgb.best_estimator_, xgb.best_score_)

    # --- Salvar o melhor modelo ---
    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"\nMelhor modelo: {best_name} (CV AUC: {results[best_name][1]:.4f})")
    joblib.dump(best_model, BEST_MODEL)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    print(f"\nModelos salvos em {MODELS_DIR}/")
    return results
