from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils.evaluacion import compute_classification_metrics

# Parámetros por defecto (sin 'use_label_encoder', deprecado)
DEFAULT_PARAMS: Dict[str, Any] = dict(
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

def build_xgb(custom_params: Optional[Dict[str, Any]] = None) -> XGBClassifier:
    """
    Crea una instancia de XGBClassifier con parámetros por defecto + overrides.
    Filtra claves desconocidas para evitar mensajes de 'not used'.
    """
    params = DEFAULT_PARAMS.copy()
    if custom_params:
        accepted = XGBClassifier().get_params()
        params.update({k: v for k, v in custom_params.items() if k in accepted})
    return XGBClassifier(**params)

def fit_and_predict(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Entrena XGB y devuelve un diccionario de métricas:
    {"accuracy", "precision", "recall", "f1", "auc"}.
    """
    model = build_xgb(params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    return metrics

# --- Compatibilidad retro opcional ---
def fit_and_predict_raw(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, XGBClassifier]:
    """
    Versión antigua: devuelve (y_pred, y_proba, model).
    Úsala solo si tienes código viejo que lo necesite.
    """
    model = build_xgb(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba, model
