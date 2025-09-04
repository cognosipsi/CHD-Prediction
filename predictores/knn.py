# predictores/knn.py
from __future__ import annotations
from typing import Tuple, Sequence, Optional
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def _apply_mask(X: np.ndarray, mask: Optional[Sequence[int]]) -> np.ndarray:
    if mask is None:
        return X
    idx = np.where(np.array(mask) == 1)[0]
    return X if idx.size == 0 else X[:, idx]

def knn_evaluator(X_train: np.ndarray, X_test: np.ndarray,
                  y_train: np.ndarray, y_test: np.ndarray,
                  mask: Optional[Sequence[int]] = None,
                  n_neighbors: int = 5) -> float:
    """
    Devuelve accuracy de validación como fitness (mayor = mejor).
    """
    Xt = _apply_mask(X_train, mask)
    Xv = _apply_mask(X_test, mask)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(Xt, y_train)
    y_pred = model.predict(Xv)
    return float(accuracy_score(y_test, y_pred))

def knn_train(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_neighbors: int = 3) -> Tuple[float, float, float, float, float]:
    """
    Entrena el modelo final y entrega métricas estándar.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # Para AUC necesitamos probabilidades; si no es binario o falla, devolvemos NaN
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return acc, prec, rec, f1, auc
