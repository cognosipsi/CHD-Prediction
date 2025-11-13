# predictores/knn.py
from __future__ import annotations
from typing import Tuple, Sequence, Optional
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def _apply_mask(X: np.ndarray, mask: Optional[Sequence[int]]) -> np.ndarray:
    if mask is None:
        return X
    idx = np.where(np.array(mask) == 1)[0]
    return X if idx.size == 0 else X[:, idx]

def knn_evaluator(X_train: np.ndarray, X_test: np.ndarray,
                  y_train: np.ndarray, y_test: np.ndarray,
                  mask: Optional[Sequence[int]] = None,
                  n_neighbors: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve accuracy de validación como fitness (mayor = mejor).
    """
    Xt = _apply_mask(X_train, mask)
    Xv = _apply_mask(X_test, mask)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(Xt, y_train)
    y_pred = model.predict(Xv)
    try:
        y_prob = model.predict_proba(Xv)[:, 1]
    except Exception:
        y_prob = None
    return y_pred, y_prob