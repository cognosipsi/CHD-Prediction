# predictores/mlp.py
from typing import Sequence, Tuple, Optional
import numpy as np
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

def mlp_evaluator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    *,
    mask: Optional[Sequence[int]] = None,
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    early_stopping=True,
    tol=1e-4,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Entrena un MLP en el subset de features indicado por 'mask' y devuelve
    (y_pred, y_proba). 'mask' puede ser:
      - None -> usar todas las columnas
      - máscara booleana/lista de 0/1
      - lista/array de índices seleccionados
    """
    n_features = X_train.shape[1]

    # Normalizar distintos formatos de mask
    if mask is None:
        idx = np.arange(n_features)
    else:
        m = np.asarray(mask)
        if m.dtype == bool:
            idx = np.where(m)[0]
        elif m.size > 0 and set(np.unique(m)).issubset({0, 1}):
            idx = np.where(m == 1)[0]
        else:
            # asumir lista de índices
            idx = np.asarray(m, dtype=int)

    # Si no hay índices seleccionados, usamos todas las características
    if idx.size == 0:
        idx = np.arange(n_features)
    Xtr = X_train[:, idx]
    Xv  = X_val[:,   idx]

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=early_stopping,
        tol=tol,
    )
    # Evitar spam de ConvergenceWarning localmente (opcional)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xv)

    # Obtener probabilidades si es posible (clase positiva)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(Xv)
            # si salida multiclass, tomar columna 1 si existe, si no tomar la columna de la clase mayoritaria
            if proba.shape[1] == 2:
                y_proba = proba[:, 1]
            else:
                # fallback: usar probabilidad de la clase predicha
                y_proba = proba[np.arange(proba.shape[0]), clf.predict(Xv)]
        except Exception:
            y_proba = None
    elif hasattr(clf, "decision_function"):
        try:
            df = clf.decision_function(Xv)
            # si devuelve (n_samples, n_classes) reducir a una única puntuación
            if df.ndim == 1:
                y_proba = 1.0 / (1.0 + np.exp(-df))
            else:
                # tomar la segunda columna cuando tenga sentido
                if df.shape[1] >= 2:
                    y_proba = 1.0 / (1.0 + np.exp(-df[:, 1]))
                else:
                    y_proba = 1.0 / (1.0 + np.exp(-df.ravel()))
        except Exception:
            y_proba = None

    return y_pred, y_proba