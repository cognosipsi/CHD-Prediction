# predictores/mlp.py
from typing import Sequence, Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def mlp_train(
    X_train, y_train, X_test, y_test,
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    tol=1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Entrena un MLP y retorna métricas (accuracy, precision, recall, f1, auc).
    """
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=early_stopping,
        tol=tol,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None
    return y_pred, y_proba, y_test

def mlp_evaluator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    mask: Sequence[int],
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    early_stopping=True,
    tol=1e-4,
) -> float:
    """
    Entrena un MLP rápido en el subset de features indicado por 'mask' y devuelve accuracy de validación.
    Diseñado para ser llamado dentro de M-ABC.
    """
    idx = np.where(np.array(mask) == 1)[0]
    Xtr = X_train[:, idx] if idx.size > 0 else X_train
    Xv  = X_val[:,   idx] if idx.size > 0 else X_val

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=early_stopping,
        tol=tol,
    )
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xv)
    return float(accuracy_score(y_val, y_pred))