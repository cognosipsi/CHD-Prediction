# optimizadores/gridSearchCV.py
from __future__ import annotations
from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # xgboost puede no estar instalado
    XGBClassifier = None

# Alias amigables
_ALIASES = {
    "knn": "knn",
    "mlp": "mlp",
    "xgb": "xgb",
    "xgboost": "xgb",
    "transformer": "transformer",
}

def get_param_grid(model: str) -> Dict[str, list]:
    m = _ALIASES.get(model.lower(), model.lower())
    if m == "knn":
        return {
            "n_neighbors": [3, 5, 10],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "euclidean"],
        }
    if m == "mlp":
        return {
            "hidden_layer_sizes": [(50,), (100,), (150,)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "max_iter": [200, 500],
        }
    if m == "xgb":
        return {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [100, 200, 500],
        }
    if m == "transformer":
        # No es un estimador sklearn; se usará manualmente en el pipeline
        return {
            "epochs": [10, 20],
            "lr": [1e-3, 1e-4],
            "batch_size": [32, 64],
            "d_model": [64, 128],
            "nhead": [4, 8],
            "num_layers": [2, 4],
            "dropout": [0.2, 0.5],
        }
    raise ValueError(f"Modelo no soportado para grid: {model}")

def _make_estimator(model: str, **base) -> Optional[object]:
    m = _ALIASES.get(model.lower(), model.lower())
    if m == "knn":
        return KNeighborsClassifier(**base)
    if m == "mlp":
        return MLPClassifier(**base)
    if m == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost no está instalado en el entorno.")
        default = dict(eval_metric="logloss")
        default.update(base or {})
        return XGBClassifier(**default)
    # transformer no es sklearn; devolver None
    return None

def run_grid_search(
    model: str,
    X,
    y,
    *,
    cv: int = 5,
    scoring: str = "f1",
    param_grid: Optional[Dict[str, list]] = None,
    n_jobs: int = -1,
    **base,
) -> Optional[Dict[str, Any]]:
    """
    Ejecuta GridSearchCV para KNN/MLP/XGB y devuelve dict con best_estimator/best_params/best_score.
    Para 'transformer' devuelve None (debe hacerse búsqueda manual).
    """
    est = _make_estimator(model, **base)
    if est is None:
        return None
    grid = param_grid or get_param_grid(model)
    scorer = make_scorer(f1_score) if scoring == "f1" else scoring
    gs = GridSearchCV(est, grid, cv=cv, scoring=scorer, n_jobs=n_jobs)
    gs.fit(X, y)
    return {
        "best_estimator": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_score": gs.best_score_,
    }
