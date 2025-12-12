# optimizadores/gridSearchCV.py
from __future__ import annotations
from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from datetime import datetime

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
    scoring: str = "roc_auc",  # === CAMBIO: ahora optimiza ROC-AUC por defecto ===
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

    # === CAMBIO: scorer por defecto ROC-AUC ===
    scorer = "roc_auc" if scoring == "roc_auc" else scoring

    gs = GridSearchCV(est, grid, cv=cv, scoring=scorer, n_jobs=n_jobs)
    gs.fit(X, y)
    return {
        "best_estimator": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_score": gs.best_score_,
    }


def save_metrics_to_csv(results, model_name, filename_prefix="model_metrics"):
    # Obtener la fecha y hora actuales para el nombre del archivo
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generar el nombre del archivo con el nombre del modelo y la fecha
    filename = f"{filename_prefix}_{model_name}_{current_time}.csv"

    # Lista para almacenar los resultados de las métricas
    metrics_list = []

    for iter_idx, result in enumerate(results, start=1):
        # === CAMBIO: usar F1 macro ===
        f1_macro = f1_score(result['y_true'], result['y_pred'], average='macro')
        roc_auc = roc_auc_score(result['y_true'], result['y_pred_prob'])
        conf_matrix = confusion_matrix(result['y_true'], result['y_pred']).tolist()
        hyperparameters = result['hyperparameters']

        metrics_list.append({
            'iteracion': iter_idx,                # id de iteración
            'f1_score_macro': round(f1_macro, 4), # CAMBIO EN EL NOMBRE DE LA COLUMNA
            'roc_auc': round(roc_auc, 4),
            'conf_matrix': conf_matrix,
            'mean_cv_score': result.get('mean_test_score'),
            'std_cv_score': result.get('std_test_score'),
            **hyperparameters
        })

    # Convertir la lista de resultados a un DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Guardar el DataFrame en un archivo CSV
    metrics_df.to_csv(filename, index=False)

    print(f"Metrics saved to {filename}")