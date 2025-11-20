# pipelines/xgb_pipeline.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import time

# === Preprocesamiento ===
from preprocesamiento.lectura_datos import load_data
from preprocesamiento.codificacion import encode_features
from preprocesamiento.escalado import scale_features
from preprocesamiento.division_dataset import split_data

# === Predictores ===
from predictores.xgb import build_xgb

# === Selectores ===
from selectores.mabc import MABCFeatureSelector
from selectores.bsocv import BSOFeatureSelector
from selectores.woa import WOAFeatureSelector
from selectores.eliminacionpearson import PearsonRedundancyEliminator

#Optimizadores
from optimizadores.gridSearchCV import save_metrics_to_csv

# === Reporte (igual que MLP) ===
from utils.evaluacion import print_from_pipeline_result, compute_classification_metrics

# === sklearn / imblearn ===

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV


"""def _apply_mask_df(X_df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    mask = np.asarray(mask).astype(int).ravel()
    if mask.shape[0] != X_df.shape[1]:
        raise ValueError(f"Máscara de longitud {mask.shape[0]} no coincide con #cols={X_df.shape[1]}")
    if mask.sum() == 0:
        # evita conjunto vacío
        mask[np.random.randint(0, mask.shape[0])] = 1
    cols = X_df.columns[mask == 1].tolist()
    return X_df.loc[:, cols]
"""

def xgb_pipeline(
    file_path: str,
    selector: Optional[str] = "none",
    *,
    encoding_method: str = "labelencoder",
    scaler_type: str = "standard",
    redundancy: Optional[str] = "none",
    xgb_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    use_smote: bool = True,
    optimizer: Optional[str] = "none", 
    **selector_params,
) -> Dict[str, Any]:
    """
    Pipeline minimal de XGBoost con imblearn:

    - Carga y codifica datos.
    - (Opcional) PearsonRedundancyEliminator.
    - (Opcional) Selector wrapper: M-ABC, WOA o BSO-CV.
    - Escalado con scale_features().
    - (Opcional) SMOTE dentro del pipeline.
    - Clasificador XGB construido con build_xgb().
    - (Opcional) GridSearchCV sobre clf__*.

    Devuelve un diccionario sencillo con métricas, nombre del selector, etc.
    """

    t0 = time.time()

    # Compatibilidad hacia atrás: si alguien pasó selector_params=dict(...)
    if "selector_params" in selector_params and isinstance(selector_params["selector_params"], dict):
        selector_params = dict(selector_params["selector_params"])

    # 1) Carga + codificación
    df = load_data(file_path)
    df = encode_features(df, encoding_method=encoding_method)

    # 2) Separar X/y (asumimos 'chd' como target)
    if 'chd' not in df.columns:
        raise KeyError("No se encontró la columna objetivo 'chd' en el dataset.")
    y = df['chd'].values
    X_df = df.drop(columns=['chd'])

    # 3) Construcción de pasos del pipeline
    steps = []

    # 3.1) Eliminación de redundancias (PearsonRedundancyEliminator)
    if redundancy is not None and str(redundancy).lower() not in {"none", "sin", "no"}:
        steps.append(("redundancy", PearsonRedundancyEliminator(metodo=redundancy)))

    # 6) Selección de características (opcional) sobre DataFrame (para preservar nombres)
    sel = (selector or "none").strip().lower() if selector is not None else "none"
    selector_est = None
    selector_name = "Sin selector (todas las variables)"

    if sel in {'m-abc', 'mabc', 'm_abc'}:
        population_size = int(selector_params.get("population_size", 20))
        max_iter = int(selector_params.get("max_iter", 30))
        limit = int(selector_params.get("limit", 5))
        patience = int(selector_params.get("patience", 10))
        cv_folds = int(selector_params.get("cv_folds", 5))
        knn_k = int(selector_params.get("knn_k", 5))
        random_state_m  = selector_params.get("random_state", 42)
        verbose = int(selector_params.get("verbose", 0))

        mabc = MABCFeatureSelector(
            knn_k=knn_k,
            population_size=population_size,
            max_iter=max_iter,
            limit=limit,
            patience=patience,
            cv_folds=cv_folds,
            random_state=random_state_m,
            verbose=verbose,
        )
        steps.append(("selector", mabc))
        selector_name = f"M-ABC(pop={population_size}, iters={max_iter}, cv={cv_folds})"

    elif sel in {'woa', 'whale', 'ballenas'}:
        # WOA con fitness por CV; si tu implementación soporta estimator pásalo vía selector_params['estimator']
        population_size = int(selector_params.get("population_size", 20))
        max_iter = int(selector_params.get("max_iter", 50))
        cv = int(selector_params.get("cv", 5))
        penalty_weight = float(selector_params.get("penalty_weight", 0.1))
        rs = selector_params.get("random_state", random_state)
        estimator = selector_params.get("estimator", None)

        woa = WOAFeatureSelector(
            population_size=population_size,
            max_iter=max_iter,
            estimator=estimator,
            cv=cv,
            penalty_weight=penalty_weight,
            random_state=rs,
        )
        steps.append(("selector", woa))
        selector_name = f"WOA(pop={population_size}, iters={max_iter}, cv={cv})"

    elif sel in {'bso-cv', 'bsocv', 'bso'}:
        # BSO-CV espera DataFrame para poder indexar columnas (usa X.iloc internamente)
        population_size = int(selector_params.get("population_size", 20))
        max_iter        = int(selector_params.get("max_iter", 50))
        cv              = int(selector_params.get("cv", 5))
        random_state_b  = selector_params.get("random_state", random_state)
        penalty_weight  = float(selector_params.get("penalty_weight", 0.01))
        verbose         = bool(selector_params.get("verbose", False))

        bso = BSOFeatureSelector(
            population_size=population_size,
            max_iter=max_iter,
            cv=cv,
            random_state=random_state_b,
            penalty_weight=penalty_weight,
            verbose=verbose,
        )
        steps.append(("selector", bso))
        selector_name = f"BSO-CV(pop={population_size}, iters={max_iter}, cv={cv})"

    else:
        raise ValueError(f"Selector desconocido: {selector}")
    
    # 3.3) Escalado (usa exactamente la misma función que MLP/KNN)
    scaler = scale_features(scaler_type)
    steps.append(("scaler", scaler))

    # 3.4) SMOTE (solo en entrenamiento; imblearn lo aplica correctamente en CV)
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))

    # 3.5) Clasificador XGB (usando build_xgb de predictores/xgb.py)
    merged_xgb_params: Dict[str, Any] = dict(xgb_params) if xgb_params else {}
    merged_xgb_params.setdefault("random_state", random_state)
    clf = build_xgb(merged_xgb_params)
    steps.append(("clf", clf))

    pipe = ImbPipeline(steps=steps)

    # 4) Train/test split usando tu helper split_data (sin SMOTE aquí)
    X_train, X_test, y_train, y_test = split_data(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        use_smote=False,
    )

    # 5) (Opcional) GridSearchCV sobre el pipeline completo
    best_estimator = pipe
    best_params = None

    opt = (optimizer or "none").strip().lower() if optimizer is not None else "none"
    if opt in {"gridsearchcv", "run_grid_search"}:
        # Grid por defecto
        default_param_grid = {
            "clf__n_estimators": [100, 200, 500],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }

        # Si xgb_params trae listas/tuplas, se usan como grid específico
        if xgb_params:
            user_grid = {}
            for k, v in xgb_params.items():
                param_name = f"clf__{k}"
                if isinstance(v, (list, tuple, np.ndarray)):
                    user_grid[param_name] = list(v)
                else:
                    user_grid[param_name] = [v]
            default_param_grid.update(user_grid)

        gs = GridSearchCV(
            pipe,
            param_grid=default_param_grid,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)

        # Recoger los resultados de cada iteración (incluyendo hiperparámetros y cada fold)
        results = []
        for i, params in enumerate(gs.cv_results_["params"]):
            for fold_idx in range(5):  # Para 5 folds
                # Realizar predicción para esta combinación de parámetros y fold
                gs.best_estimator_.fit(X_train, y_train)  # Asegurarse de que el modelo está entrenado
                y_pred = gs.best_estimator_.predict(X_test)
                y_pred_prob = gs.best_estimator_.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva

                iteration_result = {
                    'y_true': y_test,
                    'y_pred': y_pred,  # Predicciones de esta iteración
                    'y_pred_prob': y_pred_prob,  # Probabilidades de esta iteración
                    'hyperparameters': params,  # Los hiperparámetros para esta iteración
                    'cv_folds': 5,  # Número de folds de CV
                }
                results.append(iteration_result)

        # Llamada a save_metrics_to_csv con los resultados
        save_metrics_to_csv(results, model_name="xgb")

        best_estimator = gs.best_estimator_
        best_params = gs.best_params_
    else:
        best_estimator.fit(X_train, y_train)


    # 6) Evaluación en test
    y_pred = best_estimator.predict(X_test)
    if hasattr(best_estimator, "predict_proba"):
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    # 7) Resultado minimal (mismo estilo que mlp_pipeline)
    elapsed = round(time.time() - t0, 4)
    result = {
        "model": "xgb",
        "selector": selector_name,
        "metrics": metrics,
        "selected_features": list(X_df.columns),
        "elapsed_seconds": elapsed,
        "extra_info": {
            "optimizer": (
                optimizer if isinstance(optimizer, str)
                else getattr(optimizer, "__name__", str(optimizer))
            ),
            "best_params": best_params,
        },
    }

    print_from_pipeline_result(result)
    return result