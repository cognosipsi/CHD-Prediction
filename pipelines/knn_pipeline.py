from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import time

from preprocesamiento.lectura_datos import load_data
from preprocesamiento.codificacion import encode_features
from preprocesamiento.escalado import scale_features
from preprocesamiento.division_dataset import split_data
from preprocesamiento.smote import apply_smote  # Importamos SMOTE

# Selectores ya modularizados
from selectores.bsocv import BSOFeatureSelector
from selectores.mabc import MABCFeatureSelector
from selectores.woa import woa_feature_selection
from selectores.eliminacionpearson import eliminar_redundancias         # NUEVO: helper para instanciar scaler

# Predictores KNN (funciones en predictores/knn.py)
from predictores.knn import knn_evaluator

# Utilidades de evaluación
from utils.evaluacion import compute_classification_metrics, print_from_pipeline_result

#sklearn
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#Optimizadores
from optimizadores.gridSearchCV import run_grid_search

def knn_pipeline(
    file_path: str,
    selector: Optional[str] = "bso-cv",
    *,
    encoding_method: str = "labelencoder",   # coherente con main.py
    scaler_type: str = "standard",             # coherente con main.py
    redundancy: Optional[str] = "none",  # <-- NUEVO
    use_smote: bool = True,
    optimizer: Optional[str] = "none",
    **selector_params,
) -> Tuple[float, float, float, float, float]:
    """
    Pipeline KNN modular:
      - preprocesamiento/*: carga, codificación, escalado, split
      - selectores/*: BSO-CV, M-ABC o WOA (opcional)
      - predictores/knn.py: entrenamiento y evaluador KNN
      - utils/evaluacion.py: reporte de métricas

    selector: "bso-cv", "m-abc", "woa" o None (sin selección)

    Parámetros extra (selector_params) por selector:
      - M-ABC: pop_size, max_iter, limit
      - WOA: population_size, max_iter, cv, penalty_weight, random_state, n_neighbors
    """
    t0 = time.time()

    # Compatibilidad hacia atrás con llamadas antiguas pipeline(..., selector_params={...})
    if "selector_params" in selector_params and isinstance(selector_params["selector_params"], dict):
        selector_params = dict(selector_params["selector_params"])

    # 1) Cargar
    df = load_data(file_path)

    # 2) Codificar (SAHeart: 'famhist' categórica). Usa lo que venga desde main.py
    df = encode_features(df, encoding_method=encoding_method)

    # 2.2) Eliminación de redundancias (opcional)
    if redundancy is not None and str(redundancy).lower() not in {"none", "sin", "no"}:
        df = eliminar_redundancias(df, metodo=redundancy)

    # 3) Definir X, y
    if "chd" not in df.columns:
        raise ValueError("La columna objetivo 'chd' no se encuentra en el dataset.")
    X_df = df.drop(columns=["chd"])
    y = df["chd"].values

    # 3) Split
    X_train, X_test, y_train, y_test = split_data(X_df, y, use_smote=False)

    # 4) SMOTE solo en el train si se solicita
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    # 4) Selección de características (opcional)
    X_sel = X_df
    selector_name = None
    mask_for_report = None
    fitness_for_report = None

    sel = (selector or "none").strip().lower() if selector is not None else "none"

    if sel in ("bso", "bso-cv", "bsocv"):
        population_size = int(selector_params.get("population_size", 20))
        max_iter        = int(selector_params.get("max_iter", 50))
        cv              = int(selector_params.get("cv", 5))
        random_state_b  = selector_params.get("random_state", None)
        penalty_weight  = float(selector_params.get("penalty_weight", 0.01))
        verbose         = bool(selector_params.get("verbose", False))

        selector_est = BSOFeatureSelector(
            population_size=population_size,
            max_iter=max_iter,
            cv=cv,
            random_state=random_state_b,
            penalty_weight=penalty_weight,
            verbose=verbose,
        )
        selector_est.fit(X_df, df["chd"])
        X_sel = selector_est.transform(X_df)
        selector_name = f"BSO-CV (pop={population_size}, iters={max_iter}, cv={cv})"
        mask_for_report = selector_est.get_support().astype(int).tolist()
        fitness_for_report = float(selector_est.fitness_)

    elif sel in ("m-abc", "mabc", "m_abc"):

        population_size = int(selector_params.get("population_size", 20))
        max_iter        = int(selector_params.get("max_iter", 50))
        limit           = int(selector_params.get("limit", 5))
        patience        = int(selector_params.get("patience", 10))
        random_state = selector_params.get("random_state", 42)
        cv_folds     = selector_params.get("cv_folds", 5)
        knn_k           = int(selector_params.get("knn_k", 5))
        verbose = int(bool(selector_params.get("verbose", False)))

        mabc = MABCFeatureSelector(
            X_train, X_test, y_train, y_test,
            use_custom_evaluator=False,
            knn_k=knn_k,
            population_size=population_size,
            max_cycles=max_iter,     # <- importante
            limit=limit,
            patience=patience,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=verbose,
        )

        scaler = scale_features(scaler_type)  # <- tu helper
        clf = KNeighborsClassifier(n_neighbors=knn_k)

        pipe = Pipeline([
            ("mabc", mabc),
            ("scaler", scaler),
            ("clf", clf),
        ])
        # GridSearch opcional directamente sobre el pipeline
        if optimizer and str(optimizer).lower() == "gridsearchcv":
            param_grid = {
                "mabc__population_size": [population_size],
                "mabc__max_cycles": [max_iter],
                "clf__n_neighbors": [3, 5, 7],
            }
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            best_params = gs.best_params_
        else:
            model = pipe.fit(X_train, y_train)
            best_params = None

        # Reporte selección
        msel = model.named_steps["mabc"]
        mask_for_report = msel.get_support().astype(int).tolist()
        fitness_for_report = getattr(msel, "best_fitness_", None)
        selector_name = f"M-ABC (pop={population_size}, cycles={max_iter}, cv={cv_folds}, k={knn_k})"

        # Evaluación
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model.named_steps["clf"], "predict_proba"):
            try:
                y_prob = model.named_steps["clf"].predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)

        # Reporte selección
        msel = model.named_steps["mabc"]
        mask_for_report = msel.get_support().astype(int).tolist()
        fitness_for_report = getattr(msel, "best_fitness_", None)
        selector_name = f"M-ABC (pop={population_size}, cycles={max_iter}, cv={cv_folds}, k={knn_k})"

        # Evaluación
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model.named_steps["clf"], "predict_proba"):
            try:
                y_prob = model.named_steps["clf"].predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)

    elif sel in ("woa", "whale", "ballenas"):
        # WOA: para coherencia con KNN, usamos KNN en el fitness + pre-escalado temporal
        n_neighbors = selector_params.get("n_neighbors", 3)
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Pre-escalamos una copia para el fitness (no perdemos nombres de columnas para aplicar la máscara)
        X_for_fs = scale_features(X_df.values, scaler_type=scaler_type)

        population_size = selector_params.get("population_size", 20)
        max_iter = selector_params.get("max_iter", 50)
        cv = selector_params.get("cv", 5)
        penalty_weight = selector_params.get("penalty_weight", 0.01)
        random_state = selector_params.get("random_state", None)

        best_mask, best_fitness = woa_feature_selection(
            X_sel.values,               # ndarray escalado para fitness con KNN
            y,
            population_size=population_size,
            max_iter=max_iter,
            estimator=estimator,
            cv=cv,
            penalty_weight=penalty_weight,
            random_state=random_state,
        )
        idx = np.where(best_mask == 1)[0]
        X_sel = X_sel.iloc[:, idx] if idx.size > 0 else X_sel
        selector_name = f"WOA (pop={population_size}, iters={max_iter}, k={n_neighbors})"
        mask_for_report = list(map(int, best_mask.tolist()))
        fitness_for_report = float(best_fitness)

    elif sel in ("none", "sin", "no"):
        selector_name = None
    else:
        raise ValueError("Selector desconocido: usa 'bso-cv', 'm-abc', 'woa' o None.")

    # 5) Escalado final sobre columnas seleccionadas + split (lo que realmente usará el KNN final)
    # Re-split tras selección (sin SMOTE aquí)
    X_train, X_test, y_train, y_test = split_data(X_sel, y, use_smote=False)

    # SMOTE solo en el train si se solicita
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    # Escalado con tu helper get_scaler (fit en train, transform en test)
    scaler = scale_features(scaler_type)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # GridSearch en-pipeline (afina KNN y podría afinar M-ABC si amplías param_grid)
    if optimizer and str(optimizer).lower() == "gridsearchcv":
        param_grid = {
            "mabc__population_size": [population_size],
            "mabc__max_cycles": [max_iter],
            "clf__n_neighbors": [3, 5, 7],
        }
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        model = pipe.fit(X_train, y_train)
        best_params = None

    # 7) Reporte centralizado
    opt_name = optimizer.__name__ if callable(optimizer) else optimizer
    elapsed = round(time.time() - t0, 4)
    result = {
        "model": "knn",
        "selector": selector_name,
        "metrics": metrics,
        "selected_features": (
            list(X_df.columns[np.array(mask_for_report, dtype=bool)]) if mask_for_report is not None else list(X_df.columns)
        ),
        "mask": mask_for_report,
        "selector_fitness": fitness_for_report,
        "elapsed_seconds": elapsed,
        "extra_info": {
            "tiempo_s": elapsed,
            "optimizer": opt_name,
            "best_params": best_params,
        },
    }
    print_from_pipeline_result(result)
    return result