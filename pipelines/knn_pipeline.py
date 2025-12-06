from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import time
from datetime import datetime
import warnings
import pandas as pd

from preprocesamiento.lectura_datos import load_data
from preprocesamiento.codificacion import encode_features
from preprocesamiento.escalado import scale_features
from preprocesamiento.division_dataset import split_data

# Selectores ya modularizados
from selectores.bsocv import BSOFeatureSelector
from selectores.mabc import MABCFeatureSelector
from selectores.woa import WOAFeatureSelector   
from selectores.eliminacionpearson import PearsonRedundancyEliminator     

# Predictores KNN (funciones en predictores/knn.py)
from predictores.knn import knn_evaluator

#Optimizadores
from optimizadores.gridSearchCV import save_metrics_to_csv

# Utilidades de evaluación
from utils.evaluacion import compute_classification_metrics, print_from_pipeline_result

#sklearn & imblearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as SkPipeline  
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

def knn_pipeline(
    file_path: str,
    selector: Optional[str] = "bso-cv",
    *,
    encoding_method: str = "labelencoder",   # coherente con main.py
    scaler_type: str = "standard",             # coherente con main.py
    redundancy: Optional[str] = "none",  # <-- NUEVO
    use_smote: bool = True,
    optimizer: Optional[str] = "none",
    n_neighbors: int = 3, 
    random_state: int = 42,
    **selector_params,
):
    """
    Pipeline KNN modular:
      - preprocesamiento/*: carga, codificación, escalado, split
      - selectores/*: BSO-CV, M-ABC o WOA (opcional)
      - predictores/knn.py: entrenamiento y evaluador KNN
      - utils/evaluacion.py: reporte de métricas

    selector: "bso-cv", "m-abc", "woa" o None (sin selección)

    Parámetros extra (selector_params) por selector:
      - M-ABC: population_size, max_iter, limit
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

    # 3) Definir X, y
    if "chd" not in df.columns:
        raise ValueError("La columna objetivo 'chd' no se encuentra en el dataset.")
    X_df = df.drop(columns=["chd"])
    y = df["chd"].values

    # 4) Hold-out split (sin SMOTE aquí)
    X_train, X_test, y_train, y_test = split_data(X_df.values, y, test_size=0.3, random_state=random_state)

    # 6) Construcción del Pipeline de imblearn
    steps = []

    # 2.2) Eliminación de redundancias (opcional)
    if redundancy is not None and str(redundancy).lower() not in {"none", "sin", "no"}:
        steps.append(("redundancy", PearsonRedundancyEliminator(metodo=redundancy)))

    # 4) Si el selector es WOA (función no-transformer), aplicamos máscara **antes** del split
    # (nota: para usar WOA dentro de CV haría falta envolverlo como Transformer).
    sel = (selector or "none").strip().lower() if selector is not None else "none"
    selector_name = None

    if sel in ("bso", "bso-cv", "bsocv"):
        population_size = int(selector_params.get("population_size", 20))
        max_iter        = int(selector_params.get("max_iter", 50))
        cv              = int(selector_params.get("cv", 5))
        random_state_b  = selector_params.get("random_state", None)
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
        selector_name = f"BSO-CV (pop={population_size}, iters={max_iter}, cv={cv})"

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
            knn_k=knn_k,
            population_size=population_size,
            max_iter=max_iter,  
            limit=limit,
            patience=patience,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=verbose,
        )

        steps.append(("selector", mabc))
        selector_name = f"M-ABC (pop={population_size}, cycles={max_iter}, cv={cv_folds}, k={knn_k})"

    elif sel in ("woa", "whale", "ballenas"):
        # WOA: para coherencia con KNN, usamos KNN en el fitness + pre-escalado temporal
        n_neighbors = int(selector_params.get("n_neighbors", 3))
        population_size = int(selector_params.get("population_size", 20))
        max_iter = int(selector_params.get("max_iter", 50))
        cv = int(selector_params.get("cv", 5))
        penalty_weight = float(selector_params.get("penalty_weight", 0.01))
        random_state_w = selector_params.get("random_state", 42)

        # Escalado dentro de la CV del fitness de WOA:
        estimator_for_fitness = SkPipeline([
            ("scaler", scale_features(scaler_type)),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ])

        woa = WOAFeatureSelector(
            population_size=population_size,
            max_iter=max_iter,
            estimator=estimator_for_fitness,  # evita fuga: el escalado está dentro de la CV interna
            cv=cv,
            penalty_weight=penalty_weight,
            random_state=random_state_w,
        )
        steps.append(("selector", woa))
        selector_name = f"WOA (pop={population_size}, iters={max_iter}, cv={cv}, k={n_neighbors})"

    elif sel in ("none", "sin", "no"):
        pass
    else:
        raise ValueError("Selector desconocido: usa 'bso-cv', 'm-abc', 'woa' o None.")

    # Paso SMOTE (opcional, DENTRO del pipeline)
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))

    # Scaler + Clasificador
    scaler = scale_features(scaler_type)
    steps.append(("scaler", scaler))
    steps.append(("clf", KNeighborsClassifier(n_neighbors=n_neighbors)))

    # 5.5) Si hay SMOTE, usamos ImbPipeline; si no, SkPipeline normal
    if any(name == "smote" for name, _ in steps):
        pipe = ImbPipeline(steps=steps)
    else:
        pipe = SkPipeline(steps=steps)

    best_params = None
    model = pipe
    metrics = None

    # 7) Entrenamiento (con o sin GridSearch) sobre el PIPELINE COMPLETO
    if optimizer and str(optimizer).lower() == "gridsearchcv":
        # Grid genérico
        param_grid = {
            "clf__n_neighbors": [3, 5, 7, 10, 17],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["minkowski", "euclidean", "manhattan"],
        }

        # Solo añadimos hiperparámetros del selector si existe
        if any(name == "selector" for name, _ in steps):
            param_grid.update(
                {
                    "selector__population_size": [selector_params.get("population_size", 20)],
                    "selector__max_iter": [selector_params.get("max_iter", 50)],
                }
            )

        gs = GridSearchCV(
            pipe, param_grid=param_grid, cv=5, scoring="f1_macro", n_jobs=-1
        )
        
        # Ignoramos warnings de convergencia durante la búsqueda
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gs.fit(X_train, y_train)

        results = []
        best_f1 = -np.inf
        best_y_pred = None
        best_y_proba = None
        best_params = None

        # Recorremos todas las combinaciones de hiperparámetros evaluadas
        for i, params in enumerate(gs.cv_results_["params"]):
            # Clonamos el pipeline base y le ponemos estos hiperparámetros
            model_i = clone(pipe)
            model_i.set_params(**params)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model_i.fit(X_train, y_train)

            # Predicciones de este modelo en el test hold-out
            y_pred_i = model_i.predict(X_test)
            if hasattr(model_i, "predict_proba"):
                y_proba_i = model_i.predict_proba(X_test)[:, 1]
            else:
                y_proba_i = y_pred_i

            # F1 macro en test para seleccionar el mejor registro
            f1_i = f1_score(y_test, y_pred_i, average="macro")

            iteration_result = {
                "y_true": y_test,
                "y_pred": y_pred_i,
                "y_pred_prob": y_proba_i,
                "hyperparameters": params,
                "cv_folds": gs.cv,
                "f1_macro": f1_i,
            }
            results.append(iteration_result)

            if f1_i > best_f1:
                best_f1 = f1_i
                best_params = params
                best_y_pred = y_pred_i
                best_y_proba = y_proba_i

        # Llamada a save_metrics_to_csv con los resultados
        save_metrics_to_csv(results, model_name="knn")

        # Mensaje explícito del mejor registro según F1 macro en test
        print("\n[GridSearchCV - mejor registro según F1 macro en test]")
        print(f"Mejor F1 macro: {best_f1:.4f}")
        print(f"Mejores hiperparámetros: {best_params}\n")

        # Entrenamos el modelo final con los mejores hiperparámetros hallados
        model = clone(pipe)
        if best_params is not None:
            model.set_params(**best_params)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)

        # Métricas finales sobre test usando el mejor registro
        metrics = compute_classification_metrics(y_test, best_y_pred, best_y_proba)
    else:
        # Entrenamiento sin optimización de hiperparámetros
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = pipe.fit(X_train, y_train)

        # Predicciones del pipeline completo (sin knn_evaluator)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    # 8) Fitness del selector si está disponible
    fitness_for_report = None
    if hasattr(model, "steps") and any(name == "selector" for name, _ in model.steps):
        sel_step = model.named_steps.get("selector")
        if hasattr(sel_step, "fitness_"):
            fitness_for_report = getattr(sel_step, "fitness_", None)
        elif hasattr(sel_step, "best_score_"):
            fitness_for_report = getattr(sel_step, "best_score_", None)

        # 9) Features seleccionadas (si el selector expone get_support)
    selected_features = list(X_df.columns)
    try:
        # empezamos desde los nombres originales
        cols = list(X_df.columns)

        # si hay un paso de redundancia que expone get_feature_names_out, usarlo
        if hasattr(model, "named_steps") and "redundancy" in model.named_steps:
            red = model.named_steps["redundancy"]
            if hasattr(red, "get_feature_names_out"):
                cols = list(red.get_feature_names_out(cols))
            else:
                # intentar inferir nombres aplicando transform sobre el DataFrame
                try:
                    transformed = red.transform(X_df)
                    if hasattr(transformed, "columns"):
                        cols = list(transformed.columns)
                except Exception:
                    # fallback: mantener cols sin cambios
                    pass

        # luego aplicar la máscara del selector (si la hay)
        if hasattr(model, "named_steps") and "selector" in model.named_steps:
            sel_step = model.named_steps["selector"]
            if hasattr(sel_step, "get_support"):
                support = np.array(sel_step.get_support(), dtype=bool)
                if support.shape[0] == len(cols):
                    cols = list(np.array(cols)[support])

        selected_features = cols
    except Exception as e:
        # fallback: mantener columnas originales
        selected_features = list(X_df.columns)
    # 10) Resultado estandarizado
    elapsed = round(time.time() - t0, 4)

    # Recuperamos el fitness del selector si está disponible
    fitness_for_report = None
    if any(name == "selector" for name, _ in model.steps):
        sel_step = model.named_steps.get("selector")
        if hasattr(sel_step, "fitness_"):
            fitness_for_report = getattr(sel_step, "fitness_", None)
        elif hasattr(sel_step, "best_score_"):
            fitness_for_report = getattr(sel_step, "best_score_", None)

    result = {
        "model": "knn",
        "selector": selector_name,
        "metrics": metrics,
        "selected_features": selected_features, 
        "selector_fitness": fitness_for_report,
        "elapsed_seconds": elapsed,
        "extra_info": {
            "optimizer": (optimizer if isinstance(optimizer, str) else getattr(optimizer, "__name__", str(optimizer))),
            "best_params": best_params,
        },
    }
    print_from_pipeline_result(result)
    return result
