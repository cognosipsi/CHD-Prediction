# pipelines/knn_pipeline.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import time

from preprocesamiento.lectura_datos import load_data
from preprocesamiento.codificacion import encode_features
from preprocesamiento.escalado import scale_features
from preprocesamiento.division_dataset import split_data

# Selectores ya modularizados
from selectores.bsocv import bso_cv
from selectores.mabc import m_abc_feature_selection
from selectores.woa import woa_feature_selection
from selectores.eliminacionpearson import eliminar_redundancias

# Predictores KNN (funciones en predictores/knn.py)
from predictores.knn import knn_train, knn_evaluator

# Utilidades de evaluación
from utils.evaluacion import print_metrics_from_values
from utils.evaluacion import compute_classification_metrics, print_from_pipeline_result
# Estimador para fitness de WOA (usaremos KNN para coherencia con el pipeline)
from sklearn.neighbors import KNeighborsClassifier

def knn_pipeline(
    file_path: str,
    selector: Optional[str] = "bso-cv",
    *,
    encoding_method: str = "labelencoder",   # coherente con main.py
    scaler_type: str = "standard",             # coherente con main.py
    redundancy: Optional[str] = "none",  # <-- NUEVO
    use_smote: bool = True,
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

    # Limpiezas opcionales (compatibles con otros pipelines)
    drop_cols = [c for c in ["row.names", "obesity"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 2.2) Eliminación de redundancias (opcional)
    if redundancy is not None and str(redundancy).lower() not in {"none", "sin", "no"}:
        df = eliminar_redundancias(df, metodo=redundancy)

    # 3) Definir X, y
    if "chd" not in df.columns:
        raise ValueError("La columna objetivo 'chd' no se encuentra en el dataset.")
    X_df = df.drop(columns=["chd"])
    y = df["chd"].values

    # 4) Selección de características (opcional)
    X_sel = X_df
    selector_name = None
    mask_for_report = None
    fitness_for_report = None

    if selector is not None:
        sel = selector.strip().lower()

        if sel in ("bso", "bso-cv", "bsocv"):
            population_size = int(selector_params.get("population_size", 20))
            max_iter        = int(selector_params.get("max_iter", 50))
            cv              = int(selector_params.get("cv", 5))
            random_state_b  = selector_params.get("random_state", None)
            penalty_weight  = float(selector_params.get("penalty_weight", 0.01))
            verbose         = bool(selector_params.get("verbose", False))

            best_mask, best_fitness = bso_cv(
                X_df, df["chd"],
                population_size=population_size,
                max_iter=max_iter,
                cv=cv,
                random_state=random_state_b,
                penalty_weight=penalty_weight,
                verbose=verbose,
            )
            idx = [i for i, b in enumerate(best_mask) if b == 1]
            X_sel = X_df.iloc[:, idx] if idx else X_df
            selector_name = f"BSO-CV (pop={population_size}, iters={max_iter}, cv={cv})"
            mask_for_report = list(map(int, best_mask))
            fitness_for_report = float(best_fitness)

        elif sel in ("m-abc", "mabc", "m_abc"):
            X_all_scaled = scale_features(X_df.values, scaler_type=scaler_type)
            X_train, X_test, y_train, y_test = split_data(X_all_scaled, y)

            pop_size     = selector_params.get("pop_size", 20)
            max_cycles   = selector_params.get("max_cycles", selector_params.get("max_iter", 30))
            limit        = selector_params.get("limit", 5)
            patience     = selector_params.get("patience", 10)
            random_state = selector_params.get("random_state", 42)
            cv_folds     = selector_params.get("cv_folds", 5)
            knn_k        = selector_params.get("knn_k", 5)

            best_mask, best_fitness = m_abc_feature_selection(
                X_train, X_test, y_train, y_test,
                use_custom_evaluator=False,   # KNN CV interno (coherente con KNN final)
                pop_size=pop_size,
                max_cycles=max_cycles,
                limit=limit,
                patience=patience,
                random_state=random_state,
                cv_folds=cv_folds,
                knn_k=knn_k,
                verbose=selector_params.get("verbose", False),
            )

            idx = np.where(best_mask == 1)[0]
            X_sel = X_df.iloc[:, idx] if idx.size > 0 else X_df
            selector_name = f"M-ABC (pop={pop_size}, cycles={max_cycles}, cv={cv_folds})"
            mask_for_report = list(map(int, best_mask))
            fitness_for_report = float(best_fitness)

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
                X_for_fs,               # ndarray escalado para fitness con KNN
                y,
                population_size=population_size,
                max_iter=max_iter,
                estimator=estimator,
                cv=cv,
                penalty_weight=penalty_weight,
                random_state=random_state,
            )
            idx = np.where(best_mask == 1)[0]
            X_sel = X_df.iloc[:, idx] if idx.size > 0 else X_df
            selector_name = f"WOA (pop={population_size}, iters={max_iter}, k={n_neighbors})"
            mask_for_report = list(map(int, best_mask.tolist()))
            fitness_for_report = float(best_fitness)

        elif sel in ("none", "sin", "no"):
            selector_name = None
        else:
            raise ValueError("Selector desconocido: usa 'bso-cv', 'm-abc', 'woa' o None.")

    # 5) Escalado final sobre columnas seleccionadas + split (lo que realmente usará el KNN final)
    X_scaled = scale_features(X_sel.values, scaler_type=scaler_type)
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y, use_smote=use_smote  # <-- Pasa el parámetro aquí
    )
    
    # 6) Entrenamiento + evaluación del KNN final
    # Obtén las predicciones y probabilidades (ajusta knn_train si es necesario)
    y_pred, y_prob = knn_evaluator(X_train, X_test, y_train, y_test)  # Debe retornar y_pred, y_prob
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)

    # 7) Reporte centralizado
    elapsed = round(time.time() - t0, 4)
    result = {
        "model": "knn",
        "selector": selector_name,
        "metrics": metrics,
        "selected_features": list(X_sel.columns),
        "mask": mask_for_report,
        "selector_fitness": fitness_for_report,
        "elapsed_seconds": elapsed,
        "extra_info": {"tiempo_s": elapsed},
    }
    print_from_pipeline_result(result)
    return result