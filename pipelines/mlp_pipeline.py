# pipelines/mlp_pipeline.py
from __future__ import annotations
from typing import Tuple, Optional
import time
import numpy as np

from preprocesamiento.lectura_datos import load_data
from preprocesamiento.codificacion import encode_features
from preprocesamiento.escalado import scale_features
from preprocesamiento.division_dataset import split_data

# Selectores
from selectores.bsocv import BSOFeatureSelector
from selectores.mabc import m_abc_feature_selection
from selectores.woa import woa_feature_selection
from selectores.eliminacionpearson import eliminar_redundancias

# Predictor
from predictores.mlp import mlp_train

#Optimizadores
from optimizadores.gridSearchCV import run_grid_search

# Reporte
from utils.evaluacion import compute_classification_metrics, print_from_pipeline_result

def mlp_pipeline(
    file_path: str = "SAHeart.csv",
    selector: Optional[str] = "woa",
    *,
    encoding_method: str = "manual",
    scaler_type: str = "minmax",      # = mlpWOA.py
    redundancy: Optional[str] = "none",
    hidden_layer_sizes=(100,),
    activation: str = "relu",
    solver: str = "adam",
    max_iter: int = 500,              # = mlpWOA.py
    random_state: int = 42,
    early_stopping: bool = False,     # = mlpWOA.py
    tol: float = 1e-4,
    use_smote: bool = True,
    optimizer: Optional[str] = "none",
    **selector_params,
) -> dict:
    """
    Pipeline MLP con selección de características (BSO-CV / M-ABC / WOA) y escalado post-split
    (sin fuga de información), configurado para replicar el comportamiento del script monolítico.
    """
    t0 = time.time()

    # Compatibilidad: permitir selector_params={"selector_params": {...}}
    if "selector_params" in selector_params and isinstance(selector_params["selector_params"], dict):
        selector_params = dict(selector_params["selector_params"])

    # 1) Carga y codificación
    df = load_data(file_path)
    df = encode_features(df, encoding_method=encoding_method)
    df = df.drop(columns=["row.names"], errors="ignore")

    # 2) Redundancias (opcional)
    if redundancy is not None and str(redundancy).lower() not in {"none", "sin", "no"}:
        df = eliminar_redundancias(df, metodo=redundancy)

    # 3) X, y
    if "chd" not in df.columns:
        raise ValueError("La columna objetivo 'chd' no se encuentra en el dataset.")
    X_df = df.drop(columns=["chd"])
    y = df["chd"].values

    # 4) Selección de características (opcional) — incluye BSO-CV, M-ABC y WOA
    X_sel = X_df
    selector_name = None
    mask_for_report = None
    fitness_for_report = None

    if selector is not None and str(selector).lower() not in {"none", "sin", "no"}:
        sel = selector.strip().lower()

        if sel in ("bso", "bso-cv", "bsocv"):
            # Igual que antes: no requiere escalado global
            population_size = int(selector_params.get("population_size", 20))
            max_iter_s      = int(selector_params.get("max_iter", 50))
            cv              = int(selector_params.get("cv", 5))
            random_state_b  = selector_params.get("random_state", None)
            penalty_weight  = float(selector_params.get("penalty_weight", 0.01))
            verbose         = bool(selector_params.get("verbose", False))

            selector_est = BSOFeatureSelector(
                population_size=population_size,
                max_iter=max_iter_s,
                cv=cv,
                random_state=random_state_b,
                penalty_weight=penalty_weight,
                verbose=verbose,
            )
            selector_est.fit(X_df, df["chd"])
            X_sel = selector_est.transform(X_df)
            selector_name = f"BSO-CV (pop={population_size}, iters={max_iter_s}, cv={cv})"
            mask_for_report = selector_est.get_support().astype(int).tolist()
            fitness_for_report = float(selector_est.fitness_)

        elif sel in ("m-abc", "mabc", "m_abc"):
            # Escalado rápido sobre TODO X SOLO para el selector (heurística interna)
            X_scaled_all = scale_features(X_df.values, scaler_type=scaler_type)
            X_train_s, X_test_s, y_train_s, y_test_s = split_data(X_scaled_all, y)

            pop_size     = int(selector_params.get("pop_size", 20))
            max_cycles   = int(selector_params.get("max_cycles", selector_params.get("max_iter", 30)))
            limit        = int(selector_params.get("limit", 5))
            patience     = int(selector_params.get("patience", 10))
            random_state_s = selector_params.get("random_state", 42)
            cv_folds     = int(selector_params.get("cv_folds", 5))
            knn_k        = int(selector_params.get("knn_k", 5))
            verbose      = bool(selector_params.get("verbose", False))

            best_mask, best_fitness = m_abc_feature_selection(
                X_train_s, X_test_s, y_train_s, y_test_s,
                use_custom_evaluator=False,     # KNN CV interno (rápido)
                pop_size=pop_size,
                max_cycles=max_cycles,
                limit=limit,
                patience=patience,
                random_state=random_state_s,
                cv_folds=cv_folds,
                knn_k=knn_k,
                verbose=verbose,
            )

            idx = np.where(best_mask == 1)[0]
            X_sel = X_df.iloc[:, idx] if idx.size > 0 else X_df
            selector_name = f"M-ABC (pop={pop_size}, cycles={max_cycles}, cv={cv_folds})"
            mask_for_report = list(map(int, best_mask))
            fitness_for_report = float(best_fitness)

        elif sel in ("woa", "whale", "whale-optimization"):
            # Igual que el monolítico: escala TODO X para el fitness
            X_scaled_all = scale_features(X_df.values, scaler_type=scaler_type)

            # Defaults emparejados al script original
            woa_pop   = int(selector_params.get("population_size", 30))
            woa_iters = int(selector_params.get("max_iter", 50))
            woa_cv    = int(selector_params.get("cv", 5))
            woa_pen   = float(selector_params.get("penalty_weight", 0.01))
            woa_seed  = selector_params.get("random_state", random_state)

            best_mask, best_fitness = woa_feature_selection(
                X_scaled_all,
                y,
                population_size=woa_pop,
                max_iter=woa_iters,
                estimator=selector_params.get("estimator", None),
                cv=woa_cv,
                penalty_weight=woa_pen,
                random_state=woa_seed,
            )

            idx = np.where(best_mask == 1)[0]
            X_sel = X_df.iloc[:, idx] if idx.size > 0 else X_df
            selector_name = f"WOA (pop={woa_pop}, iters={woa_iters}, cv={woa_cv}, pen={woa_pen})"
            mask_for_report = list(map(int, best_mask))
            fitness_for_report = float(best_fitness)

        else:
            raise ValueError("Selector desconocido: usa 'bso-cv', 'm-abc', 'woa' o None.")

    # 5) Split y escalado PARA EL MLP (igual que monolítico; sin fuga)
    #    Nota: fijamos test_size=0.2 para replicar mlpWOA.py
    X_train, X_test, y_train, y_test = split_data(
        X_sel.values, y, test_size=0.2, random_state=random_state, use_smote=use_smote
    )
    
    scaler = scale_features(scaler_type)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 6) Entrenamiento + (opcional) optimización del MLP
    best_params = None

    # Acepta "gridsearchcv" o "run_grid_search" como etiqueta
    usa_gs = optimizer is not None and str(optimizer).lower() in {"gridsearchcv", "run_grid_search"}

    if usa_gs:
        # Ejecuta GridSearchCV con el estimador sklearn MLPClassifier
        gs = run_grid_search("mlp", X_train, y_train, cv=5)
        if gs is not None:
            best_mlp = gs["best_estimator"]
            best_params = gs["best_params"]

            # Predicción y métricas con el mejor modelo
            y_pred = best_mlp.predict(X_test)
            y_proba = best_mlp.predict_proba(X_test)[:, 1] if hasattr(best_mlp, "predict_proba") else None
            metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        else:
            # Fallback a tu entrenamiento "normal" si el grid no se pudo realizar
            y_pred, y_proba, y_test_real = mlp_train(
                X_train, y_train, X_test, y_test,
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=early_stopping,
                tol=tol,
            )
            metrics = compute_classification_metrics(y_test_real, y_pred, y_proba)
    else:
        # Entrenamiento "normal" (sin optimización)
        y_pred, y_proba, y_test_real = mlp_train(
            X_train, y_train, X_test, y_test,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            tol=tol,
        )
        metrics = compute_classification_metrics(y_test_real, y_pred, y_proba)


    # 7) Reporte
    opt_name = optimizer.__name__ if callable(optimizer) else optimizer
    elapsed = round(time.time() - t0, 4)
    result = {
        "model": "mlp",
        "selector": selector_name,
        "metrics": metrics,
        "selected_features": list(X_sel.columns),
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