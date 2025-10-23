from __future__ import annotations
from typing import Tuple, Optional
import time
import numpy as np

from preprocesamiento.lectura_datos import load_data
from preprocesamiento.codificacion import encode_features
from preprocesamiento.escalado import scale_features
from preprocesamiento.division_dataset import split_data

# Selectores ya modularizados
from selectores.bsocv import BSOFeatureSelector
from selectores.mabc import m_abc_feature_selection
from selectores.woa import woa_feature_selection
from selectores.eliminacionpearson import eliminar_redundancias

# Predictores (ahora sí existen ambas funciones)
from predictores.transformer import transformer_train

#Optimizadores
from optimizadores.gridSearchCV import get_param_grid

# Utilidades de evaluación
from utils.evaluacion import print_metrics_from_values
from utils.evaluacion import print_from_pipeline_result


def transformer_pipeline(
    file_path: str,
    selector: Optional[str] = "woa",
    *,
    encoding_method: str = "manual",
    scaler_type: str = "minmax",
    redundancy: Optional[str] = "none",
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    use_smote: bool = True,
    optimizer: Optional[str] = "none",
    **selector_params,
) -> dict:
    """
    Pipeline para entrenar y evaluar un clasificador basado en Transformer sobre SAHeart,
    usando la estructura modularizada.
    selector: "bso-cv", "m-abc", "woa" o None (sin selección de características)
    """
    t0 = time.time()

    # --- Robustez: desanidar si vino selector_params=dict(...) desde main antiguos
    if "selector_params" in selector_params and isinstance(selector_params["selector_params"], dict):
        selector_params = dict(selector_params["selector_params"])

    # 1) Cargar dataset
    df = load_data(file_path)

    # 2) Codificar categóricas
    df = encode_features(df, encoding_method=encoding_method)

    # 3) Limpieza específica (si existiera columna id tipo 'row.names')
    drop_cols = [c for c in ["row.names"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 2.2) Eliminación de redundancias (opcional)
    if redundancy is not None and str(redundancy).lower() not in {"none", "sin", "no"}:
        df = eliminar_redundancias(df, metodo=redundancy)

    # 4) Definir X, y
    if "chd" not in df.columns:
        raise ValueError("La columna objetivo 'chd' no se encuentra en el dataset.")
    X_df = df.drop(columns=["chd"])  # conservamos nombres para reportes
    y = df["chd"].values

    # 5) Selección de características (opcional)
    X_sel = X_df
    selector_name = None
    mask_for_report = None
    fitness_for_report = None

    if selector is not None:
        sel = selector.strip().lower()

        if sel in ("bso", "bso-cv", "bsocv"):
            # Normaliza/filtra kwargs para BSO-CV
            population_size = int(selector_params.get("population_size", 20))
            max_iter = int(selector_params.get("max_iter", 50))
            cv = int(selector_params.get("cv", 5))
            random_state = selector_params.get("random_state", None)
            penalty_weight = float(selector_params.get("penalty_weight", 0.01))
            verbose = bool(selector_params.get("verbose", False))

            selector_est = BSOFeatureSelector(
                population_size=population_size,
                max_iter=max_iter,
                cv=cv,
                random_state=random_state,
                penalty_weight=penalty_weight,
                verbose=verbose,
            )
            selector_est.fit(X_df, df["chd"])
            X_sel = selector_est.transform(X_df)

            selector_name = f"BSO-CV (pop={population_size}, iters={max_iter}, cv={cv})"
            mask_for_report = selector_est.get_support().astype(int).tolist()
            fitness_for_report = float(selector_est.fitness_)

        elif sel in ("m-abc", "mabc", "m_abc"):
            # Pre-escala para que el fitness KNN sea estable
            X_scaled_all = scale_features(X_df.values, scaler_type=scaler_type)
            X_train, X_test, y_train, y_test = split_data(X_scaled_all, y)

            # Defaults tipo "script monolítico": 20x30 + CV=5
            pop_size     = selector_params.get("pop_size", 20)
            max_cycles   = selector_params.get("max_cycles", selector_params.get("max_iter", 30))
            limit        = selector_params.get("limit", 5)
            patience     = selector_params.get("patience", 10)
            random_state = selector_params.get("random_state", 42)
            cv_folds     = selector_params.get("cv_folds", 5)
            knn_k        = selector_params.get("knn_k", 5)

            # ¡OJO! No pasamos evaluator_fn: usamos el KNN CV interno (rápido)
            best_mask, best_fitness = m_abc_feature_selection(
                X_train, X_test, y_train, y_test,
                use_custom_evaluator=False,
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
            fitness_for_report = best_fitness
            
        elif sel in ("woa",):
            X_scaled_all = scale_features(X_df.values, scaler_type=scaler_type)

            population_size = selector_params.get("population_size", 20)
            max_iter = selector_params.get("max_iter", 50)
            cv = selector_params.get("cv", 5)
            penalty_weight = selector_params.get("penalty_weight", 0.01)
            estimator = selector_params.get("estimator", None)
            random_state = selector_params.get("random_state", None)

            best_mask, best_fitness = woa_feature_selection(
                X_scaled_all,
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
            selector_name = f"WOA (pop={population_size}, iters={max_iter}, cv={cv})"
            mask_for_report = list(map(int, best_mask))
            fitness_for_report = best_fitness

        elif sel in ("none", "sin", "no"):
            selector_name = None
        else:
            raise ValueError("Selector desconocido: usa 'bso-cv', 'm-abc', 'woa' o None.")

    # 6) Escalado y split con las columnas finales
    X_scaled = scale_features(X_sel.values, scaler_type=scaler_type)
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y, use_smote=use_smote  # <-- Pasa el parámetro aquí
    )

    # 7) Entrenamiento + evaluación (con optimización opcional)
    best_params = None
    metrics = None

    def _optimizer_label(opt):
        if opt is None:
            return None
        if isinstance(opt, str):
            return opt
        return getattr(opt, "__name__", opt.__class__.__name__)

    usa_grid = optimizer is not None and str(optimizer).lower() in {"gridsearchcv", "run_grid_search", "grid", "search"}

    if usa_grid:
        # Obtenemos el grid “sugerido” del archivo de optimización
        full_grid = get_param_grid("transformer")
        # Para no explotar el tiempo de cómputo, acotamos a un sub-grid pequeño
        epochs_grid     = full_grid.get("epochs", [epochs])[:2]
        lr_grid         = full_grid.get("lr", [lr])[:2]
        batch_grid      = full_grid.get("batch_size", [batch_size])[:2]

        # Búsqueda manual (f1 como métrica si viene en el dict de retorno)
        best_f1 = -1.0
        for e in epochs_grid:
            for lr_ in lr_grid:
                for bs in batch_grid:
                    cand_metrics = transformer_train(
                        X_train, y_train, X_test, y_test,
                        epochs=e, lr=lr_, batch_size=bs,
                        d_model=d_model, nhead=nhead, num_layers=num_layers,
                    )
                    # intenta leer alguna clave estándar
                    f1 = None
                    for k in ("f1", "f1_score", "macro_f1", "f1_macro"):
                        if k in cand_metrics and isinstance(cand_metrics[k], (int, float)):
                            f1 = cand_metrics[k]
                            break
                    if f1 is None:
                        # si tu función devuelve 'metrics' anidado o similar, adapta aquí
                        f1 = -1.0
                    if f1 > best_f1:
                        best_f1 = f1
                        metrics = cand_metrics
                        best_params = {"epochs": e, "lr": lr_, "batch_size": bs}

    # Si no hay optimizer o no mejoró, entrena con los hiperparámetros base
    if metrics is None:
        metrics = transformer_train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, lr=lr, batch_size=batch_size,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
        )


    # 8) Reporte
    opt_name = _optimizer_label(optimizer)
    elapsed = round(time.time() - t0, 4)
    result = {
        "model": "transformer",
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