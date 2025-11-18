from __future__ import annotations
from typing import Optional
import time
import numpy as np

from preprocesamiento.lectura_datos import load_data
from preprocesamiento.codificacion import encode_features
from preprocesamiento.escalado import scale_features
from preprocesamiento.division_dataset import split_data

# Selectores ya modularizados
from selectores.bsocv import BSOFeatureSelector
from selectores.mabc import MABCFeatureSelector
from selectores.woa import WOAFeatureSelector
from selectores.eliminacionpearson import PearsonRedundancyEliminator

# Predictores (ahora sí existen ambas funciones)
from predictores.transformer import SklearnTransformerClassifier

#Optimizadores
from optimizadores.gridSearchCV import get_param_grid

# Utilidades de evaluación
from utils.evaluacion import print_from_pipeline_result, compute_classification_metrics

# sklearn / imblearn
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


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
    Pipeline para entrenar y evaluar un clasificador basado en Transformer sobre SAHeart.

    - Soporta selectores de características: "bso-cv", "m-abc", "woa" o None.
    - Usa SklearnTransformerClassifier como estimador final.
    - SMOTE se aplica dentro del Pipeline de imblearn (sin filtrado de información).
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

    # 4) Eliminación de redundancias (opcional) con wrapper sklearn
    if redundancy is not None and str(redundancy).lower() not in {"none", "sin", "no"}:
        red = PearsonRedundancyEliminator(metodo=redundancy)
        df = red.fit_transform(df)

    # 4) Definir X, y
    if "chd" not in df.columns:
        raise ValueError("La columna objetivo 'chd' no se encuentra en el dataset.")
    X_df = df.drop(columns=["chd"])  # conservamos nombres para reportes
    y = df["chd"].values

    # 5) Selección de características (opcional)
    X_sel = X_df
    selector_name: Optional[str] = None
    mask_for_report = None
    fitness_for_report = None

    if selector is not None:
        sel = selector.strip().lower()
        X_arr = X_df.values  # los selectores trabajan sobre ndarray; usamos la máscara sobre X_df

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
            selector_est.fit(X_arr, y)
            mask = selector_est.get_support()
            idx = np.where(mask)[0]
            X_sel = X_df.iloc[:, idx] if idx.size > 0 else X_df

            selector_name = f"BSO-CV (pop={population_size}, iters={max_iter}, cv={cv})"
            mask_for_report = mask.astype(int).tolist()
            fitness_for_report = float(getattr(selector_est, "fitness_", np.nan))

        elif sel in ("m-abc", "mabc", "m_abc"):
            # Defaults tipo "script monolítico": 20x30 + CV=5
            population_size = int(selector_params.get("population_size", 20))
            max_iter = int(selector_params.get("max_iter", 30))
            limit = int(selector_params.get("limit", 5))
            patience = int(selector_params.get("patience", 10))
            random_state = selector_params.get("random_state", 42)
            cv_folds = int(selector_params.get("cv_folds", 5))
            knn_k = int(selector_params.get("knn_k", 5))
            verbose      = bool(selector_params.get("verbose", False))

            selector_est = MABCFeatureSelector(
                knn_k=knn_k,
                population_size=population_size,
                max_iter=max_iter,
                limit=limit,
                patience=patience,
                cv_folds=cv_folds,
                random_state=random_state,
                verbose=verbose,
            )

            selector_est.fit(X_arr, y)
            mask = selector_est.get_support()
            idx = np.where(mask)[0]
            X_sel = X_df.iloc[:, idx] if idx.size > 0 else X_df
            selector_name = f"M-ABC (pop={population_size}, cycles={max_iter}, cv={cv_folds})"
            mask_for_report = mask.astype(int).tolist()
            fitness_for_report = float(getattr(selector_est, "best_fitness_", np.nan))
            
        elif sel in ("woa",):
            X_scaled_all = scale_features(X_df.values, scaler_type=scaler_type)

            population_size = int(selector_params.get("population_size", 20))
            max_iter = int(selector_params.get("max_iter", 50))
            cv = int(selector_params.get("cv", 5))
            penalty_weight = float(selector_params.get("penalty_weight", 0.01))
            estimator = selector_params.get("estimator", None)
            random_state = selector_params.get("random_state", None)

            selector_est = WOAFeatureSelector(
                population_size=population_size,
                max_iter=max_iter,
                estimator=estimator,
                cv=cv,
                penalty_weight=penalty_weight,
                random_state=random_state,
            )

            selector_est.fit(X_arr, y)
            # WOAFeatureSelector expone best_mask_ y best_score_
            mask = selector_est.best_mask_.astype(bool)
            idx = np.where(mask)[0]
            X_sel = X_df.iloc[:, idx] if idx.size > 0 else X_df

            selector_name = f"WOA (pop={population_size}, iters={max_iter}, cv={cv})"
            mask_for_report = mask.astype(int).tolist()
            fitness_for_report = float(getattr(selector_est, "best_score_", np.nan))

        elif sel in ("none", "sin", "no"):
            selector_name = None
        else:
            raise ValueError("Selector desconocido: usa 'bso-cv', 'm-abc', 'woa' o None.")

    # 6) Escalado y split con las columnas finales
    X_scaled = scale_features(X_sel.values, scaler_type=scaler_type)
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y, use_smote=False    # <-- Pasa el parámetro aquí
    )

    # 8) Construcción del Pipeline de imblearn con el wrapper del Transformer
    steps = []

    if use_smote:
        smote_random_state = selector_params.get("random_state", 42)
        steps.append(("smote", SMOTE(random_state=smote_random_state)))

    clf = SklearnTransformerClassifier(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        random_state=selector_params.get("random_state", None),
    )

    steps.append(("clf", clf))
    pipe = ImbPipeline(steps=steps)

    # 7) Entrenamiento + evaluación (con optimización opcional)
    best_params = None

    # Ejecuta GridSearchCV con el estimador sklearn wrapper del Transformer
    if optimizer and str(optimizer).lower() == "gridsearchcv":
        full_grid = get_param_grid("transformer")
        # Convertimos el grid genérico a nombres de parámetros dentro del Pipeline (clf__...)
        param_grid = {
            "clf__epochs": full_grid.get("epochs", [epochs]),
            "clf__lr": full_grid.get("lr", [lr]),
            "clf__batch_size": full_grid.get("batch_size", [batch_size]),
            "clf__d_model": full_grid.get("d_model", [d_model]),
            "clf__nhead": full_grid.get("nhead", [nhead]),
            "clf__num_layers": full_grid.get("num_layers", [num_layers]),
            "clf__dropout": full_grid.get("dropout", [0.2]),
        }
        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=int(selector_params.get("cv", 5)),
            scoring="f1",
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        model = pipe.fit(X_train, y_train)

    # 8.c) Evaluación en el conjunto de test
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)


    # 9) Reporte
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
            "optimizer": str(optimizer) if optimizer is not None else None,
            "best_params": best_params,
        },
    }
    print_from_pipeline_result(result)
    return result