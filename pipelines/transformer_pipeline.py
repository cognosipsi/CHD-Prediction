from __future__ import annotations
from typing import Optional
import time
import warnings
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

# Predictores
from predictores.transformer import SklearnTransformerClassifier

# Optimizadores
from optimizadores.gridSearchCV import get_param_grid, save_metrics_to_csv

# Utilidades de evaluación
from utils.evaluacion import print_from_pipeline_result, compute_classification_metrics

# sklearn / imblearn
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score


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
    dropout: float = 0.2,
    use_smote: bool = True,
    optimizer: Optional[str] = "none",
    random_state: int = 42,
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
    if (
        "selector_params" in selector_params
        and isinstance(selector_params["selector_params"], dict)
    ):
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

    # 5) Definir X, y
    if "chd" not in df.columns:
        raise ValueError("La columna objetivo 'chd' no se encuentra en el dataset.")
    X_df = df.drop(columns=["chd"])  # conservamos nombres para reportes
    y = df["chd"].values

    # 6) Split train/test (ANTES de selección de características para evitar fugas)
    test_size = float(selector_params.get("test_size", 0.3))
    X_train_df, X_test_df, y_train, y_test = split_data(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        use_smote=False,  # SMOTE solo dentro del Pipeline
    )

    # 7) Selección de características (opcional)
    X_train_sel_df = X_train_df
    X_test_sel_df = X_test_df
    selector_name: Optional[str] = None
    mask_for_report = None
    fitness_for_report = None

    if selector is not None:
        sel = selector.strip().lower()
        X_train_arr = X_train_df.values  # los selectores trabajan sobre ndarray

        if sel in ("bso", "bso-cv", "bsocv"):
            # Normaliza/filtra kwargs para BSO-CV
            population_size = int(selector_params.get("population_size", 20))
            max_iter = int(selector_params.get("max_iter", 50))
            cv = int(selector_params.get("cv", 5))
            random_state_sel = int(selector_params.get("random_state", random_state))
            penalty_weight = float(selector_params.get("penalty_weight", 0.01))
            verbose = bool(selector_params.get("verbose", False))

            selector_est = BSOFeatureSelector(
                population_size=population_size,
                max_iter=max_iter,
                cv=cv,
                random_state=random_state_sel,
                penalty_weight=penalty_weight,
                verbose=verbose,
            )
            selector_est.fit(X_train_arr, y_train)
            mask = selector_est.get_support()
            idx = np.where(mask)[0]

            # Aplicamos la máscara a train y test usando las columnas de X_df
            if idx.size > 0:
                X_train_sel_df = X_train_df.iloc[:, idx]
                X_test_sel_df = X_test_df.iloc[:, idx]
            else:
                X_train_sel_df = X_train_df
                X_test_sel_df = X_test_df

            selector_name = (
                f"BSO-CV (pop={population_size}, iters={max_iter}, cv={cv})"
            )
            mask_for_report = mask.astype(int).tolist()
            fitness_for_report = float(getattr(selector_est, "fitness_", np.nan))

        elif sel in ("m-abc", "mabc", "m_abc"):
            # Defaults tipo "script monolítico": 20x30 + CV=5
            population_size = int(selector_params.get("population_size", 20))
            max_iter = int(selector_params.get("max_iter", 30))
            limit = int(selector_params.get("limit", 5))
            patience = int(selector_params.get("patience", 10))
            random_state_sel = int(selector_params.get("random_state", random_state))
            cv_folds = int(selector_params.get("cv_folds", 5))
            knn_k = int(selector_params.get("knn_k", 5))
            verbose = bool(selector_params.get("verbose", False))

            selector_est = MABCFeatureSelector(
                knn_k=knn_k,
                population_size=population_size,
                max_iter=max_iter,
                limit=limit,
                patience=patience,
                cv_folds=cv_folds,
                random_state=random_state_sel,
                verbose=verbose,
            )

            selector_est.fit(X_train_arr, y_train)
            mask = selector_est.get_support()
            idx = np.where(mask)[0]

            if idx.size > 0:
                X_train_sel_df = X_train_df.iloc[:, idx]
                X_test_sel_df = X_test_df.iloc[:, idx]
            else:
                X_train_sel_df = X_train_df
                X_test_sel_df = X_test_df

            selector_name = (
                f"M-ABC (pop={population_size}, cycles={max_iter}, cv={cv_folds})"
            )
            mask_for_report = mask.astype(int).tolist()
            fitness_for_report = float(
                getattr(selector_est, "best_fitness_", np.nan)
            )

        elif sel in ("woa",):
            scaler_for_woa = scale_features(scaler_type)
            X_scaled_train = scaler_for_woa.fit_transform(X_train_df.values)

            population_size = int(selector_params.get("population_size", 20))
            max_iter = int(selector_params.get("max_iter", 50))
            cv = int(selector_params.get("cv", 5))
            penalty_weight = float(selector_params.get("penalty_weight", 0.01))
            estimator = selector_params.get("estimator", None)
            random_state_sel = int(selector_params.get("random_state", random_state))

            selector_est = WOAFeatureSelector(
                population_size=population_size,
                max_iter=max_iter,
                estimator=estimator,
                cv=cv,
                penalty_weight=penalty_weight,
                random_state=random_state_sel,
            )

            selector_est.fit(X_scaled_train, y_train)
            # WOAFeatureSelector expone best_mask_ y best_score_
            mask = selector_est.best_mask_.astype(bool)
            idx = np.where(mask)[0]

            if idx.size > 0:
                X_train_sel_df = X_train_df.iloc[:, idx]
                X_test_sel_df = X_test_df.iloc[:, idx]
            else:
                X_train_sel_df = X_train_df
                X_test_sel_df = X_test_df

            selector_name = f"WOA (pop={population_size}, iters={max_iter}, cv={cv})"
            mask_for_report = mask.astype(int).tolist()
            fitness_for_report = float(
                getattr(selector_est, "best_score_", np.nan)
            )

        elif sel in ("none", "sin", "no"):
            selector_name = None
        else:
            raise ValueError(
                "Selector desconocido: usa 'bso-cv', 'm-abc', 'woa' o None."
            )

    # 8) Datos finales para el modelo (post-selección, pre-escalado)
    X_train_final = X_train_sel_df.values
    X_test_final = X_test_sel_df.values

    # 9) Construcción del Pipeline de imblearn con el wrapper del Transformer
    steps = []

    # Escalado dentro del pipeline usando scale_features (devuelve el scaler sklearn)
    scaler_step = scale_features(scaler_type)
    steps.append(("scaler", scaler_step))

    if use_smote:
        smote_random_state = int(selector_params.get("random_state", random_state))
        steps.append(("smote", SMOTE(random_state=smote_random_state)))

    clf = SklearnTransformerClassifier(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        random_state=random_state,
    )

    steps.append(("clf", clf))
    pipe = ImbPipeline(steps=steps)

    # 10) Entrenamiento + evaluación (con optimización opcional)
    best_params = None

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
            "clf__dropout": full_grid.get("dropout", dropout),
        }
        cv_folds = int(selector_params.get("cv", 5))

        # scorer manual para F1 macro (promedio de las clases)
        def f1_macro_scorer(estimator, X_val, y_val):
            y_pred = estimator.predict(X_val)
            return f1_score(y_val, y_pred, average="macro")

        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=f1_macro_scorer,  # optimizamos F1 macro
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gs.fit(X_train_final, y_train)

        # Guardar resultados de todas las combinaciones de hiperparámetros en CSV
        results = []

        for params in gs.cv_results_["params"]:
            # Clonamos el pipeline base y le ponemos estos hiperparámetros
            model_i = clone(pipe)
            model_i.set_params(**params)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model_i.fit(X_train_final, y_train)

            y_pred_i = model_i.predict(X_test_final)
            # Se asume que el clasificador expone predict_proba; si no, usamos y_pred
            if hasattr(model_i, "predict_proba"):
                y_proba_i = model_i.predict_proba(X_test_final)[:, 1]
            else:
                y_proba_i = y_pred_i

            iteration_result = {
                "y_true": y_test,
                "y_pred": y_pred_i,
                "y_pred_prob": y_proba_i,
                "hyperparameters": params,
                "selector": selector_name,
                "model_name": "transformer",
            }
            results.append(iteration_result)

        # Llamada a save_metrics_to_csv con los resultados
        save_metrics_to_csv(results, model_name="transformer")

        model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        # Sin optimizador: ajuste directo
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = pipe.fit(X_train_final, y_train)

    # 11) Evaluación en el conjunto de test
    y_pred = model.predict(X_test_final)
    try:
        y_proba = model.predict_proba(X_test_final)[:, 1]
    except Exception:
        y_proba = None

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    # 12) Reporte
    elapsed = round(time.time() - t0, 4)
    result = {
        "model": "transformer",
        "selector": selector_name,
        "metrics": metrics,
        "selected_features": list(X_train_sel_df.columns),
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
