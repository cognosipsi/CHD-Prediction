# pipelines/mlp_pipeline.py
from __future__ import annotations
from typing import Optional
import time
import numpy as np

from preprocesamiento.lectura_datos import load_data
from preprocesamiento.codificacion import encode_features
from preprocesamiento.escalado import scale_features
from preprocesamiento.division_dataset import split_data

# Selectores
from selectores.bsocv import BSOFeatureSelector
from selectores.mabc import MABCFeatureSelector
from selectores.woa import WOAFeatureSelector
from selectores.eliminacionpearson import PearsonRedundancyEliminator

# Predictor
from predictores.mlp import mlp_evaluator

#Optimizadores
from optimizadores.gridSearchCV import save_metrics_to_csv

# Reporte
from utils.evaluacion import compute_classification_metrics, print_from_pipeline_result

#sklearn & imblearn
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

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
    max_iter: int = 1000,              # = mlpWOA.py
    random_state: int = 42,
    early_stopping: bool = True,     # = mlpWOA.py
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
        df = PearsonRedundancyEliminator().fit_transform(df)

    # 3) X, y
    if "chd" not in df.columns:
        raise ValueError("La columna objetivo 'chd' no se encuentra en el dataset.")
    X_df = df.drop(columns=["chd"])
    y = df["chd"].values
    feature_names = list(X_df.columns)

    # 4) Hold-out split (sin SMOTE aquí)
    X_train, X_test, y_train, y_test = split_data(X_df.values, y, test_size=0.3, random_state=random_state)

    # 6) Construcción del Pipeline de imblearn
    steps = []

    # 4) Selección de características (opcional) — incluye BSO-CV, M-ABC y WOA
    sel = (selector or "none").strip().lower() if selector is not None else "none"
    selector_name = None

    if sel in ("bso", "bso-cv", "bsocv"):
        # Igual que antes: no requiere escalado global
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
        # Escalado rápido sobre TODO X SOLO para el selector (heurística interna)

        population_size     = int(selector_params.get("population_size", 20))
        max_iter   = int(selector_params.get("max_iter", selector_params.get("max_iter", 30)))
        limit        = int(selector_params.get("limit", 5))
        patience     = int(selector_params.get("patience", 10))
        random_state_s = selector_params.get("random_state", 42)
        cv_folds     = int(selector_params.get("cv_folds", 5))
        knn_k        = int(selector_params.get("knn_k", 5))
        verbose      = bool(selector_params.get("verbose", False))

        mabc = MABCFeatureSelector(
            knn_k=knn_k,
            population_size=population_size,
            max_iter=max_iter,     # <- importante
            limit=limit,
            patience=patience,
            cv_folds=cv_folds,
            random_state=random_state_s,
            verbose=verbose,
        )
        steps.append(("selector", mabc))
        selector_name = f"M-ABC (pop={population_size}, cycles={max_iter}, cv={cv_folds}, k={knn_k})"

    elif sel in ("woa", "whale", "whale-optimization"):
        # WOA: para coherencia con MLP, usamos KNN en el fitness + pre-escalado temporal
        n_neighbors = int(selector_params.get("n_neighbors", 3))
        population_size = int(selector_params.get("population_size", 20))
        max_iter = int(selector_params.get("max_iter", 50))
        cv = int(selector_params.get("cv", 5))
        penalty_weight = float(selector_params.get("penalty_weight", 0.01))
        random_state_w = selector_params.get("random_state", 42)

        # Escalado dentro de la CV del fitness de WOA:
        estimator_for_fitness = Pipeline([
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
        selector_name = "none"
        pass
    else:
        raise ValueError("Selector desconocido: usa 'bso-cv', 'm-abc', 'woa' o None.")

    # Paso SMOTE (opcional, DENTRO del pipeline)
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
        
    # Scaler + Clasificador
    steps.append(("scaler", scale_features(scaler_type)))

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=early_stopping,
        tol=tol,
    )
    steps.append(("clf", clf))

    # Decidimos si el pipeline base lleva SMOTE (ImbPipeline) o no
    if any(name == "smote" for name, _ in steps):
        pipe = ImbPipeline(steps=steps)
    else:
        pipe = Pipeline(steps=steps)

    # 6) Entrenamiento (con o sin GridSearch) sobre el PIPELINE COMPLETO
    best_params = None
    model = pipe

    # Ejecuta GridSearchCV con el estimador sklearn MLPClassifier
    if optimizer and str(optimizer).lower() == "gridsearchcv":
        param_grid = {
            "selector__population_size": [selector_params.get("population_size", 20)],
            "selector__max_iter": [selector_params.get("max_iter", 50)],
            "clf__hidden_layer_sizes": [(50,), (100,), (150,)],  # Agregado según el archivo gridSearchCV.py
            "clf__activation": ["relu", "tanh"],                  # Agregado según el archivo gridSearchCV.py
            "clf__solver": ["adam", "sgd"],                       # Agregado según el archivo gridSearchCV.py
            "clf__max_iter": [200, 500, 1000],
        }

        # Solo agregamos hiperparámetros del selector si el paso existe
        has_selector = any(name == "selector" for name, _ in steps)
        if has_selector:
            param_grid.update(
                {
                    "selector__population_size": [
                        selector_params.get("population_size", 20)
                    ],
                    "selector__max_iter": [selector_params.get("max_iter", 50)],
                }
            )

        gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
        
        # Silenciar los warnings de convergencia dentro de GridSearchCV
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gs.fit(X_train, y_train)

        # Recoger resultados por combinación de hiperparámetros (una fila por iteración)
        results = []
        best_f1_macro = float("-inf")
        best_result_for_print = None

        for i, params in enumerate(gs.cv_results_["params"]):
            model_i = clone(pipe)
            model_i.set_params(**params)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model_i.fit(X_train, y_train)

            y_pred_i = model_i.predict(X_test)
            if hasattr(model_i, "predict_proba"):
                y_proba_i = model_i.predict_proba(X_test)[:, 1]
            else:
                y_proba_i = y_pred_i

            # Métricas en test para poder quedarnos con el mejor F1
            metrics_i = compute_classification_metrics(y_test, y_pred_i, y_proba_i)

            # F1 macro como criterio principal (con pequeños fallbacks por compatibilidad)
            if isinstance(metrics_i, dict):
                f1_macro_i = metrics_i.get("f1_macro")
                if f1_macro_i is None:
                    # Por si aún no actualizaste compute_classification_metrics
                    f1_macro_i = metrics_i.get("f1_score", metrics_i.get("f1", 0.0))
            else:
                f1_macro_i = 0.0

            if f1_macro_i is None:
                f1_macro_i = 0.0

            if f1_macro_i > best_f1_macro:
                best_f1_macro = f1_macro_i
                best_result_for_print = {
                    "iteration": i + 1,
                    "hyperparameters": params,
                    "metrics": metrics_i,
                }

            iteration_result = {
                "y_true": y_test,
                "y_pred": y_pred_i,
                "y_pred_prob": y_proba_i,
                "hyperparameters": params,
                "mean_test_score": gs.cv_results_["mean_test_score"][i],
                "std_test_score": gs.cv_results_["std_test_score"][i],
            }
            results.append(iteration_result)

        # Guardar métricas de todas las iteraciones del grid
        save_metrics_to_csv(results, model_name="mlp")

        # Imprimir resultados del registro con mejor F1 en test
        if best_result_for_print is not None:
            print("\n=== Registro con mejor F1-Macro (test) ===")
            print(f"Iteración (1-based): {best_result_for_print['iteration']}")
            print("Hiperparámetros:")
            for k, v in best_result_for_print["hyperparameters"].items():
                print(f"  {k}: {v}")
            print("Métricas:")
            for k, v in best_result_for_print["metrics"].items():
                print(f"  {k}: {v}")
                
        # El modelo final sigue siendo el best_estimator_ de GridSearchCV
        model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        model = pipe.fit(X_train, y_train)

    # 7) Evaluación en test
    # Recuperamos (si existe) el selector del pipeline final
    sel_step = None
    if hasattr(model, "named_steps"):
        sel_step = model.named_steps.get("selector")

    if optimizer and str(optimizer).lower() == "gridsearchcv":
        # Con GridSearchCV usamos directamente el best_estimator_
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
    else:
        # Sin GridSearchCV usamos mlp_evaluator, pero usando la máscara del selector del pipeline
        mask = sel_step.get_support() if (
            sel_step is not None and hasattr(sel_step, "get_support")
        ) else None

        y_pred, y_proba = mlp_evaluator(
            X_train,
            y_train,
            X_test,
            mask=mask,
            use_smote=use_smote,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            tol=tol,
        )

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    # 8) Determinar características seleccionadas (coherente con transformer_pipeline)
    selected_features = feature_names
    if sel_step is not None and hasattr(sel_step, "get_support"):
        support = sel_step.get_support()
        if support is not None:
            support_arr = np.array(support)
            # Caso máscara booleana
            if support_arr.dtype == bool and support_arr.size == len(feature_names):
                selected_features = [
                    name for name, keep in zip(feature_names, support_arr) if keep
                ]
            # Caso índices enteros
            elif np.issubdtype(support_arr.dtype, np.integer):
                selected_features = [
                    feature_names[idx]
                    for idx in support_arr
                    if 0 <= idx < len(feature_names)
                ]

    # 9) Resultado estandarizado
    elapsed = round(time.time() - t0, 4)
    result = {
        "model": "mlp",
        "selector": selector_name,
        "metrics": metrics,
        "selected_features": selected_features,
        "elapsed_seconds": elapsed,
        "extra_info": {
            "optimizer": (optimizer if isinstance(optimizer, str) else getattr(optimizer, "__name__", str(optimizer))),
            "best_params": best_params,
        },
    }
    print_from_pipeline_result(result)
    return result