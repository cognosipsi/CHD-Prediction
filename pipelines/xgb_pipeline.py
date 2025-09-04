# pipelines/xgb_pipeline.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import time

# === Preprocesamiento ===
from preprocesamiento.lectura_datos import load_data
from preprocesamiento.codificacion import encode_features
from preprocesamiento.escalado import scale_features
from preprocesamiento.division_dataset import split_data

# === Predictores ===
from predictores.xgb import fit_and_predict

# === Selectores ===
from selectores.mabc import m_abc_feature_selection
from selectores.bsocv import bso_cv
from selectores.woa import woa_feature_selection
from selectores.eliminacionpearson import eliminar_redundancias

# === Reporte (igual que MLP) ===
from utils.evaluacion import print_metrics_from_values


def _apply_mask_df(X_df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    mask = np.asarray(mask).astype(int).ravel()
    if mask.shape[0] != X_df.shape[1]:
        raise ValueError(f"Máscara de longitud {mask.shape[0]} no coincide con #cols={X_df.shape[1]}")
    if mask.sum() == 0:
        # evita conjunto vacío
        mask[np.random.randint(0, mask.shape[0])] = 1
    cols = X_df.columns[mask == 1].tolist()
    return X_df.loc[:, cols]


def xgb_pipeline(
    file_path: str,
    selector: Optional[str] = "none",
    *,
    encoding_method: str = "labelencoder",
    scaler_type: str = "standard",
    redundancy: Optional[str] = "none",
    xgb_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    **selector_params,
) -> Dict[str, Any]:
    """
    Pipeline de XGBoost + selección de características.
    - selector: 'none' | 'm-abc' | 'mabc' | 'woa' | 'bso-cv' | 'bsocv'
    - selector_params: hiperparámetros del selector (pop_size, max_cycles, etc.).
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

    # 3) Redundancia (opcional)
    if redundancy and redundancy != 'none':
        # puedes pasar selector_params['redundancy_threshold'] si tu función lo soporta
        thr = selector_params.get('redundancy_threshold', None)
        try:
            if thr is None:
                X_df = eliminar_redundancias(X_df)
            else:
                X_df = eliminar_redundancias(X_df, threshold=thr)
        except TypeError:
            # compatibilidad con firmas antiguas
            X_df = eliminar_redundancias(X_df)

    # 4) Escalado previo (si el selector lo requiere para fitness)
    X_scaled = scale_features(X_df.values, scaler_type=scaler_type)

    # 5) Split preliminar (algunas heurísticas lo usan; no afecta el flujo final)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=test_size, random_state=random_state)

    # 6) Selección de características (opcional) sobre DataFrame (para preservar nombres)
    sel = (selector or 'none').lower()
    selector_name = 'none'
    feature_mask: Optional[np.ndarray] = None
    fitness_for_report: Optional[float] = None
    X_sel_df = X_df

    if sel in {'none', 'sin', 'ninguno', 'no'}:
        X_sel_df = X_df

    elif sel in {'m-abc', 'mabc', 'm_abc'}:
        # Pre-escalamos y dividimos para el fitness interno del M-ABC
        X_scaled = scale_features(X_df.values, scaler_type=scaler_type)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=test_size, random_state=random_state)

        pop_size     = selector_params.get("pop_size", 20)
        max_cycles   = selector_params.get("max_cycles", selector_params.get("max_iter", 30))
        limit        = selector_params.get("limit", 5)
        patience     = selector_params.get("patience", 10)
        cv_folds     = selector_params.get("cv_folds", 5)
        knn_k        = selector_params.get("knn_k", 5)
        rs           = selector_params.get("random_state", random_state)
        verbose      = selector_params.get("verbose", False)

        best_mask, best_fitness = m_abc_feature_selection(
            X_train, X_test, y_train, y_test,
            use_custom_evaluator=False,
            pop_size=pop_size,
            max_cycles=max_cycles,
            limit=limit,
            patience=patience,
            random_state=rs,
            cv_folds=cv_folds,
            knn_k=knn_k,
            verbose=verbose,
        )
        feature_mask = np.asarray(best_mask).astype(int)
        fitness_for_report = float(best_fitness)
        selector_name = f"M-ABC(pop={pop_size}, cycles={max_cycles}, cv={cv_folds})"
        X_sel_df = _apply_mask_df(X_df, feature_mask)

    elif sel in {'woa', 'whale', 'ballenas'}:
        # WOA con fitness por CV; si tu implementación soporta estimator pásalo vía selector_params['estimator']
        population_size = selector_params.get("population_size", 20)
        max_iter        = selector_params.get("max_iter", 50)
        cv              = selector_params.get("cv", 5)
        random_state_w  = selector_params.get("random_state", random_state)
        penalty_weight  = selector_params.get("penalty_weight", 0.01)
        verbose         = selector_params.get("verbose", False)

        X_for_fs = scale_features(X_df.values, scaler_type=scaler_type)

        mask, woa_fit = woa_feature_selection(
            X_for_fs, y,
            population_size=population_size,
            max_iter=max_iter,
            estimator=selector_params.get("estimator", None),
            cv=cv,
            penalty_weight=penalty_weight,
            random_state=random_state_w,
        )
        feature_mask = np.asarray(mask).astype(int)
        fitness_for_report = float(woa_fit)
        selector_name = f"WOA(pop={population_size}, iters={max_iter}, cv={cv})"
        X_sel_df = _apply_mask_df(X_df, feature_mask)

    elif sel in {'bso-cv', 'bsocv', 'bso'}:
        # BSO-CV espera DataFrame para poder indexar columnas (usa X.iloc internamente)
        population_size = int(selector_params.get("population_size", 20))
        max_iter        = int(selector_params.get("max_iter", 50))
        cv              = int(selector_params.get("cv", 5))
        random_state_b  = selector_params.get("random_state", random_state)
        penalty_weight  = float(selector_params.get("penalty_weight", 0.01))
        verbose         = bool(selector_params.get("verbose", False))

        mask, bso_fit = bso_cv(
            X_df, df["chd"],
            population_size=population_size,
            max_iter=max_iter,
            cv=cv,
            random_state=random_state_b,
            penalty_weight=penalty_weight,
            verbose=verbose,
        )
        feature_mask = np.asarray(mask).astype(int)
        fitness_for_report = float(bso_fit)
        selector_name = f"BSO-CV(pop={population_size}, iters={max_iter}, cv={cv})"
        X_sel_df = _apply_mask_df(X_df, feature_mask)

    else:
        raise ValueError(f"Selector desconocido: {selector}")

    # 7) Escalado y split sobre el subconjunto de columnas seleccionado (flujo final)
    X_scaled = scale_features(X_sel_df.values, scaler_type=scaler_type)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=test_size, random_state=random_state)

    # 8) Entrena y evalúa XGB sobre las features seleccionadas
    params = xgb_params or {}
    metrics = fit_and_predict(
        X_train, y_train,
        X_test,  y_test,
        params=params,
    )

    # 9) Reporte (idéntico al estilo del MLP)
    selected_columns: List[str] = list(X_sel_df.columns)
    elapsed = round(time.time() - t0, 4)
    print_metrics_from_values(
        metrics.get("accuracy", float("nan")),
        metrics.get("precision", float("nan")),
        metrics.get("recall", float("nan")),
        metrics.get("f1", float("nan")),
        metrics.get("auc", float("nan")),
        model="XGBClassifier",
        selector_name=selector_name if selector_name != 'none' else None,
        selected_columns=selected_columns,
        mask=feature_mask.tolist() if feature_mask is not None else None,
        fitness=fitness_for_report,
        elapsed_seconds=elapsed,
        extra_info={"tiempo_s": elapsed},
    )

    # 10) Empaquetar resultado
    result: Dict[str, Any] = {
        "model": "XGBClassifier",
        "selector": selector_name,
        "selected_features": selected_columns,
        "n_selected": len(selected_columns),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "extra_info": {"tiempo_s": elapsed},
    }
    if feature_mask is not None:
        result["mask"] = feature_mask.tolist()
    if fitness_for_report is not None:
        result["selector_fitness"] = fitness_for_report

    return result
