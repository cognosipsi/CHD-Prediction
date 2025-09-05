from typing import Any, Dict, Iterable, Mapping, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

def compute_classification_metrics(
    y_true,
    y_pred,
    y_prob: Optional[Iterable] = None,
) -> Dict[str, float]:
    """
    Calcula métricas de clasificación y retorna:
    accuracy, precision, recall, f1, auc (NaN si no hay y_prob).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    metrics["auc"] = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")

        # Calculando la matriz de confusión para obtener los valores de los TP, TN, FP y FN
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positive"] = tp
    metrics["false_positive"] = fp
    metrics["true_negative"] = tn
    metrics["false_negative"] = fn

    return metrics

def _print_extra_info(extra_info: Mapping[str, Any]) -> None:
    if not extra_info:
        return
    print("Info extra:")
    for k in sorted(extra_info.keys()):
        print(f"  - {k}: {extra_info[k]}")

def print_metrics(
    metrics: Mapping[str, float],
    *,
    model: Optional[str] = None,
    selector_name: Optional[str] = None,
    selected_columns: Optional[Iterable[str]] = None,
    mask: Optional[Iterable[int]] = None,
    fitness: Optional[float] = None,
    elapsed_seconds: Optional[float] = None,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Imprime un reporte estándar de métricas y metadatos.
    Los parámetros después de '*' son SOLO por nombre (keyword-only).
    """
    if model:
        print(f"Modelo: {model}")
    if selector_name:
        print(f"Selector: {selector_name}")
    if selected_columns is not None:
        cols = list(selected_columns)
        print(f"Features seleccionadas: {len(cols)}")
        print("Columnas:", cols)
    if mask is not None:
        print("Máscara de features (1=seleccionada):", list(mask))
    if fitness is not None:
        print(f"Fitness (durante selección): {fitness:.4f}")

    if elapsed_seconds is not None:
        print(f"Tiempo total (s): {elapsed_seconds:.4f}")

    if extra_info:
        if "tiempo_s" in extra_info and elapsed_seconds is None:
            try:
                print(f"Tiempo total (s): {float(extra_info['tiempo_s']):.4f}")
            except Exception:
                pass
        _print_extra_info(extra_info)

    acc = metrics.get("accuracy", float("nan"))
    pre = metrics.get("precision", float("nan"))
    rec = metrics.get("recall", float("nan"))
    f1  = metrics.get("f1", float("nan"))
    auc = metrics.get("auc", float("nan"))
    tp  = metrics.get("true_positive", float("nan"))
    fp  = metrics.get("false_positive", float("nan"))
    tn  = metrics.get("true_negative", float("nan"))
    fn  = metrics.get("false_negative", float("nan"))

    print(f"Accuracy:  {acc:.4f}" if isinstance(acc, (int, float)) else f"Accuracy:  {acc}")
    print(f"Precision: {pre:.4f}" if isinstance(pre, (int, float)) else f"Precision: {pre}")
    print(f"Recall:    {rec:.4f}" if isinstance(rec, (int, float)) else f"Recall:    {rec}")
    print(f"F1 Score:  {f1:.4f}" if isinstance(f1, (int, float)) else f"F1 Score:  {f1}")
    if isinstance(auc, (int, float)) and auc == auc:  # no NaN
        print(f"AUC:       {auc:.4f}")
    else:
        print("AUC:       N/A")
    print(f"True Positive:  {tp}")
    print(f"False Positive: {fp}")
    print(f"True Negative:  {tn}")
    print(f"False Negative: {fn}")

def print_metrics_from_values(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    auc: float,
    *,
    model: Optional[str] = None,
    selector_name: Optional[str] = None,
    selected_columns: Optional[Iterable[str]] = None,
    mask: Optional[Iterable[int]] = None,
    fitness: Optional[float] = None,
    elapsed_seconds: Optional[float] = None,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Helper cuando el modelo ya devuelve las métricas como valores sueltos.
    Soporta los mismos kwargs que `print_metrics`.
    """
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }
    print_metrics(
        metrics,
        model=model,
        selector_name=selector_name,
        selected_columns=selected_columns,
        mask=mask,
        fitness=fitness,
        elapsed_seconds=elapsed_seconds,
        extra_info=extra_info,
    )

def print_from_pipeline_result(result: Mapping[str, Any]) -> None:
    """
    Impresor canónico para el dict que retorna cada pipeline.
    Claves toleradas:
      model, selector, metrics, selected_features, n_selected, mask,
      selector_fitness, elapsed_seconds, extra_info
    """
    if not isinstance(result, Mapping):
        print(result)
        return

    model = result.get("model")
    selector_name = result.get("selector")
    metrics = result.get("metrics", {})
    selected_columns = result.get("selected_features")
    mask = result.get("mask")
    fitness = result.get("selector_fitness")
    elapsed_seconds = result.get("elapsed_seconds")
    extra_info = result.get("extra_info")

    if selected_columns is None and "n_selected" in result:
        print(f"Features seleccionadas: {result['n_selected']}")

    print_metrics(
        metrics,
        model=model,
        selector_name=selector_name,
        selected_columns=selected_columns,
        mask=mask,
        fitness=fitness,
        elapsed_seconds=elapsed_seconds,
        extra_info=extra_info,
    )
