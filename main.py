# Pipelines de predictores
from pipelines.knn_pipeline import knn_pipeline
from pipelines.mlp_pipeline import mlp_pipeline
from pipelines.xgb_pipeline import xgb_pipeline
from pipelines.transformer_pipeline import transformer_pipeline

#Optimizadores
from optimizadores.gridSearchCV import run_grid_search

# --- Opciones disponibles ---
PIPELINES = {
    "knn": knn_pipeline,
    "mlp": mlp_pipeline,
    "transformer": transformer_pipeline,
    "xgb": xgb_pipeline,
}

SELECTORS = {
    "bso-cv": "bso_cv",
    "m-abc": "m-abc",
    "woa": "woa",
    "none": None,
}

OPTIMIZERS = {
    "gridsearchcv": run_grid_search,
    "none": None,
}

ENCODINGS = ["labelencoder", "manual"]
SCALERS = ["minmax", "standard"]
REDUNDANCY = ["pearson", "none"]

# --- Helpers de entrada ---
def _prompt_choice(opciones, mensaje="Elige una opción", allow_empty=False, default=None):
    opciones_list = list(opciones if isinstance(opciones, (list, tuple)) else opciones.keys())
    opciones_str = ", ".join(opciones_list)
    while True:
        raw = input(f"{mensaje} ({opciones_str}){(' ['+default+']' if default else '')}: ").strip().lower()
        if allow_empty and raw == "" and default is not None:
            return default
        if raw in opciones_list:
            return raw
        print(f"Opción no válida. Intenta de nuevo ({opciones_str}).")

def _build_selector_params(selector: str) -> dict:
    """
    Construye 'selector_params' con valores por defecto coherentes con las
    implementaciones de selectores.
    """
    selector = (selector or "none").lower()
    if selector in {"m-abc", "mabc", "m_abc"}:
        return {
            "population_size": 20,
            "max_iter": 50,  # alias de max_iter
            "limit": 5,
            "patience": 10,
            "cv_folds": 5,
            "knn_k": 5,
            "random_state": 42,
            "verbose": False,
        }
    if selector in {"woa"}:
        return {
            "population_size": 20,
            "max_iter": 50,
            "cv": 5,
            "random_state": 42,
            "penalty_weight": 0.01,
            "verbose": False,
        }
    if selector in {"bso-cv", "bsocv", "bso"}:
        return {
            "population_size": 20,
            "max_iter": 50,
            "cv": 5,
            "random_state": 42,
            "penalty_weight": 0.01,
            "verbose": False,
        }
    return {}

# Pretty print uniforme para todos los pipelines que devuelven un dict
def _pretty_print_result(res: dict):
    if not isinstance(res, dict):
        print(res)
        return
    print("\n=== RESUMEN ===")
    print(f"Modelo: {res.get('model')}")
    sel_name = res.get("selector", "none")
    if sel_name and sel_name != "none":
        print(f"Selector: {sel_name}")
    metrics = res.get("metrics", {})
    if metrics:
        print("Métricas:")
        for k, v in metrics.items():
            print(f"  - {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  - {k}: {v}")
    n_selected = res.get("n_selected")
    if n_selected is not None:
        print(f"Features seleccionadas: {n_selected}")
    cols = res.get("selected_features")
    if cols:
        print("Columnas:")
        print(", ".join(map(str, cols)))

def _normalize_scaler_name(name: str) -> str:
    aliases = {
        "standardscaler": "standard",
        "std": "standard",
        "zscore": "standard",
        "minmaxscaler": "minmax",
    }
    return aliases.get(name, name)

if __name__ == "__main__":
    file_path = "SAHeart.csv"

    modelo = _prompt_choice(list(PIPELINES.keys()), "¿Qué modelo deseas evaluar?")
    optimizer = _prompt_choice(list(OPTIMIZERS.keys()),
                              "¿Qué modelo de optimización de parámetros deseas usar? "
                              "(elige 'none' para no usar)", allow_empty=True, default="none")
    selector = _prompt_choice(list(SELECTORS.keys()),
                              "¿Qué método de selección usar? (elige 'none' para no usar)",
                              allow_empty=True, default="none")
    encoding_method = _prompt_choice(ENCODINGS, "¿Método de codificación?",
                                     allow_empty=True, default="labelencoder")
    scaler_type = _prompt_choice(SCALERS + ["standardscaler", "minmaxscaler"], "¿Tipo de escalado?",
                                 allow_empty=True, default="standard")
    redundancy = _prompt_choice(REDUNDANCY, "¿Desea eliminar redundancias?",
                                allow_empty=True, default="none")

    # Normalizamos posibles alias de scaler
    scaler_type = _normalize_scaler_name(scaler_type)
    selector_params = _build_selector_params(selector)

    print(f"\nEjecutando pipeline: {modelo.upper()} | selector={selector} | "
          f"encoding={encoding_method} | scaler={scaler_type} | redundancy={redundancy} | archivo={file_path}",
          flush=True)

    # Ejecutar pipeline y mostrar resultado
    res = PIPELINES[modelo](
        file_path,
        selector=None if selector == 'none' else selector,
        encoding_method=encoding_method,
        scaler_type=scaler_type,
        redundancy=None if redundancy == 'none' else redundancy,
        optimizer=None if OPTIMIZERS.get(optimizer) in (None, "none") else "gridsearchcv",
        **(selector_params or {}),
    )
    _pretty_print_result(res)