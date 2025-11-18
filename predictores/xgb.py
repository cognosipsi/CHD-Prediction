from typing import Optional, Dict, Any
from xgboost import XGBClassifier

# Parámetros por defecto (sin 'use_label_encoder', deprecado)
DEFAULT_PARAMS: Dict[str, Any] = dict(
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

def build_xgb(custom_params: Optional[Dict[str, Any]] = None) -> XGBClassifier:
    """
    Crea una instancia de XGBClassifier con parámetros por defecto + overrides.
    Filtra claves desconocidas para evitar mensajes de 'not used'.
    """
    params = DEFAULT_PARAMS.copy()
    if custom_params:
        accepted = XGBClassifier().get_params()
        params.update({k: v for k, v in custom_params.items() if k in accepted})
    return XGBClassifier(**params)