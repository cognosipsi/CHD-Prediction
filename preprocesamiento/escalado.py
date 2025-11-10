from typing import Literal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin

def scale_features(scaler_type: Literal["minmax", "standard"]) -> TransformerMixin:
    """
    Devuelve una instancia del scaler según el parámetro.
    """
    if scaler_type == "minmax":
        return MinMaxScaler()
    elif scaler_type == "standard":
        return StandardScaler()
    else:
        raise ValueError("scaler_type debe ser 'minmax' o 'standard'")
