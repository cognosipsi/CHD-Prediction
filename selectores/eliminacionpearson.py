# selectores/eliminacionpearson.py
from __future__ import annotations
from typing import Optional, Sequence, List
import pandas as pd

def eliminar_redundancias(
    df: pd.DataFrame,
    metodo: Optional[str] = "pearson",
    columnas_objetivo: Sequence[str] = ("obesity",),
) -> pd.DataFrame:
    """
    Elimina variables redundantes detectadas por un análisis (p. ej., mapa de Pearson).
    En tu caso, se decidió que 'obesity' es redundante.

    Parámetros:
      - df: DataFrame de entrada.
      - metodo: "pearson" para aplicar la lista fija; "none"/None para no aplicar.
      - columnas_objetivo: columnas a eliminar cuando metodo="pearson" (por defecto ["obesity"]).

    Retorna:
      - df sin las columnas redundantes (si existen).
    """
    if metodo is None or str(metodo).lower() in {"none", "sin", "no"}:
        return df

    metodo_l = str(metodo).lower().strip()
    if metodo_l in {"pearson", "pearson_obesity", "mapa_pearson"}:
        to_drop: List[str] = [c for c in columnas_objetivo if c in df.columns]
        if to_drop:
            return df.drop(columns=to_drop)
        return df

    raise ValueError(
        f"Método de eliminación de redundancias desconocido: {metodo}. "
        "Usa 'pearson' o 'none'."
    )
