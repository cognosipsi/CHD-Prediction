# eliminacionpearson.py
from __future__ import annotations
from typing import Optional, Sequence, Union
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PearsonRedundancyEliminator(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn que elimina únicamente la columna 'obesity' si existe.

    - Si X es un pandas.DataFrame: eliminará la columna 'obesity' (si está presente).
    - Si X es un numpy.ndarray u otra estructura sin nombres: no elimina nada.

    Compatibilidad:
    ---------------
    Se mantiene la firma de __init__(metodo, columnas_objetivo, strict) para
    compatibilidad con código previo, pero estos parámetros se IGNORAN.
    El comportamiento es fijo: solo se elimina 'obesity' cuando exista.
    """

    def __init__(
        self,
        metodo: Optional[str] = "pearson",
        columnas_objetivo: Optional[Sequence[Union[str, int]]] = None,
        strict: bool = False,
    ):
        # Se guardan por compatibilidad con get_params/set_params de sklearn,
        # pero se ignoran en la lógica.
        self.metodo = metodo
        self.columnas_objetivo = columnas_objetivo
        self.strict = strict

    def fit(self, X, y=None):
        # Detectar tipo de entrada y guardar metadatos esperados por sklearn
        self.is_dataframe_ = isinstance(X, pd.DataFrame)
        if self.is_dataframe_:
            self.feature_names_in_ = list(X.columns)
            self.n_features_in_ = X.shape[1]
            self.has_obesity_ = "obesity" in X.columns
        else:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X debe ser 2D (n_muestras, n_features).")
            self.feature_names_in_ = None
            self.n_features_in_ = X.shape[1]
            # Sin nombres de columnas, no podemos identificar 'obesity'
            self.has_obesity_ = False

        # Mensajes de compatibilidad si el usuario intenta configurar columnas/método
        if self.columnas_objetivo not in (None, (), [], ("obesity",)):
            warnings.warn(
                "PearsonRedundancyEliminator ahora siempre elimina solo 'obesity'. "
                "Los parámetros 'columnas_objetivo' y 'metodo' se ignoran.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.metodo not in (None, "pearson", "pearson_obesity", "mapa_pearson", "none", "sin", "no"):
            warnings.warn(
                "El parámetro 'metodo' se ignora. El transformador elimina únicamente 'obesity'.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    def transform(self, X):
        check_is_fitted(self, ["is_dataframe_", "has_obesity_", "n_features_in_"])
        if self.is_dataframe_:
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
            if self.has_obesity_ and "obesity" in X_df.columns:
                return X_df.drop(columns=["obesity"], errors="ignore")
            return X_df
        else:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X debe ser 2D (n_muestras, n_features).")
            # En ndarray no hay nombres, no se elimina nada
            return X

    def get_feature_names_out(self, input_features=None):
        """
        Devuelve los nombres de características tras la transformación.

        - Si se ajustó con DataFrame: devuelve los nombres originales menos 'obesity'.
        - Si se ajustó con ndarray: devuelve input_features (o None) sin cambios.
        """
        if self.is_dataframe_ and self.feature_names_in_ is not None:
            return np.array([c for c in self.feature_names_in_ if c != "obesity"], dtype=object)
        if input_features is None:
            return input_features
        # Si el pipeline provee nombres, retiramos 'obesity' de allí también
        return np.array([c for c in input_features if c != "obesity"], dtype=object)