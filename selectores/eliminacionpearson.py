# selectores/eliminacionpearson.py
from __future__ import annotations
from typing import Optional, Sequence, Union, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PearsonRedundancyEliminator(BaseEstimator, TransformerMixin):
    """
    Eliminador simple de redundancias basado en una decisión fija (p. ej., mapa de Pearson).
    Mantiene la lógica original: si metodo="pearson", elimina las columnas indicadas en
    `columnas_objetivo` (por nombre o posición). Si metodo=None/"none"/"sin"/"no", no hace nada.

    Parámetros
    ----------
    metodo : {"pearson", "pearson_obesity", "mapa_pearson", "none", None}, opcional (por defecto "pearson")
        Método a aplicar. "pearson" aplica la lista fija de columnas a remover.
        "none"/None (o sin/no) ⇒ no elimina columnas.
    columnas_objetivo : Sequence[Union[str, int]], opcional (por defecto ("obesity",))
        Columnas a eliminar cuando metodo="pearson". Pueden ser nombres (para DataFrame)
        o posiciones enteras (para ndarray).
    strict : bool, opcional (por defecto False)
        Si True, un metodo desconocido produce ValueError; si False, actúa como no-op.

    Notas
    -----
    - Si X es DataFrame, prioriza eliminar por NOMBRE de columna.
    - Si X es ndarray y `columnas_objetivo` contiene enteros, elimina por ÍNDICE.
    - Si X es ndarray y `columnas_objetivo` contiene nombres (strings), no elimina nada
      (los nombres no existen en un ndarray) para mantener un comportamiento seguro.
    """

    def __init__(
        self,
        metodo: Optional[str] = "pearson",
        columnas_objetivo: Sequence[Union[str, int]] = ("obesity",),
        strict: bool = False,
    ):
        self.metodo = metodo
        self.columnas_objetivo = tuple(columnas_objetivo)
        self.strict = strict

    # ------------------------- API sklearn -------------------------

    def fit(self, X, y=None):
        metodo = None if self.metodo is None else str(self.metodo).lower().strip()
        self._metodo_ = metodo

        self.is_dataframe_ = isinstance(X, pd.DataFrame)
        if self.is_dataframe_:
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim != 2:
                raise ValueError("X debe ser 2D (n_samples, n_features).")
            # Nombres sintéticos para cumplir la API (no se usan para eliminar por nombre)
            self.feature_names_in_ = np.asarray([f"x{i}" for i in range(X_arr.shape[1])], dtype=object)

        # Por defecto: no eliminar nada
        self.cols_to_drop_idx_ = []   # índices a eliminar
        self.cols_to_drop_names_ = [] # nombres a eliminar (cuando X es DataFrame)

        # Ramas de comportamiento
        if metodo is None or metodo in {"none", "sin", "no"}:
            return self

        if metodo in {"pearson", "pearson_obesity", "mapa_pearson"}:
            # Resolver a índices y nombres según corresponda
            idx_to_drop: List[int] = []
            names_to_drop: List[str] = []

            # Si es DF: intentar por nombre; si es ndarray: intentar por índice si son enteros
            if self.is_dataframe_:
                present = set(self.feature_names_in_.tolist())
                for c in self.columnas_objetivo:
                    if isinstance(c, str) and c in present:
                        names_to_drop.append(c)
                    elif isinstance(c, int) and 0 <= c < len(self.feature_names_in_):
                        idx_to_drop.append(c)
                # Completar idx a partir de nombres
                if names_to_drop:
                    name_set = set(names_to_drop)
                    idx_to_drop.extend(i for i, n in enumerate(self.feature_names_in_) if n in name_set)
            else:
                # ndarray: sólo índices enteros tienen efecto
                for c in self.columnas_objetivo:
                    if isinstance(c, int) and 0 <= c < len(self.feature_names_in_):
                        idx_to_drop.append(c)

            # Guardar únicos y ordenados
            self.cols_to_drop_idx_ = sorted(set(idx_to_drop))
            self.cols_to_drop_names_ = [self.feature_names_in_[i] for i in self.cols_to_drop_idx_]
            return self

        # Método desconocido
        if self.strict:
            raise ValueError(
                f"Método de eliminación de redundancias desconocido: {self.metodo}. "
                "Usa 'pearson' o 'none'."
            )
        # Si no es estricto, actúa como no-op
        return self

    def transform(self, X):
        check_is_fitted(self, ["feature_names_in_", "cols_to_drop_idx_", "cols_to_drop_names_", "_metodo_"])

        # No-op según metodo
        if self._metodo_ is None or self._metodo_ in {"none", "sin", "no"}:
            return X

        if isinstance(X, pd.DataFrame):
            if self.cols_to_drop_names_:
                cols_presentes = [c for c in self.cols_to_drop_names_ if c in X.columns]
                if cols_presentes:
                    return X.drop(columns=cols_presentes)
            # Si no hay nombres presentes pero sí índices válidos, intentamos por posición
            if self.cols_to_drop_idx_:
                keep_idx = [i for i in range(X.shape[1]) if i not in set(self.cols_to_drop_idx_)]
                return X.iloc[:, keep_idx]
            return X

        # ndarray
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X debe ser 2D (n_samples, n_features).")

        if self.cols_to_drop_idx_:
            mask = np.ones(X_arr.shape[1], dtype=bool)
            mask[self.cols_to_drop_idx_] = False
            return X_arr[:, mask]

        return X_arr

    def get_feature_names_out(self, input_features=None):
        """Devuelve los nombres de características tras la eliminación."""
        if input_features is None:
            input_features = self.feature_names_in_
        names = np.asarray(input_features, dtype=object)

        if self.cols_to_drop_idx_:
            keep_idx = [i for i in range(len(names)) if i not in set(self.cols_to_drop_idx_)]
            return names[keep_idx]

        return names


# ----------------- Compatibilidad hacia atrás (función original) -----------------

def eliminar_redundancias(
    df: pd.DataFrame,
    metodo: Optional[str] = "pearson",
    columnas_objetivo: Sequence[Union[str, int]] = ("obesity",),
) -> pd.DataFrame:
    """
    Conserva la API histórica devolviendo un DataFrame transformado.
    Equivale a ajustar y transformar con PearsonRedundancyEliminator.
    """
    tr = PearsonRedundancyEliminator(metodo=metodo, columnas_objetivo=columnas_objetivo)
    return tr.fit_transform(df)
