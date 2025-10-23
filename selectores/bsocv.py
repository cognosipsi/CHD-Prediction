from __future__ import annotations

from typing import Iterable, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

#sklearn imports

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

def _fitness(X: pd.DataFrame, y: Iterable, individual, *, cv: int = 5, penalty_weight: float = 0.01) -> float:
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if not selected_features:
        # Penalización máxima si no hay features
        return 1.0
    X_selected = X.iloc[:, selected_features]
    model = KNeighborsClassifier(n_neighbors=5)
    accuracy = cross_val_score(model, X_selected, y, cv=cv).mean()
    feature_ratio = len(selected_features) / X.shape[1]
    # Minimizar: (1-accuracy) + penalización por número de features
    return 1 - accuracy + penalty_weight * feature_ratio

def _crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    point = np.random.randint(1, len(parent1)-1)
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child

def bso_cv(
        X: pd.DataFrame, 
        y: Iterable, 
        *, 
        population_size: int=20, 
        max_iter: int=50, 
        cv: int=5,
        random_state: Optional[int]=None, 
        penalty_weight: float=0.01, 
        verbose: bool =False,
        ) -> Tuple[np.ndarray, float]:
    """
    BSO-CV simple con KNN y penalización por cantidad de características.
    Retorna:
      - best_individual: vector binario (np.ndarray) de longitud n_features
      - best_fitness: float (menor es mejor)
    """
    if random_state is not None:
        np.random.seed(int(random_state))

    if not isinstance(X, pd.DataFrame):
        # Aseguramos DataFrame para usar .iloc
        X = pd.DataFrame(X)

    population = np.random.randint(0, 2, size=(population_size, X.shape[1]))
    best_individual = None
    best_fitness = np.inf

    for iteration in range(max_iter):
        fitness_scores = np.array([
            _fitness(X, y, ind, cv=cv, penalty_weight=penalty_weight)
            for ind in population
        ])
        sorted_idx = np.argsort(fitness_scores)
        population = population[sorted_idx]

        if fitness_scores[0] < best_fitness:
            best_fitness = float(fitness_scores[0])
            best_individual = population[0].copy()

        # Crossover por parejas
        new_population = []
        for i in range(0, population.shape[0], 2):
            if i + 1 < population.shape[0]:
                child1 = _crossover(population[i], population[i+1])
                child2 = _crossover(population[i+1], population[i])
                new_population.extend([child1, child2])
            else:
                new_population.append(population[i])
        population = np.array(new_population)

        if verbose:
            print(f"[BSO-CV] iter={iteration+1}/{max_iter} fitness={best_fitness:.4f}")

    return best_individual, best_fitness

# ======================================================================
# 2) WRAPPER SKLEARN (para usar en Pipeline / GridSearchCV)
# ======================================================================

ArrayLike = Union[pd.DataFrame, np.ndarray]


def _as_dataframe(X: ArrayLike) -> pd.DataFrame:
    """Convierte a DataFrame si viene como ndarray, preserva DataFrame si ya lo es."""
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])


class BSOFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selector de características basado en BSO-CV, compatible con sklearn.

    Parámetros
    ----------
    population_size : int, default=20
    max_iter : int, default=50
    cv : int, default=5
    random_state : int | None, default=None
    penalty_weight : float, default=0.01
    verbose : bool, default=False

    Atributos tras fit()
    --------------------
    selected_mask_ : np.ndarray (bool), shape (n_features,)
    selected_idx_ : np.ndarray (int)
    fitness_ : float
    n_features_in_ : int
    feature_names_in_ : np.ndarray (str)
    """

    def __init__(
        self,
        *,
        population_size: int = 20,
        max_iter: int = 50,
        cv: int = 5,
        random_state: Optional[int] = None,
        penalty_weight: float = 0.01,
        verbose: bool = False,
    ) -> None:
        self.population_size = population_size
        self.max_iter = max_iter
        self.cv = cv
        self.random_state = random_state
        self.penalty_weight = penalty_weight
        self.verbose = verbose

    def fit(self, X: ArrayLike, y: Iterable) -> "BSOFeatureSelector":
        X_df = _as_dataframe(X)
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = np.array(list(X_df.columns), dtype=str)

        best_individual, best_fitness = bso_cv(
            X_df,
            pd.Series(y) if not isinstance(y, pd.Series) else y,
            population_size=self.population_size,
            max_iter=self.max_iter,
            cv=self.cv,
            random_state=self.random_state,
            penalty_weight=self.penalty_weight,
            verbose=self.verbose,
        )

        mask = np.array(best_individual, dtype=int).ravel()
        if mask.size != self.n_features_in_:
            raise ValueError(
                f"La máscara devuelta por bso_cv tiene longitud {mask.size} "
                f"y no coincide con n_features_in_={self.n_features_in_}."
            )

        # Fallback si el algoritmo selecciona 0 columnas: dejar pasar todo
        if int(mask.sum()) == 0:
            if self.verbose:
                print("[BSOFeatureSelector] Advertencia: selección vacía; se mantienen todas las columnas.")
            mask = np.ones_like(mask)

        self.selected_mask_ = mask.astype(bool)
        self.selected_idx_ = np.flatnonzero(self.selected_mask_)
        self.fitness_ = float(best_fitness)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        check_is_fitted(self, attributes=["selected_mask_", "selected_idx_"])
        X_df = _as_dataframe(X)

        if X_df.shape[1] != self.n_features_in_:
            raise ValueError(
                "El número de columnas de X en transform() no coincide con el visto en fit()."
            )

        X_sel = X_df.iloc[:, self.selected_idx_]
        return X_sel if isinstance(X, pd.DataFrame) else X_sel.to_numpy()

    # Métodos utilitarios estilo sklearn
    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        check_is_fitted(self, attributes=["selected_mask_", "selected_idx_"])
        return self.selected_idx_.tolist() if indices else self.selected_mask_.copy()

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        check_is_fitted(self, attributes=["selected_idx_", "feature_names_in_"])
        if input_features is None:
            input_features = self.feature_names_in_
        input_features = np.asarray(list(input_features), dtype=str)
        return input_features[self.selected_idx_]

    def __repr__(self) -> str:  # solo cosmético
        params = (
            f"population_size={self.population_size}, max_iter={self.max_iter}, cv={self.cv}, "
            f"penalty_weight={self.penalty_weight}, random_state={self.random_state}, verbose={self.verbose}"
        )
        fitted = hasattr(self, "selected_idx_")
        extra = (
            f", selected={len(getattr(self, 'selected_idx_', []))}/{getattr(self, 'n_features_in_', '?')}"
            if fitted else ""
        )
        return f"BSOFeatureSelector({params}{extra})"