from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, Union
import numpy as np
import warnings

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class MABCFeatureSelector(BaseEstimator, TransformerMixin):
    """
    M-ABC (Modified Artificial Bee Colony) para selección de características, como
    *transformer* compatible con `sklearn.Pipeline`, **manteniendo la lógica del algoritmo original**:

    - Cada solución es una máscara booleana (binaria) sobre las *n_features*.
    - **Objetivo**: maximizar *fitness* = *accuracy* de KNN (por defecto), sin
      penalización por cardinalidad.
    - **Vecindad**: *flip* de 1 bit (operador binario simple).
    - **Onlookers**: selección por ruleta proporcional al *fitness* (maximizando).
    - **Trials/Scouts**: contador respecto del mejor global; *scout* si supera `limit`.
    - **Parada temprana** por `patience` ciclos sin mejora del mejor global.
    - **Evaluación**: por defecto CV estratificada (si `cv_folds>=2`);
       si `cv_folds<2`, se usa *hold-out* interno (train/val) en `fit`.

    Parámetros
    ----------
    knn_k : int, default=5
        Número de vecinos para el KNN base cuando `estimator` es None (por defecto).
    estimator : sklearn estimator or None, default=None
        Estimador para evaluar subconjuntos. Si es None, se usa `KNeighborsClassifier(knn_k)`.
        La métrica usada es *accuracy* (para mantener la lógica del original).
    population_size : int, default=20
        Tamaño de la población/colonia (empleadas = observadoras = `population_size`).
    max_iter : int, default=30
        Número máximo de ciclos del algoritmo ABC.
    limit : int, default=5
        Umbral de `trial` para convertir una solución en *scout* (reinicializar).
    patience : int, default=10
        Corta si no mejora el mejor global en `patience` ciclos.
    cv_folds : int or None, default=5
        Si `cv_folds>=2`, se usa CV estratificada interna en `fit`. Si `<2` o `None`, se
        usa *hold-out* con `val_size`.
    val_size : float, default=0.2
        Proporción para el *hold-out* cuando `cv_folds<2`.
    random_state : int or RandomState or None, default=42
        Semilla para reproducibilidad.
    verbose : int, default=0
        Si >0, imprime progreso por ciclo.
    n_jobs : int or None, default=None
        Paralelismo para `cross_val_score`.

    Atributos (tras `fit`)
    ----------------------
    best_mask_ : np.ndarray[bool]
        Máscara óptima encontrada.
    best_fitness_ : float
        Fitness (accuracy) de la mejor máscara.
    n_selected_ : int
        Cantidad de características seleccionadas.
    n_features_in_ : int
        Número de características de entrada.
    feature_names_in_ : list[str] or None
        Si `X` era DataFrame, los nombres de columnas.
    """

    def __init__(
        self,
        *,
        knn_k: int = 5,
        estimator=None,
        population_size: int = 20,
        max_iter: int = 30,
        limit: int = 5,
        patience: int = 10,
        cv_folds: Optional[int] = 5,
        val_size: float = 0.2,
        random_state: Optional[Union[int, np.random.RandomState]] = 42,
        verbose: int = 0,
        n_jobs: Optional[int] = None,
    ):
        self.knn_k = knn_k
        self.estimator = estimator
        self.population_size = population_size
        self.max_iter = max_iter
        self.limit = limit
        self.patience = patience
        self.cv_folds = cv_folds
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    # ---------------- utilidades internas -----------------
    @staticmethod
    def _ensure_nonempty_mask(mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool).ravel()
        if not mask.any():
            j = np.random.randint(0, mask.size)
            mask[j] = True
        return mask

    @staticmethod
    def _roulette_prob(f: np.ndarray) -> np.ndarray:
        # proporción robusta (maximizando)
        f = np.asarray(f, dtype=float)
        f = f - f.min() + 1e-12
        s = f.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback uniforme
            return np.ones_like(f) / f.size
        return f / s

    def _make_estimator(self):
        if self.estimator is None:
            return KNeighborsClassifier(n_neighbors=self.knn_k)
        return clone(self.estimator)

    def _eval_mask_cv(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        mask = self._ensure_nonempty_mask(mask)
        Xsel = X[:, mask]
        if Xsel.shape[1] == 0:
            return 0.0
        est = self._make_estimator()
        cv = StratifiedKFold(
            n_splits=int(self.cv_folds), shuffle=True, random_state=self.random_state
        )
        scores = cross_val_score(est, Xsel, y, cv=cv, scoring="accuracy", n_jobs=self.n_jobs)
        return float(np.mean(scores))

    def _eval_mask_holdout(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        mask = self._ensure_nonempty_mask(mask)
        Xtr, Xv, ytr, yv = train_test_split(
            X, y, test_size=self.val_size, stratify=y, random_state=self.random_state
        )
        Xtr = Xtr[:, mask]
        Xv = Xv[:, mask]
        if Xtr.shape[1] == 0:
            return 0.0
        est = self._make_estimator()
        est.fit(Xtr, ytr)
        preds = est.predict(Xv)
        return float(accuracy_score(yv, preds))

    # ---------------- API sklearn -----------------
    def fit(self, X, y):
        is_df = hasattr(X, "iloc")
        self.feature_names_in_ = X.columns.tolist() if is_df else None

        X_arr, y_arr = check_X_y(
            X.values if is_df else X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float64,
            force_all_finite="allow-nan",
        )
        self.n_features_in_ = X_arr.shape[1]

        rng = check_random_state(self.random_state)
        n_features = self.n_features_in_
        pop = rng.rand(self.population_size, n_features) < 0.5
        # evitar individuos vacíos
        for i in range(self.population_size):
            if not pop[i].any():
                pop[i, rng.randint(n_features)] = True

        # evaluador (CV o hold-out interno)
        use_cv = (self.cv_folds is not None) and (self.cv_folds >= 2)
        eval_fn = (lambda m: self._eval_mask_cv(X_arr, y_arr, m)) if use_cv \
                  else (lambda m: self._eval_mask_holdout(X_arr, y_arr, m))

        # caché de evaluaciones
        cache: Dict[Tuple[int, ...], float] = {}

        def eval_cached(mask_bool: np.ndarray) -> float:
            key = tuple(np.flatnonzero(mask_bool))
            if key in cache:
                return cache[key]
            val = float(eval_fn(mask_bool))
            cache[key] = val
            return val

        fitness = np.array([eval_cached(ind) for ind in pop], dtype=float)
        best_idx = int(np.argmax(fitness))
        best_mask = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])

        trial = np.zeros(self.population_size, dtype=int)
        no_improve = 0

        for cycle in range(self.max_iter):
            # Empleadas: flip de 1 bit, aceptar si mejora
            for i in range(self.population_size):
                cand = pop[i].copy()
                j = rng.randint(n_features)
                cand[j] = ~cand[j]
                if not cand.any():
                    cand[rng.randint(n_features)] = True
                f_new = eval_cached(cand)
                if f_new > fitness[i]:
                    pop[i] = cand
                    fitness[i] = f_new

            # Observadoras: ruleta sobre fitness (max)
            probs = self._roulette_prob(fitness)
            for _ in range(self.population_size):
                i = rng.choice(self.population_size, p=probs)
                cand = pop[i].copy()
                j = rng.randint(n_features)
                cand[j] = ~cand[j]
                if not cand.any():
                    cand[rng.randint(n_features)] = True
                f_new = eval_cached(cand)
                if f_new > fitness[i]:
                    pop[i] = cand
                    fitness[i] = f_new

            # Mejor global y contadores de trial (respecto al mejor global)
            cur_idx = int(np.argmax(fitness))
            cur_best = float(fitness[cur_idx])
            if cur_best > best_fit:
                best_fit = cur_best
                best_mask = pop[cur_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            for i in range(self.population_size):
                if not np.array_equal(pop[i], best_mask):
                    trial[i] += 1
                else:
                    trial[i] = 0

            # Scouts: reinicializar si trial >= limit
            for i in range(self.population_size):
                if trial[i] >= self.limit:
                    new_sol = rng.rand(n_features) < 0.5
                    if not new_sol.any():
                        new_sol[rng.randint(n_features)] = True
                    pop[i] = new_sol
                    fitness[i] = eval_cached(new_sol)
                    trial[i] = 0

            if self.verbose:
                print(
                    f"[M-ABC] ciclo {cycle+1}/{self.max_iter} | best={best_fit:.4f} | k={int(best_mask.sum())}"
                )

            if self.patience is not None and no_improve >= self.patience:
                if self.verbose:
                    print(f"[M-ABC] early stop: sin mejora en {self.patience} ciclos.")
                break

        self.best_mask_ = best_mask.astype(bool)
        self.best_fitness_ = float(best_fit)
        self.n_selected_ = int(self.best_mask_.sum())
        return self

    def transform(self, X):
        if not hasattr(self, "best_mask_"):
            raise AttributeError("MABCFeatureSelector no está 'fit'. Llama a fit(X, y) primero.")
        is_df = hasattr(X, "iloc")
        if is_df:
            return X.loc[:, self.best_mask_]
        X_arr = check_array(
            X, accept_sparse=False, ensure_2d=True, dtype=np.float64, force_all_finite="allow-nan"
        )
        return X_arr[:, self.best_mask_]

    # helpers API sklearn
    def get_support(self, indices: bool = False):
        if not hasattr(self, "best_mask_"):
            raise AttributeError("MABCFeatureSelector no está 'fit'.")
        return np.flatnonzero(self.best_mask_) if indices else self.best_mask_.copy()

    def get_feature_names_out(self, input_features=None):
        mask = self.get_support()
        if input_features is None:
            if getattr(self, "feature_names_in_", None) is not None:
                input_features = self.feature_names_in_
            else:
                input_features = [f"x{i}" for i in range(self.n_features_in_)]
        input_features = np.asarray(input_features, dtype=object)
        return input_features[mask]


# --------- helper funcional (opcional) ----------

def mabc_fs(
    X, y,
    *,
    knn_k: int = 5,
    estimator=None,
    population_size: int = 20,
    max_iter: int = 30,
    limit: int = 5,
    patience: int = 10,
    cv_folds: Optional[int] = 5,
    val_size: float = 0.2,
    random_state: Optional[Union[int, np.random.RandomState]] = 42,
    verbose: int = 0,
    n_jobs: Optional[int] = None,
):
    """Atajo estilo función: devuelve `(mask, fitness)`.
    Mantiene la lógica del M-ABC original (maximizar accuracy con KNN por defecto).
    """
    sel = MABCFeatureSelector(
        knn_k=knn_k,
        estimator=estimator,
        population_size=population_size,
        max_iter=max_iter,
        limit=limit,
        patience=patience,
        cv_folds=cv_folds,
        val_size=val_size,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    ).fit(X, y)
    return sel.get_support(), sel.best_fitness_
