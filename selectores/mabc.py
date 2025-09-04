# mabc.py
from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def _ensure_nonempty_mask(mask: np.ndarray) -> np.ndarray:
    """Evita máscaras vacías: si todas son 0, activa un índice aleatorio."""
    mask = np.asarray(mask, dtype=int).ravel()
    if mask.sum() == 0:
        j = np.random.randint(0, mask.shape[0])
        mask[j] = 1
    return mask


def _fitness_knn_holdout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    mask = _ensure_nonempty_mask(mask)
    Xtr = X_train[:, mask == 1]
    Xv = X_val[:, mask == 1]
    if Xtr.shape[1] == 0:
        return 0.0
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xv)
    return float(accuracy_score(y_val, preds))


def _fitness_knn_cv(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 5,
    cv_folds: int = 5,
    random_state: Optional[int] = 42,
) -> float:
    mask = _ensure_nonempty_mask(mask)
    Xsel = X[:, mask == 1]
    if Xsel.shape[1] == 0:
        return 0.0
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, Xsel, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def _roulette_prob(f: np.ndarray) -> np.ndarray:
    # Probabilidades robustas para selección proporcional al fitness
    f = np.asarray(f, dtype=float)
    f = f - f.min() + 1e-12
    return f / f.sum()


def m_abc_feature_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    evaluator_fn: Optional[Callable[[np.ndarray], float]] = None,
    *,
    # === hiperparámetros ABC ===
    pop_size: int = 20,         # como tu script: n_bees=20
    max_cycles: int = 30,       # como tu script: n_iter=30
    limit: int = 5,             # umbral para “scout”
    patience: int = 10,         # early-stop si no mejora el global en 'patience' ciclos
    random_state: Optional[int] = 42,
    verbose: bool = False,
    # === fitness (k-NN como en tu script) ===
    use_custom_evaluator: bool = False,
    knn_k: int = 5,
    cv_folds: Optional[int] = 5,   # <-- NUEVO: si >=2, usa CV estratificada (como tu .py)
    # si cv_folds=None o <2, usa hold-out (X_train/X_val)
) -> Tuple[np.ndarray, float]:
    """
    M-ABC binario para selección de características.

    Por defecto, el fitness es accuracy de k-NN con CV estratificada (cv=5),
    igual que en tu 'transformerMABC.py'. Alternativamente:
      - cv_folds < 2  -> usa hold-out (train->val)
      - use_custom_evaluator=True -> usa evaluator_fn(mask) (¡mucho más caro!)
    """
    rng = np.random.default_rng(random_state)
    n_features = X_train.shape[1]

    # Elegir evaluador local
    if use_custom_evaluator and evaluator_fn is not None:
        def local_eval(mask: np.ndarray) -> float:
            return float(evaluator_fn(mask))
    else:
        if (cv_folds is not None) and (cv_folds >= 2):
            X_all, y_all = X_train, y_train  # CV sobre train (más correcto y rápido)
            def local_eval(mask: np.ndarray) -> float:
                return _fitness_knn_cv(
                    X_all, y_all, mask, n_neighbors=knn_k,
                    cv_folds=int(cv_folds), random_state=random_state
                )
        else:
            def local_eval(mask: np.ndarray) -> float:
                return _fitness_knn_holdout(
                    X_train, y_train, X_val, y_val, mask, n_neighbors=knn_k
                )

    # Cache de evaluaciones por máscara
    fitness_cache: Dict[Tuple[int, ...], float] = {}

    def eval_cached(mask: np.ndarray) -> float:
        key = tuple(np.asarray(mask, dtype=int).tolist())
        if key in fitness_cache:
            return fitness_cache[key]
        val = local_eval(mask)
        fitness_cache[key] = float(val)
        return float(val)

    # Inicialización población (evita individuos vacíos)
    population = rng.integers(0, 2, size=(pop_size, n_features), dtype=int)
    for i in range(pop_size):
        if population[i].sum() == 0:
            population[i, int(rng.integers(0, n_features))] = 1

    fitness = np.array([eval_cached(ind) for ind in population], dtype=float)
    best_idx = int(np.argmax(fitness))
    best_sol = population[best_idx].copy()
    best_fit = float(fitness[best_idx])

    trial = np.zeros(pop_size, dtype=int)
    no_improve = 0

    for cycle in range(max_cycles):
        # --- Empleadas (flip de 1 bit con aceptación si mejora) ---
        for i in range(pop_size):
            cand = population[i].copy()
            j = int(rng.integers(0, n_features))
            cand[j] = 1 - cand[j]
            cand = _ensure_nonempty_mask(cand)
            f_cand = eval_cached(cand)
            if f_cand > fitness[i]:
                population[i] = cand
                fitness[i] = f_cand

        # --- Observadoras (ruleta) ---
        probs = _roulette_prob(fitness)
        for _ in range(pop_size):
            i = int(rng.choice(np.arange(pop_size), p=probs))
            cand = population[i].copy()
            j = int(rng.integers(0, n_features))
            cand[j] = 1 - cand[j]
            cand = _ensure_nonempty_mask(cand)
            f_cand = eval_cached(cand)
            if f_cand > fitness[i]:
                population[i] = cand
                fitness[i] = f_cand

        # --- Mejor global y contadores ---
        cur_idx = int(np.argmax(fitness))
        cur_best = float(fitness[cur_idx])
        if cur_best > best_fit:
            best_fit = cur_best
            best_sol = population[cur_idx].copy()
            no_improve = 0
        else:
            no_improve += 1

        for i in range(pop_size):
            if not np.array_equal(population[i], best_sol):
                trial[i] += 1
            else:
                trial[i] = 0

        # --- Scouts: re-inicializa si supera 'limit' ---
        for i in range(pop_size):
            if trial[i] >= limit:
                new_sol = rng.integers(0, 2, size=n_features, dtype=int)
                if new_sol.sum() == 0:
                    new_sol[int(rng.integers(0, n_features))] = 1
                f_new = eval_cached(new_sol)
                population[i] = new_sol
                fitness[i] = f_new
                trial[i] = 0

        if verbose:
            n_feat = int(best_sol.sum())
            print(f"[M-ABC] ciclo {cycle+1}/{max_cycles}  best={best_fit:.4f}  |S|={n_feat}")

        # Early stopping por paciencia
        if no_improve >= patience:
            if verbose:
                print(f"[M-ABC] early stop: sin mejora en {patience} ciclos.")
            break

    return best_sol, best_fit
