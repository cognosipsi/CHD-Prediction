from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def _fitness(
    X,
    y,
    individual: np.ndarray,
    estimator: Optional[ClassifierMixin] = None,
    cv: int = 5,
    penalty_weight: float = 0.1,   # igual que el monolítico
    random_state: Optional[int] = None,
) -> float:
    """
    Fitness alineado con el script monolítico:
      mean_accuracy_cv * (1 - penalty_weight * (n_sel / n_total))
    """
    idx = [i for i, bit in enumerate(individual) if bit == 1]
    if not idx:
        return float("-inf")

    n_total = X.shape[1]
    X_sel = X[:, idx] if not hasattr(X, "iloc") else X.iloc[:, idx]

    if estimator is None:
        estimator = LogisticRegression(penalty="l1", solver="liblinear", max_iter=500)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    for tr, va in skf.split(X_sel, y):
        Xt = X_sel[tr] if isinstance(X_sel, np.ndarray) else X_sel.iloc[tr]
        Xv = X_sel[va] if isinstance(X_sel, np.ndarray) else X_sel.iloc[va]
        estimator.fit(Xt, y[tr])
        y_pred = estimator.predict(Xv)
        scores.append(accuracy_score(y[va], y_pred))

    mean_acc = float(np.mean(scores))
    frac = len(idx) / float(n_total)
    return mean_acc * (1.0 - penalty_weight * frac)

def woa_feature_selection(
    X,
    y,
    population_size: int = 20,
    max_iter: int = 50,
    estimator: Optional[ClassifierMixin] = None,
    cv: int = 5,
    penalty_weight: float = 0.1,     # por defecto 0.1 como en el monolítico
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, float]:
    """
    WOA binario (población inicial binaria, como tu script original).
    """
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]

    # Población INICIAL BINARIA, no continua
    agents = rng.integers(0, 2, size=(population_size, n_features)).astype(np.int8)

    def _binarize(A: np.ndarray) -> np.ndarray:
        # Umbral 0.5 directamente (inputs continuos tras update)
        return (A >= 0.5).astype(np.int8)

    # Evaluación inicial
    best_score = -np.inf
    best_agent = agents[0].copy()
    for i in range(population_size):
        sc = _fitness(X, y, agents[i], estimator, cv, penalty_weight, random_state)
        if sc > best_score:
            best_score, best_agent = sc, agents[i].copy()

    b = 1.0
    for t in range(max_iter):
        a = 2.0 - 2.0 * (t / (max_iter - 1 if max_iter > 1 else 1))
        for i in range(population_size):
            Xi = agents[i].astype(float)

            r1 = rng.random(n_features); r2 = rng.random(n_features)
            A = 2*a*r1 - a
            C = 2*r2
            p = rng.random()

            best_cont = best_agent.astype(float)
            if p < 0.5:
                if np.mean(np.abs(A)) < 1.0:
                    D = np.abs(C*best_cont - Xi)
                    Xnew = best_cont - A*D
                else:
                    j = int(rng.integers(0, population_size))
                    Xrand = agents[j].astype(float)
                    D = np.abs(C*Xrand - Xi)
                    Xnew = Xrand - A*D
            else:
                l = rng.uniform(-1.0, 1.0, size=n_features)
                Dp = np.abs(best_cont - Xi)
                Xnew = Dp * np.exp(b*l) * np.cos(2*np.pi*l) + best_cont

            # De continuo a binario vía sigmoide+umbral 0.5
            S = 1.0 / (1.0 + np.exp(-Xnew))
            agents[i] = (S >= 0.5).astype(np.int8)

            sc = _fitness(X, y, agents[i], estimator, cv, penalty_weight, random_state)
            old_sc = _fitness(X, y, (Xi >= 0.5).astype(np.int8), estimator, cv, penalty_weight, random_state)
            if sc >= old_sc:
                if sc > best_score:
                    best_score, best_agent = sc, agents[i].copy()

    best_mask = best_agent.astype(np.int8)
    return best_mask, float(best_score)

def selected_columns(X, mask: np.ndarray) -> List[str]:
    idx = np.where(mask == 1)[0]
    if hasattr(X, "columns"):
        return list(X.columns[idx])
    return [str(i) for i in idx]

__all__ = ["woa_feature_selection", "selected_columns"]
