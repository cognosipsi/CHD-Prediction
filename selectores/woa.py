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
    penalty_weight: float = 0.1,
    random_state: Optional[int] = None,
) -> float:
    """
    Igual a tu script monolítico:
      mean_accuracy_cv  -  penalty_weight * (#features seleccionadas)
    Penaliza por **cantidad**, no por fracción.
    """
    idx = [i for i, bit in enumerate(individual) if bit == 1]
    if not idx:
        return float("-inf")

    X_sel = X[:, idx] if not hasattr(X, "iloc") else X.iloc[:, idx]

    if estimator is None:
        estimator = LogisticRegression(penalty="l1", solver="liblinear", max_iter=500)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    for tr, va in skf.split(X_sel, y):
        estimator.fit(X_sel[tr] if isinstance(X_sel, np.ndarray) else X_sel.iloc[tr], y[tr])
        y_pred = estimator.predict(X_sel[va] if isinstance(X_sel, np.ndarray) else X_sel.iloc[va])
        scores.append(accuracy_score(y[va], y_pred))

    return float(np.mean(scores)) - penalty_weight * len(idx)

def woa_feature_selection(
    X,
    y,
    population_size: int = 30,   # = mlpWOA.py
    max_iter: int = 50,          # = mlpWOA.py
    estimator: Optional[ClassifierMixin] = None,
    cv: int = 5,
    penalty_weight: float = 0.01,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, float]:
    """
    WOA binario (mismo espíritu que tu monolítico: posiciones continuas -> sigmoide -> binarización).
    """
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    # Población continua en [0,1], luego binarizada por sigmoide
    agents = rng.random((population_size, n_features))
    def _binarize(A): 
        S = 1.0 / (1.0 + np.exp(-A))
        return (S >= 0.5).astype(np.int8)

    best_score = -np.inf
    best_agent = agents[0].copy()

    # Evaluación inicial
    bin_agents = _binarize(agents)
    for i in range(population_size):
        sc = _fitness(X, y, bin_agents[i], estimator, cv, penalty_weight, random_state)
        if sc > best_score:
            best_score, best_agent = sc, agents[i].copy()

    b = 1.0
    for t in range(max_iter):
        a = 2.0 - 2.0 * (t / (max_iter - 1 if max_iter > 1 else 1))
        for i in range(population_size):
            Xi = agents[i].copy()
            r1 = rng.random(n_features); r2 = rng.random(n_features)
            A = 2*a*r1 - a
            C = 2*r2
            p = rng.random()

            best_bin = _binarize(best_agent)
            if p < 0.5:
                if np.mean(np.abs(A)) < 1.0:
                    D = np.abs(C*best_agent - Xi)
                    Xnew = best_agent - A*D
                else:
                    j = int(rng.integers(0, population_size))
                    Xrand = agents[j]
                    D = np.abs(C*Xrand - Xi)
                    Xnew = Xrand - A*D
            else:
                l = rng.uniform(-1.0, 1.0, size=n_features)
                Dp = np.abs(best_agent - Xi)
                Xnew = Dp * np.exp(b*l) * np.cos(2*np.pi*l) + best_agent

            agents[i] = Xnew
            bin_i = _binarize(agents[i])
            sc = _fitness(X, y, bin_i, estimator, cv, penalty_weight, random_state)
            old_sc = _fitness(X, y, _binarize(Xi), estimator, cv, penalty_weight, random_state)
            if sc >= old_sc:
                if sc > best_score:
                    best_score, best_agent = sc, agents[i].copy()

    best_mask = _binarize(best_agent).astype(np.int8)
    return best_mask, float(best_score)

def selected_columns(X, mask: np.ndarray) -> List[str]:
    idx = np.where(mask == 1)[0]
    if hasattr(X, "columns"):
        return list(X.columns[idx])
    return [str(i) for i in idx]

__all__ = ["woa_feature_selection", "selected_columns"]
