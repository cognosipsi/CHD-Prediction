from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class WOAFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        population_size: int = 30,
        max_iter: int = 50,
        estimator: Optional[BaseEstimator] = None,
        cv: int = 5,
        penalty_weight: float = 0.1,
        random_state: Optional[int] = 42
    ):
        self.population_size = population_size
        self.max_iter = max_iter
        self.estimator = estimator
        self.cv = cv
        self.penalty_weight = penalty_weight
        self.random_state = random_state

    def fit(self, X, y):
        """
        Ajusta el selector de características WOA utilizando el conjunto de datos.
        """
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]

        # Población continua en [0,1]
        agents = rng.random((self.population_size, n_features))

        def _sigmoid(Z: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-Z))

        def _binarize(Z: np.ndarray) -> np.ndarray:
            """
            Si el vector está en R, aplica sigmoide y umbral.
            Si ya está en [0,1], el sigmoide es opcional, pero lo dejamos para consistencia.
            """
            S = _sigmoid(Z)
            return (S >= 0.5).astype(np.int8)

        # Evaluación inicial (binarizando para el fitness)
        best_score = -np.inf
        best_agent = agents[0].copy()
        for i in range(self.population_size):
            sc = self._fitness(X, y, _binarize(agents[i]))
            if sc > best_score:
                best_score, best_agent = sc, agents[i].copy()

        # Parámetro del espiral
        b = 1.0

        for t in range(self.max_iter):
            # a decrece linealmente de 2 a 0
            a = 2.0 - 2.0 * (t / (self.max_iter - 1 if self.max_iter > 1 else 1))
            for i in range(self.population_size):
                Xi = agents[i]

                r1 = rng.random(n_features)
                r2 = rng.random(n_features)
                A = 2 * a * r1 - a
                C = 2 * r2
                p = rng.random()

                best_cont = best_agent  # continuo

                if p < 0.5:
                    if np.mean(np.abs(A)) < 1.0:
                        # cazar al mejor
                        D = np.abs(C * best_cont - Xi)
                        Xnew = best_cont - A * D
                    else:
                        # explorar con un agente aleatorio
                        j = int(rng.integers(0, self.population_size))
                        Xrand = agents[j]
                        D = np.abs(C * Xrand - Xi)
                        Xnew = Xrand - A * D
                else:
                    # espiral hacia el mejor
                    l = rng.uniform(-1.0, 1.0, size=n_features)
                    Dp = np.abs(best_cont - Xi)
                    Xnew = Dp * np.exp(b * l) * np.cos(2 * np.pi * l) + best_cont

                # Mantenemos representación continua y evaluamos con binarización
                new_agent = _sigmoid(Xnew)  # (0,1)
                sc_new = self._fitness(X, y, _binarize(new_agent))
                sc_old = self._fitness(X, y, _binarize(Xi))

                if sc_new >= sc_old:
                    agents[i] = new_agent
                    if sc_new > best_score:
                        best_score, best_agent = sc_new, new_agent.copy()

        self.best_mask_ = _binarize(best_agent).astype(np.int8)
        self.best_score_ = float(best_score)

        return self

    def transform(self, X):
        """
        Devuelve el conjunto de datos con solo las características seleccionadas.
        """
        return X[:, self.best_mask_ == 1]

    def _fitness(
        self,
        X,
        y,
        individual: np.ndarray,
        estimator: Optional[BaseEstimator] = None,
    ) -> float:
        """
        Fitness alineado con tu versión monolítica:
          mean_accuracy_cv * (1 - penalty_weight * (n_sel / n_total))
        """
        # Índices de columnas seleccionadas (bits = 1)
        idx = [i for i, bit in enumerate(individual) if int(bit) == 1]
        if not idx:
            return float("-inf")

        n_total = X.shape[1]
        X_sel = X[:, idx] if not hasattr(X, "iloc") else X.iloc[:, idx]

        # Clasificador base para la evaluación en CV (mismo espíritu del monolítico)
        if estimator is None:
            estimator = self.estimator if self.estimator else LogisticRegression(penalty="l1", solver="liblinear", max_iter=500)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True)
        scores = []
        for tr, va in skf.split(X_sel, y):
            Xt = X_sel[tr] if isinstance(X_sel, np.ndarray) else X_sel.iloc[tr]
            Xv = X_sel[va] if isinstance(X_sel, np.ndarray) else X_sel.iloc[va]
            estimator.fit(Xt, y[tr])
            y_pred = estimator.predict(Xv)
            scores.append(accuracy_score(y[va], y_pred))

        mean_acc = float(np.mean(scores))
        frac = len(idx) / float(n_total)
        return mean_acc * (1.0 - self.penalty_weight * frac)

    def selected_columns(self, X) -> List[str]:
        """
        Devuelve las columnas seleccionadas del conjunto de datos.
        """
        idx = np.where(self.best_mask_ == 1)[0]
        if hasattr(X, "columns"):
            return list(X.columns[idx])
        return [str(i) for i in idx]

