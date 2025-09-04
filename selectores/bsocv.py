import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def _fitness(X, y, individual, *, cv=5, penalty_weight=0.01):
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

def _crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1)-1)
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child

def bso_cv(X, y, *, population_size=20, max_iter=50, cv=5,
           random_state=None, penalty_weight=0.01, verbose=False):
    """
    BSO-CV simple con KNN y penalización por cantidad de características.
    Parámetros:
      - population_size: tamaño de población binaria
      - max_iter: número de iteraciones
      - cv: folds de cross-validation para KNN
      - random_state: seed (opcional)
      - penalty_weight: peso de la penalización por #features
    """
    if random_state is not None:
        np.random.seed(int(random_state))

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
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                child1 = _crossover(population[i], population[i+1])
                child2 = _crossover(population[i+1], population[i])
                new_population.extend([child1, child2])
            else:
                new_population.append(population[i])
        population = np.array(new_population)

        if verbose:
            print(f"[BSO-CV] iter={iteration+1}/{max_iter} fitness={best_fitness:.4f}")

    return best_individual, best_fitness
