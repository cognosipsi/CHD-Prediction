from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.3, random_state=42, use_smote=False):
    """
    Divide el dataset en train/test y opcionalmente aplica SMOTE al conjunto de entrenamiento.
    Parámetros:
      - X, y: datos y etiquetas
      - test_size: proporción de test
      - random_state: semilla
      - use_smote: si True, aplica SMOTE en el train
      - smote_kwargs: dict opcional para parámetros de SMOTE
    Retorna:
      - X_train, X_test, y_train, y_test (X_train e y_train balanceados si use_smote)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test