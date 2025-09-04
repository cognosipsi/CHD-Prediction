from sklearn.preprocessing import MinMaxScaler, StandardScaler

def _make_scaler(scaler_type: str):
    if scaler_type == "minmax":
        return MinMaxScaler()
    elif scaler_type == "standard":
        return StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

def scale_features(X, scaler_type):
    """
    Igual que antes: fit_transform sobre TODO X (útil para heurísticas como WOA).
    No usar para el modelo final (produce fuga).
    """
    scaler = _make_scaler(scaler_type)
    return scaler.fit_transform(X)

def scale_train_test(X_train, X_test, scaler_type):
    """
    Escalado correcto para el modelo final: fit SOLO en train, transform en train y test.
    """
    scaler = _make_scaler(scaler_type)
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    return Xtr, Xte, scaler
