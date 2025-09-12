from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(X, y, sampling_strategy='auto', random_state=42):
    """
    Aplica la técnica SMOTE para balancear las clases del dataset.
    
    Parámetros:
    - X: Características (DataFrame o ndarray).
    - y: Etiquetas (vector o Serie).
    - sampling_strategy: Define la cantidad de sobre-muestreo a realizar (por defecto, igual al número de la clase mayoritaria).
    - random_state: Semilla para asegurar reproducibilidad.

    Retorna:
    - X_resampled: Características después de aplicar SMOTE.
    - y_resampled: Etiquetas después de aplicar SMOTE.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
