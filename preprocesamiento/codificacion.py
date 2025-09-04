from sklearn.preprocessing import LabelEncoder

def encode_features(df, encoding_method):
    """
    Codifica las características categóricas usando uno de los dos métodos: manual o LabelEncoder.
    
    Parámetros:
    - df: DataFrame con las características a codificar.
    - encoding_method: Método de codificación ("manual" o "labelencoder").
    
    Retorna:
    - df: DataFrame con las características codificadas.
    """
    
    if encoding_method == "manual":
        # Codificación manual para 'famhist'
        df['famhist'] = df['famhist'].map({'Absent': 0, 'Present': 1})
    
    elif encoding_method == "labelencoder":
        # Codificación usando LabelEncoder
        label_encoder = LabelEncoder()
        df['famhist'] = label_encoder.fit_transform(df['famhist'])
    
    else:
        raise ValueError("encoding_method debe ser 'manual' o 'labelencoder'")

    return df
