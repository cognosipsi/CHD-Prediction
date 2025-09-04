import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['row.names'], errors='ignore')  # Eliminar columna si existe
    return df
