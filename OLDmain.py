import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# === Preprocesamiento ===
from utils.evaluacion import cargar_y_preprocesar  # Función que debes crear en preprocesamiento.py

# === Selectores de características ===
from selectores.bsocv import seleccionar_caracteristicas_bsocv
from selectores.woa import seleccionar_caracteristicas_woa
from selectores.mabc import seleccionar_caracteristicas_boruta

# === Modelos predictivos ===
from predictores.xgb import entrenar_xgb
from predictores.mlp import entrenar_mlp
from predictores.knn import entrenar_knn
from predictores.transformer import entrenar_transformer

# 1. Preprocesamiento
X, y = cargar_y_preprocesar("SAHeart.csv")

# 2. Selección de características

selecciones = {
    "Boruta": seleccionar_caracteristicas_boruta(X, y),
    "BSOCV": seleccionar_caracteristicas_bsocv(X, y),
    "WOA": seleccionar_caracteristicas_woa(X, y),
}

# === 3. Definir modelos ===

modelos = {
    "KNN": entrenar_knn,
    "MLP": entrenar_mlp,
    "XGB": entrenar_xgb,
    "Transformer": entrenar_transformer,
}


# === 4. Evaluar combinaciones de selector + modelo ===
resultados = []

for nombre_sel, columnas in selecciones.items():
    X_sel = X[columnas]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)
    for nombre_mod, funcion_modelo in modelos.items():
        print(f"\n[{nombre_sel} + {nombre_mod}]")
        y_pred, y_prob = funcion_modelo(X_train, X_test, y_train, y_test)
        resultados.append({
            "Selector": nombre_sel,
            "Modelo": nombre_mod,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob)
        })

# === Mostrar resultados ===
df_resultados = pd.DataFrame(resultados)
print(df_resultados)
