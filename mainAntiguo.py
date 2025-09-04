from pipelines.knn_pipeline import knn_pipeline
from pipelines.mlp_pipeline import mlp_pipeline
from preprocesamiento.lectura_datos import load_data

# Usar la ruta relativa para el archivo CSV
file_path = 'SAHeart.csv'
df = load_data(file_path)

PIPELINES = {
    "knn": knn_pipeline,
    "mlp": mlp_pipeline,
}

def _prompt_model() -> str:
    opciones = ", ".join(PIPELINES.keys())
    print("Modelos disponibles:", opciones)
    print("También puedes ingresar 1=knn, 2=mlp")
    while True:
        choice = input("¿Qué modelo de predicción deseas evaluar? ").strip().lower()
        # Atajos numéricos
        if choice == "1":
            choice = "knn"
        elif choice == "2":
            choice = "mlp"

        if choice in PIPELINES:
            return choice
        else:
            print(f"Opción no válida: '{choice}'. Intenta de nuevo ({opciones} / 1 / 2).")

if __name__ == "__main__":
    modelo = _prompt_model()
    print(f"Ejecutando pipeline: {modelo.upper()} con archivo {file_path}...")
    PIPELINES[modelo](file_path)