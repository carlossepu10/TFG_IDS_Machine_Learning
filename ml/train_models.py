import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================
# 1. Cargar dataset
# ==============================

ruta_dataset = "ml/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Cargando dataset...")
df = pd.read_csv(ruta_dataset)

# Limpiar espacios en nombres de columnas
df.columns = df.columns.str.strip()

print("Dataset cargado correctamente")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])


# ==============================
# 2. Limpieza básica
# ==============================

print("\nLimpiando datos...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# BENIGN = 0, DDoS = 1
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

print("Datos limpios")
print("Filas después de limpieza:", df.shape[0])

print("\nDistribución de clases:")
print(df["Label"].value_counts())


# ==============================
# 3. Separar X e y
# ==============================

X = df.drop("Label", axis=1)
y = df["Label"]

X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

y = y.loc[X.index]


# ==============================
# 4. Separar entrenamiento y prueba
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDatos separados:")
print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)


# ==============================
# 5. Definir modelos
# ==============================

modelos = {
    "random_forest": {
        "nombre": "Random Forest",
        "modelo": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    },
    "decision_tree": {
        "nombre": "Decision Tree",
        "modelo": DecisionTreeClassifier(
            random_state=42
        )
    },
    "knn": {
        "nombre": "KNN",
        "modelo": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )
    }
}


# ==============================
# 6. Entrenar y evaluar modelos
# ==============================

os.makedirs("ml/models", exist_ok=True)

metricas_modelos = {}

for clave, datos in modelos.items():
    nombre = datos["nombre"]
    modelo = datos["modelo"]

    print("\n================================")
    print(f"Entrenando modelo: {nombre}")
    print("================================")

    modelo.fit(X_train, y_train)

    print(f"Modelo {nombre} entrenado correctamente")

    print(f"Evaluando modelo {nombre}...")

    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    matriz = confusion_matrix(y_test, y_pred)
    reporte = classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "DDoS"]
    )

    print("\nAccuracy:", accuracy)
    print("\nMatriz de confusión:")
    print(matriz)
    print("\nReporte de clasificación:")
    print(reporte)

    ruta_modelo = f"ml/models/model_{clave}.pkl"

    with open(ruta_modelo, "wb") as f:
        pickle.dump(modelo, f)

    metricas_modelos[clave] = {
        "nombre": nombre,
        "accuracy": accuracy,
        "confusion_matrix": matriz,
        "classification_report": reporte,
        "features": list(X.columns),
        "ruta_modelo": ruta_modelo
    }

    print(f"Modelo guardado en {ruta_modelo}")


# ==============================
# 7. Guardar métricas globales
# ==============================

with open("ml/metrics_models.pkl", "wb") as f:
    pickle.dump(metricas_modelos, f)

# Para mantener compatibilidad con lo que ya tenías,
# guardamos también Random Forest como model.pkl y metrics.pkl
with open("ml/model.pkl", "wb") as f:
    pickle.dump(modelos["random_forest"]["modelo"], f)

with open("ml/metrics.pkl", "wb") as f:
    pickle.dump(metricas_modelos["random_forest"], f)

print("\n================================")
print("Entrenamiento finalizado")
print("================================")
print("Modelos guardados en ml/models/")
print("Métricas comparativas guardadas en ml/metrics_models.pkl")
print("Random Forest guardado también en ml/model.pkl por compatibilidad")