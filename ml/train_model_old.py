import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================
# 1. Cargar dataset
# ==============================

ruta_dataset = "ml/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Cargando dataset...")
df = pd.read_csv(ruta_dataset)

# Limpiar espacios en los nombres de columnas
df.columns = df.columns.str.strip()

print("Dataset cargado correctamente")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])


# ==============================
# 2. Limpieza básica
# ==============================

print("\nLimpiando datos...")

# Sustituir infinitos por NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Eliminar filas con valores nulos
df.dropna(inplace=True)

# Convertir etiquetas a formato numérico
# BENIGN = 0
# Ataque = 1
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

print("Datos limpios")
print("Filas después de limpieza:", df.shape[0])

print("\nDistribución de clases:")
print(df["Label"].value_counts())


# ==============================
# 3. Separar variables X e y
# ==============================

X = df.drop("Label", axis=1)
y = df["Label"]

# Asegurarnos de que todas las columnas sean numéricas
X = X.apply(pd.to_numeric, errors="coerce")

# Sustituir infinitos por NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Eliminar filas con NaN en X
X.dropna(inplace=True)

# Ajustar y a los mismos índices que X
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
# 5. Entrenar modelo Random Forest
# ==============================

print("\nEntrenando modelo Random Forest...")

modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

modelo.fit(X_train, y_train)

print("Modelo entrenado correctamente")


# ==============================
# 6. Evaluar modelo
# ==============================

print("\nEvaluando modelo...")

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


# ==============================
# 7. Guardar modelo y métricas
# ==============================

os.makedirs("ml", exist_ok=True)

with open("ml/model.pkl", "wb") as f:
    pickle.dump(modelo, f)

metricas = {
    "accuracy": accuracy,
    "confusion_matrix": matriz,
    "classification_report": reporte,
    "features": list(X.columns)
}

with open("ml/metrics.pkl", "wb") as f:
    pickle.dump(metricas, f)

print("\nModelo guardado en ml/model.pkl")
print("Métricas guardadas en ml/metrics.pkl")