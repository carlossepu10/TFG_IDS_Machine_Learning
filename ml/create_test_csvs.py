import pandas as pd
import os

# ==============================
# CONFIGURACIÓN
# ==============================

ruta_dataset = "ml/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
carpeta_salida = "ml/dataset/test"

os.makedirs(carpeta_salida, exist_ok=True)

# ==============================
# CARGAR DATASET GRANDE
# ==============================

print("Cargando dataset...")

df = pd.read_csv(ruta_dataset)
df.columns = df.columns.str.strip()

print("Dataset cargado correctamente")
print("Filas totales:", df.shape[0])
print("Columnas:", df.shape[1])

print("\nClases disponibles:")
print(df["Label"].value_counts())

# ==============================
# SEPARAR BENIGN Y DDoS
# ==============================

df_benigno = df[df["Label"] == "BENIGN"]
df_ddos = df[df["Label"] == "DDoS"]

# ==============================
# CREAR CSV BENIGNO
# ==============================

test_benigno = df_benigno.sample(n=50, random_state=42)

ruta_benigno = os.path.join(carpeta_salida, "test_benigno.csv")
test_benigno.to_csv(ruta_benigno, index=False)

print("\nCreado:", ruta_benigno)

# ==============================
# CREAR CSV DDoS
# ==============================

test_ddos = df_ddos.sample(n=50, random_state=42)

ruta_ddos = os.path.join(carpeta_salida, "test_ddos.csv")
test_ddos.to_csv(ruta_ddos, index=False)

print("Creado:", ruta_ddos)

# ==============================
# CREAR CSV MIXTO
# ==============================

test_mixto = pd.concat([
    df_benigno.sample(n=25, random_state=1),
    df_ddos.sample(n=25, random_state=1)
])

# Mezclar filas
test_mixto = test_mixto.sample(frac=1, random_state=42)

ruta_mixto = os.path.join(carpeta_salida, "test_mixto.csv")
test_mixto.to_csv(ruta_mixto, index=False)

print("Creado:", ruta_mixto)

print("\nArchivos de prueba creados correctamente.")