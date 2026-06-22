import pandas as pd

ruta = "ml/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(ruta)

# Limpiar espacios en los nombres de columnas
df.columns = df.columns.str.strip()

print("Número de filas:", df.shape[0])
print("Número de columnas:", df.shape[1])

print("\nPrimeras columnas:")
print(df.columns[:10])

print("\nÚltimas columnas:")
print(df.columns[-10:])

print("\nColumnas completas:")
for col in df.columns:
    print(col)

print("\nValores únicos de la columna Label:")
print(df["Label"].unique())

print("\nConteo de clases:")
print(df["Label"].value_counts())