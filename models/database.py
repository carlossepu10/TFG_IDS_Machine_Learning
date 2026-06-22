import pymysql
from config import DB_CONFIG

def get_connection():
    return pymysql.connect(**DB_CONFIG)

def init_db():
    connection = get_connection()
    cursor = connection.cursor()

    # Tabla Usuario
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Usuario (
            id INT AUTO_INCREMENT PRIMARY KEY,
            nombre VARCHAR(100) NOT NULL,
            email VARCHAR(150) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            rol VARCHAR(20) NOT NULL
        )
    """)

    # Tabla Analisis
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Analisis (
            id INT AUTO_INCREMENT PRIMARY KEY,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            estado VARCHAR(50),
            id_usuario INT,
            FOREIGN KEY (id_usuario) REFERENCES Usuario(id)
                ON DELETE CASCADE
        )
    """)

    # Tabla Resultado
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Resultado (
            id INT AUTO_INCREMENT PRIMARY KEY,
            clasificacion VARCHAR(50),
            probabilidad FLOAT,
            id_analisis INT UNIQUE,
            FOREIGN KEY (id_analisis) REFERENCES Analisis(id)
                ON DELETE CASCADE
        )
    """)

    # Tabla ArchivoCSV
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ArchivoCSV (
            id INT AUTO_INCREMENT PRIMARY KEY,
            nombre VARCHAR(150),
            ruta VARCHAR(255),
            fechaSubida TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            id_analisis INT UNIQUE,
            FOREIGN KEY (id_analisis) REFERENCES Analisis(id)
                ON DELETE CASCADE
        )
    """)

    connection.commit()
    connection.close()