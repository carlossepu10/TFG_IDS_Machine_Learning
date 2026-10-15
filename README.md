TFG: Sistema IDS basado en Machine Learning


Sistema de detección de intrusiones (IDS) que utiliza Machine Learning para clasificar el tráfico de red como benigno o malicioso a partir de archivos CSV. Incluye una aplicación web para que los usuarios suban sus capturas, consulten los resultados y gestionen su historial.

Trabajo de Fin de Grado de Ingeniería Telemática (Universitat de València).


Qué hace
Análisis de tráfico: el usuario sube un CSV con tráfico de red y el sistema indica el porcentaje de tráfico malicioso y benigno.
Exportación filtrada: permite generar un archivo que contenga solo el tráfico benigno o solo el malicioso.
Cuentas de usuario: registro e inicio de sesión.
Historial: cada usuario puede consultar sus análisis anteriores.
Panel de administración: el administrador accede a estadísticas y dashboards con la actividad de todos los usuarios.
Dataset

El modelo se entrena con CIC-IDS2017, un dataset de referencia en ciberseguridad creado por el Canadian Institute for Cybersecurity (CIC) que contiene tráfico de red benigno y diferentes tipos de ataques.

Tecnologías
Área	Tecnologías
Backend	Python, Flask
Machine Learning	scikit-learn, pandas, NumPy, joblib
Base de datos	MySQL (PyMySQL)
Frontend	HTML5, CSS, Bootstrap, JavaScript
Estructura del proyecto
├── app.py              # Punto de entrada de la aplicación Flask
├── config.py           # Configuración de la conexión a la base de datos
├── requirements.txt    # Dependencias de Python
├── ml/                 # Código de Machine Learning
├── models/             # Acceso y gestión de la base de datos
├── routes/             # Rutas de la aplicación (autenticación y análisis ML)
├── templates/          # Plantillas HTML
└── static/img/         # Recursos estáticos
Instalación y ejecución

Requisitos: Python 3.11 o superior y un servidor MySQL en local.

Clona el repositorio:
bash
   git clone https://github.com/carlossepu10/TFG_IDS_Machine_Learning.git
   cd TFG_IDS_Machine_Learning
Crea y activa un entorno virtual:
bash
   python -m venv venv
   source venv/bin/activate      # En Windows: venv\Scripts\activate
Instala las dependencias:
bash
   pip install -r requirements.txt
Crea la base de datos trafico_db en MySQL y ajusta en config.py el host, usuario, contraseña y puerto con los de tu entorno:
python
   DB_CONFIG = {
       "host": "localhost",
       "user": "TU_USUARIO",
       "password": "TU_CONTRASEÑA",
       "database": "trafico_db",
       "port": 3306
   }
Arranca la aplicación (las tablas se inicializan al arrancar):
bash
   python app.py
Abre http://127.0.0.1:5000 en el navegador.
Limitaciones

Es un proyecto académico pensado para ejecutarse en local, no para un despliegue en producción.

Autor

Carlos Sepúlveda Martínez, Ingeniero Telemático

GitHub: carlossepu10
LinkedIn: carlos-sepúlveda-martínez
