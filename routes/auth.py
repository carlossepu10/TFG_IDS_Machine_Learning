from flask import Blueprint, request, session, flash, redirect, url_for, render_template, current_app, send_file
from models.database import get_connection
from werkzeug.utils import secure_filename

import bcrypt
import os
import pickle
import pandas as pd
import numpy as np
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


auth_bp = Blueprint("auth", __name__)


# =========================
# CONFIGURACIÓN DE MODELOS
# =========================

MODELOS_DISPONIBLES = {
    "random_forest": {
        "nombre": "Random Forest",
        "ruta": "ml/models/model_random_forest.pkl"
    },
    "decision_tree": {
        "nombre": "Decision Tree",
        "ruta": "ml/models/model_decision_tree.pkl"
    },
    "knn": {
        "nombre": "KNN",
        "ruta": "ml/models/model_knn.pkl"
    }
}


# =========================
# INICIO
# =========================

@auth_bp.route("/")
def inicio():
    return render_template("inicio.html")


# =========================
# FUNCIONES AUXILIARES ML
# =========================

def cargar_metricas_modelos():
    with open("ml/metrics_models.pkl", "rb") as f:
        metricas = pickle.load(f)

    return metricas


def cargar_modelo_y_metricas(modelo_seleccionado="random_forest"):
    metricas_modelos = cargar_metricas_modelos()

    if modelo_seleccionado not in MODELOS_DISPONIBLES:
        modelo_seleccionado = "random_forest"

    ruta_modelo = MODELOS_DISPONIBLES[modelo_seleccionado]["ruta"]

    with open(ruta_modelo, "rb") as f:
        modelo = pickle.load(f)

    metricas = metricas_modelos[modelo_seleccionado]

    return modelo, metricas


def obtener_nombre_modelo(modelo_seleccionado):
    if modelo_seleccionado in MODELOS_DISPONIBLES:
        return MODELOS_DISPONIBLES[modelo_seleccionado]["nombre"]

    return MODELOS_DISPONIBLES["random_forest"]["nombre"]


def analizar_csv_con_modelo(ruta_csv, modelo_seleccionado="random_forest"):
    modelo, metricas = cargar_modelo_y_metricas(modelo_seleccionado)

    columnas_modelo = metricas["features"]

    df = pd.read_csv(ruta_csv)
    df.columns = df.columns.str.strip()

    if "Label" in df.columns:
        df = df.drop("Label", axis=1)

    columnas_faltantes = [col for col in columnas_modelo if col not in df.columns]

    if columnas_faltantes:
        raise ValueError(
            "El CSV no contiene todas las columnas necesarias. "
            "Faltan columnas como: " + ", ".join(columnas_faltantes[:10])
        )

    df = df[columnas_modelo]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("Después de limpiar el CSV no quedan filas válidas para analizar.")

    filas_originales = len(df)

    if modelo_seleccionado == "knn" and len(df) > 5000:
        df = df.sample(n=5000, random_state=42)

    predicciones = modelo.predict(df)

    total = len(predicciones)
    benignos = int((predicciones == 0).sum())
    maliciosos = int((predicciones == 1).sum())

    porcentaje_benigno = round((benignos / total) * 100, 2)
    porcentaje_malicioso = round((maliciosos / total) * 100, 2)

    if maliciosos > benignos:
        clasificacion = "Malicioso"
    else:
        clasificacion = "Normal"

    return {
        "total": total,
        "filas_originales": filas_originales,
        "benignos": benignos,
        "maliciosos": maliciosos,
        "porcentaje_benigno": porcentaje_benigno,
        "porcentaje_malicioso": porcentaje_malicioso,
        "clasificacion": clasificacion,
        "probabilidad": porcentaje_malicioso / 100,
        "modelo": modelo_seleccionado,
        "modelo_nombre": obtener_nombre_modelo(modelo_seleccionado)
    }


def analizar_csv_con_modelo_y_generar_filtrados(ruta_csv, modelo_seleccionado="random_forest"):
    modelo, metricas = cargar_modelo_y_metricas(modelo_seleccionado)

    columnas_modelo = metricas["features"]

    df_original = pd.read_csv(ruta_csv)
    df_original.columns = df_original.columns.str.strip()

    df_modelo = df_original.copy()

    if "Label" in df_modelo.columns:
        df_modelo = df_modelo.drop("Label", axis=1)

    columnas_faltantes = [col for col in columnas_modelo if col not in df_modelo.columns]

    if columnas_faltantes:
        raise ValueError(
            "El CSV no contiene todas las columnas necesarias. "
            "Faltan columnas como: " + ", ".join(columnas_faltantes[:10])
        )

    df_modelo = df_modelo[columnas_modelo]

    df_modelo.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_modelo = df_modelo.apply(pd.to_numeric, errors="coerce")

    indices_validos = df_modelo.dropna().index

    df_modelo_limpio = df_modelo.loc[indices_validos]
    df_original_limpio = df_original.loc[indices_validos]

    if df_modelo_limpio.empty:
        raise ValueError("Después de limpiar el CSV no quedan filas válidas para analizar.")

    filas_originales = len(df_modelo_limpio)

    if modelo_seleccionado == "knn" and len(df_modelo_limpio) > 5000:
        df_modelo_limpio = df_modelo_limpio.sample(n=5000, random_state=42)
        df_original_limpio = df_original_limpio.loc[df_modelo_limpio.index]

    predicciones = modelo.predict(df_modelo_limpio)

    total = len(predicciones)
    benignos = int((predicciones == 0).sum())
    maliciosos = int((predicciones == 1).sum())

    porcentaje_benigno = round((benignos / total) * 100, 2)
    porcentaje_malicioso = round((maliciosos / total) * 100, 2)

    if maliciosos > benignos:
        clasificacion = "Malicioso"
    else:
        clasificacion = "Normal"

    carpeta_filtrados = os.path.join(
        current_app.config.get("UPLOAD_FOLDER", "uploads"),
        "filtrados"
    )
    os.makedirs(carpeta_filtrados, exist_ok=True)

    nombre_base = os.path.splitext(os.path.basename(ruta_csv))[0]

    nombre_benigno = f"{nombre_base}_benigno_{modelo_seleccionado}.csv"
    nombre_malicioso = f"{nombre_base}_malicioso_{modelo_seleccionado}.csv"

    ruta_csv_limpio = os.path.join(carpeta_filtrados, nombre_benigno)
    ruta_csv_malicioso = os.path.join(carpeta_filtrados, nombre_malicioso)

    df_benigno = df_original_limpio[predicciones == 0]
    df_malicioso = df_original_limpio[predicciones == 1]

    df_benigno.to_csv(ruta_csv_limpio, index=False)
    df_malicioso.to_csv(ruta_csv_malicioso, index=False)

    return {
        "total": total,
        "filas_originales": filas_originales,
        "benignos": benignos,
        "maliciosos": maliciosos,
        "porcentaje_benigno": porcentaje_benigno,
        "porcentaje_malicioso": porcentaje_malicioso,
        "clasificacion": clasificacion,
        "probabilidad": porcentaje_malicioso / 100,
        "modelo": modelo_seleccionado,
        "modelo_nombre": obtener_nombre_modelo(modelo_seleccionado),
        "ruta_csv_limpio": ruta_csv_limpio,
        "ruta_csv_malicioso": ruta_csv_malicioso
    }


def actualizar_resultado_guardado(analisis_id, ruta_csv, modelo_usado):
    resultado_ml = analizar_csv_con_modelo_y_generar_filtrados(ruta_csv, modelo_usado)

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE Resultado
        SET clasificacion = %s,
            probabilidad = %s,
            modelo = %s,
            total_muestras = %s,
            benignos = %s,
            maliciosos = %s,
            porcentaje_benigno = %s,
            porcentaje_malicioso = %s,
            filas_originales = %s,
            ruta_csv_limpio = %s,
            ruta_csv_malicioso = %s
        WHERE id_analisis = %s
    """, (
        resultado_ml["clasificacion"],
        float(resultado_ml["probabilidad"]),
        modelo_usado,
        resultado_ml["total"],
        resultado_ml["benignos"],
        resultado_ml["maliciosos"],
        resultado_ml["porcentaje_benigno"],
        resultado_ml["porcentaje_malicioso"],
        resultado_ml["filas_originales"],
        resultado_ml["ruta_csv_limpio"],
        resultado_ml["ruta_csv_malicioso"],
        analisis_id
    ))

    connection.commit()
    connection.close()

    return resultado_ml


# =========================
# REGISTER
# =========================

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        nombre = request.form.get("nombre", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not nombre or not email or not password:
            flash("Todos los campos son obligatorios", "danger")
            return redirect(url_for("auth.register"))

        connection = None

        try:
            connection = get_connection()
            cursor = connection.cursor()

            cursor.execute(
                "SELECT id FROM Usuario WHERE email = %s",
                (email,)
            )

            usuario_existente = cursor.fetchone()

            if usuario_existente:
                flash("Ya existe un usuario registrado con ese correo electrónico", "danger")
                connection.close()
                return redirect(url_for("auth.register"))

            hashed_password = bcrypt.hashpw(
                password.encode("utf-8"),
                bcrypt.gensalt()
            )

            cursor.execute(
                "INSERT INTO Usuario (nombre, email, password, rol, activo) VALUES (%s, %s, %s, %s, %s)",
                (nombre, email, hashed_password.decode("utf-8"), "USER", 1)
            )

            connection.commit()
            connection.close()

            flash("Usuario registrado correctamente. Ya puedes iniciar sesión", "success")
            return redirect(url_for("auth.login"))

        except Exception as e:
            if connection:
                connection.rollback()
                connection.close()

            flash(f"Error al registrar el usuario: {str(e)}", "danger")
            return redirect(url_for("auth.register"))

    return render_template("register.html")


# =========================
# LOGIN
# =========================

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Debes introducir email y contraseña", "danger")
            return redirect(url_for("auth.login"))

        connection = None

        try:
            connection = get_connection()
            cursor = connection.cursor()

            cursor.execute("SELECT * FROM Usuario WHERE email = %s", (email,))
            user = cursor.fetchone()

            connection.close()

            if user:
                stored_password = user[3]
                user_activo = user[5] if len(user) > 5 else 1

                if user_activo == 0:
                    flash("Tu cuenta se encuentra desactivada temporalmente. Contacta con el administrador.", "danger")
                    return redirect(url_for("auth.login"))

                if bcrypt.checkpw(password.encode("utf-8"), stored_password.encode("utf-8")):
                    session["user_id"] = user[0]
                    session["user_name"] = user[1]
                    session["user_role"] = user[4]

                    flash("Login correcto", "success")
                    return redirect(url_for("auth.analizar"))
                else:
                    flash("Contraseña incorrecta", "danger")
            else:
                flash("Usuario no encontrado", "danger")

        except Exception as e:
            if connection:
                connection.close()

            flash(f"Error al iniciar sesión: {str(e)}", "danger")

    return render_template("login.html")


# =========================
# LOGOUT
# =========================

@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Sesión cerrada correctamente", "info")
    return redirect(url_for("auth.login"))


# =========================
# PERFIL DE USUARIO
# =========================

@auth_bp.route("/perfil", methods=["GET", "POST"])
def perfil():
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, nombre, email, password, rol, activo
        FROM Usuario
        WHERE id = %s
    """, (session["user_id"],))

    usuario = cursor.fetchone()

    if not usuario:
        connection.close()
        flash("Usuario no encontrado", "danger")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        password_actual = request.form.get("password_actual", "")
        nueva_password = request.form.get("nueva_password", "")
        repetir_password = request.form.get("repetir_password", "")

        if not password_actual or not nueva_password or not repetir_password:
            connection.close()
            flash("Todos los campos de contraseña son obligatorios", "danger")
            return redirect(url_for("auth.perfil"))

        stored_password = usuario[3]

        if not bcrypt.checkpw(password_actual.encode("utf-8"), stored_password.encode("utf-8")):
            connection.close()
            flash("La contraseña actual no es correcta", "danger")
            return redirect(url_for("auth.perfil"))

        if nueva_password != repetir_password:
            connection.close()
            flash("La nueva contraseña y su repetición no coinciden", "danger")
            return redirect(url_for("auth.perfil"))

        if len(nueva_password) < 6:
            connection.close()
            flash("La nueva contraseña debe tener al menos 6 caracteres", "danger")
            return redirect(url_for("auth.perfil"))

        hashed_password = bcrypt.hashpw(
            nueva_password.encode("utf-8"),
            bcrypt.gensalt()
        )

        cursor.execute(
            "UPDATE Usuario SET password = %s WHERE id = %s",
            (hashed_password.decode("utf-8"), session["user_id"])
        )

        connection.commit()
        connection.close()

        flash("Contraseña actualizada correctamente", "success")
        return redirect(url_for("auth.perfil"))

    connection.close()

    return render_template("perfil.html", usuario=usuario)


# =========================
# ADMIN - GESTIÓN DE USUARIOS
# =========================

@auth_bp.route("/admin/usuarios")
def admin_usuarios():
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    if session.get("user_role") != "ADMIN":
        flash("Acceso restringido solo para administradores", "danger")
        return redirect(url_for("auth.analizar"))

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, nombre, email, rol, activo
        FROM Usuario
        ORDER BY id ASC
    """)

    usuarios = cursor.fetchall()

    connection.close()

    return render_template("admin_usuarios.html", usuarios=usuarios)


@auth_bp.route("/admin/usuarios/<int:user_id>/cambiar_estado", methods=["POST"])
def cambiar_estado_usuario(user_id):
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    if session.get("user_role") != "ADMIN":
        flash("Acceso restringido solo para administradores", "danger")
        return redirect(url_for("auth.analizar"))

    if user_id == session["user_id"]:
        flash("No puedes desactivar tu propia cuenta de administrador", "warning")
        return redirect(url_for("auth.admin_usuarios"))

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "SELECT activo FROM Usuario WHERE id = %s",
        (user_id,)
    )

    usuario = cursor.fetchone()

    if not usuario:
        connection.close()
        flash("Usuario no encontrado", "danger")
        return redirect(url_for("auth.admin_usuarios"))

    estado_actual = usuario[0]
    nuevo_estado = 0 if estado_actual == 1 else 1

    cursor.execute(
        "UPDATE Usuario SET activo = %s WHERE id = %s",
        (nuevo_estado, user_id)
    )

    connection.commit()
    connection.close()

    if nuevo_estado == 1:
        flash("Usuario activado correctamente", "success")
    else:
        flash("Usuario desactivado correctamente", "success")

    return redirect(url_for("auth.admin_usuarios"))


# =========================
# ANALIZAR CSV
# =========================

@auth_bp.route("/analizar", methods=["GET", "POST"])
def analizar():
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    connection = get_connection()
    cursor = connection.cursor()

    if request.method == "POST":

        if "archivo" not in request.files:
            flash("No se ha seleccionado ningún archivo", "danger")
            connection.close()
            return redirect(url_for("auth.analizar"))

        archivo = request.files["archivo"]
        modelo_seleccionado = request.form.get("modelo", "random_forest")

        if modelo_seleccionado not in MODELOS_DISPONIBLES:
            modelo_seleccionado = "random_forest"

        if archivo.filename == "":
            flash("No se seleccionó archivo", "danger")
            connection.close()
            return redirect(url_for("auth.analizar"))

        if not archivo.filename.lower().endswith(".csv"):
            flash("Solo se permiten archivos .csv", "danger")
            connection.close()
            return redirect(url_for("auth.analizar"))

        nombre_seguro = secure_filename(archivo.filename)

        upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        ruta = os.path.join(upload_folder, nombre_seguro)
        archivo.save(ruta)

        analisis_id = None

        try:
            cursor.execute(
                "INSERT INTO Analisis (estado, id_usuario, modelo) VALUES (%s, %s, %s)",
                ("Pendiente", session["user_id"], modelo_seleccionado)
            )

            analisis_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO ArchivoCSV (nombre, ruta, id_analisis) VALUES (%s, %s, %s)",
                (nombre_seguro, ruta, analisis_id)
            )

            connection.commit()

            resultado_ml = analizar_csv_con_modelo_y_generar_filtrados(ruta, modelo_seleccionado)

            cursor.execute(
                """
                INSERT INTO Resultado (
                    clasificacion,
                    probabilidad,
                    id_analisis,
                    modelo,
                    total_muestras,
                    benignos,
                    maliciosos,
                    porcentaje_benigno,
                    porcentaje_malicioso,
                    filas_originales,
                    ruta_csv_limpio,
                    ruta_csv_malicioso
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    resultado_ml["clasificacion"],
                    float(resultado_ml["probabilidad"]),
                    analisis_id,
                    modelo_seleccionado,
                    resultado_ml["total"],
                    resultado_ml["benignos"],
                    resultado_ml["maliciosos"],
                    resultado_ml["porcentaje_benigno"],
                    resultado_ml["porcentaje_malicioso"],
                    resultado_ml["filas_originales"],
                    resultado_ml["ruta_csv_limpio"],
                    resultado_ml["ruta_csv_malicioso"]
                )
            )

            cursor.execute(
                "UPDATE Analisis SET estado = %s WHERE id = %s",
                ("Finalizado", analisis_id)
            )

            connection.commit()

            mensaje_extra = ""

            if modelo_seleccionado == "knn" and resultado_ml["filas_originales"] > resultado_ml["total"]:
                mensaje_extra = (
                    f" KNN ha analizado una muestra de {resultado_ml['total']} filas "
                    f"de un total de {resultado_ml['filas_originales']} para evitar tiempos excesivos."
                )

            flash(
                f"Archivo analizado correctamente con {resultado_ml['modelo_nombre']}. "
                f"Resultado: {resultado_ml['clasificacion']}. "
                f"Tráfico malicioso detectado: {resultado_ml['porcentaje_malicioso']}%."
                f"{mensaje_extra}",
                "success"
            )

            return redirect(url_for("auth.ver_resultado", analisis_id=analisis_id))

        except Exception as e:
            try:
                connection.rollback()

                if analisis_id is not None:
                    cursor.execute(
                        "UPDATE Analisis SET estado = %s WHERE id = %s",
                        ("Error", analisis_id)
                    )
                    connection.commit()

            except Exception:
                pass

            flash(f"Error al analizar el archivo: {str(e)}", "danger")
            connection.close()
            return redirect(url_for("auth.analizar"))

    cursor.execute("""
        SELECT Analisis.id,
               ArchivoCSV.nombre,
               Analisis.estado,
               Resultado.clasificacion,
               Resultado.probabilidad,
               COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
               Analisis.fecha
        FROM Analisis
        JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
        LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
        WHERE Analisis.id_usuario = %s
        ORDER BY Analisis.id DESC
    """, (session["user_id"],))

    analisis = cursor.fetchall()

    connection.close()

    return render_template(
        "analizar.html",
        analisis=analisis,
        modelos=MODELOS_DISPONIBLES
    )


# =========================
# VER RESULTADO DETALLADO
# =========================

@auth_bp.route("/resultado/<int:analisis_id>")
def ver_resultado(analisis_id):
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    connection = get_connection()
    cursor = connection.cursor()

    if session.get("user_role") == "ADMIN":
        cursor.execute("""
            SELECT Analisis.id,
                   ArchivoCSV.nombre,
                   ArchivoCSV.ruta,
                   Analisis.estado,
                   Resultado.clasificacion,
                   Resultado.probabilidad,
                   COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
                   Analisis.fecha,
                   Resultado.total_muestras,
                   Resultado.benignos,
                   Resultado.maliciosos,
                   Resultado.porcentaje_benigno,
                   Resultado.porcentaje_malicioso,
                   Resultado.filas_originales,
                   Resultado.ruta_csv_limpio,
                   Resultado.ruta_csv_malicioso
            FROM Analisis
            JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
            LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
            WHERE Analisis.id = %s
        """, (analisis_id,))
    else:
        cursor.execute("""
            SELECT Analisis.id,
                   ArchivoCSV.nombre,
                   ArchivoCSV.ruta,
                   Analisis.estado,
                   Resultado.clasificacion,
                   Resultado.probabilidad,
                   COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
                   Analisis.fecha,
                   Resultado.total_muestras,
                   Resultado.benignos,
                   Resultado.maliciosos,
                   Resultado.porcentaje_benigno,
                   Resultado.porcentaje_malicioso,
                   Resultado.filas_originales,
                   Resultado.ruta_csv_limpio,
                   Resultado.ruta_csv_malicioso
            FROM Analisis
            JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
            LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
            WHERE Analisis.id = %s AND Analisis.id_usuario = %s
        """, (analisis_id, session["user_id"]))

    resultado = cursor.fetchone()
    connection.close()

    if not resultado:
        flash("Análisis no encontrado", "danger")
        return redirect(url_for("auth.dashboard_global" if session.get("user_role") == "ADMIN" else "auth.analizar"))

    if resultado[3] != "Finalizado":
        flash("El análisis todavía no está finalizado o terminó con error.", "warning")
        return redirect(url_for("auth.dashboard_global" if session.get("user_role") == "ADMIN" else "auth.analizar"))

    try:
        modelo_usado = resultado[6] if resultado[6] else "random_forest"

        if resultado[8] is None or resultado[8] == 0:
            resultado_ml = actualizar_resultado_guardado(
                analisis_id=resultado[0],
                ruta_csv=resultado[2],
                modelo_usado=modelo_usado
            )

            total = resultado_ml["total"]
            benignos = resultado_ml["benignos"]
            maliciosos = resultado_ml["maliciosos"]
            porcentaje_benigno = resultado_ml["porcentaje_benigno"]
            porcentaje_malicioso = resultado_ml["porcentaje_malicioso"]
            filas_originales = resultado_ml["filas_originales"]

        else:
            total = resultado[8]
            benignos = resultado[9]
            maliciosos = resultado[10]
            porcentaje_benigno = resultado[11]
            porcentaje_malicioso = resultado[12]
            filas_originales = resultado[13]

        if maliciosos > benignos:
            resultado_general = "TRÁFICO MALICIOSO"
            tipo_resultado = "danger"
            descripcion = "El modelo ha detectado una mayoría de muestras asociadas a tráfico DDoS."
        elif benignos > maliciosos:
            resultado_general = "TRÁFICO BENIGNO"
            tipo_resultado = "success"
            descripcion = "El modelo ha detectado una mayoría de muestras asociadas a tráfico normal."
        else:
            resultado_general = "RESULTADO MIXTO"
            tipo_resultado = "warning"
            descripcion = "El archivo contiene una proporción similar de tráfico benigno y tráfico DDoS."

        return render_template(
            "resultado.html",
            r=resultado,
            resultado_general=resultado_general,
            tipo_resultado=tipo_resultado,
            descripcion=descripcion,
            total=total,
            benignos=benignos,
            maliciosos=maliciosos,
            porcentaje_benigno=porcentaje_benigno,
            porcentaje_malicioso=porcentaje_malicioso,
            filas_originales=filas_originales,
            modelo_usado=modelo_usado,
            modelo_nombre=obtener_nombre_modelo(modelo_usado)
        )

    except Exception as e:
        flash(f"No se pudieron cargar los detalles del resultado: {str(e)}", "danger")
        return redirect(url_for("auth.dashboard_global" if session.get("user_role") == "ADMIN" else "auth.analizar"))


# =========================
# ELIMINAR ANÁLISIS
# =========================

@auth_bp.route("/eliminar_analisis/<int:analisis_id>", methods=["POST"])
def eliminar_analisis(analisis_id):
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    connection = get_connection()
    cursor = connection.cursor()

    try:
        if session.get("user_role") == "ADMIN":
            cursor.execute("""
                SELECT ArchivoCSV.ruta,
                       Resultado.ruta_csv_limpio,
                       Resultado.ruta_csv_malicioso
                FROM Analisis
                JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
                LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
                WHERE Analisis.id = %s
            """, (analisis_id,))
        else:
            cursor.execute("""
                SELECT ArchivoCSV.ruta,
                       Resultado.ruta_csv_limpio,
                       Resultado.ruta_csv_malicioso
                FROM Analisis
                JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
                LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
                WHERE Analisis.id = %s AND Analisis.id_usuario = %s
            """, (analisis_id, session["user_id"]))

        archivo = cursor.fetchone()

        if not archivo:
            flash("Análisis no encontrado o sin permisos para eliminarlo", "danger")
            connection.close()
            return redirect(url_for("auth.dashboard_global" if session.get("user_role") == "ADMIN" else "auth.analizar"))

        ruta_archivo = archivo[0]
        ruta_csv_limpio = archivo[1]
        ruta_csv_malicioso = archivo[2]

        cursor.execute("DELETE FROM Resultado WHERE id_analisis = %s", (analisis_id,))
        cursor.execute("DELETE FROM ArchivoCSV WHERE id_analisis = %s", (analisis_id,))
        cursor.execute("DELETE FROM Analisis WHERE id = %s", (analisis_id,))

        connection.commit()
        connection.close()

        if ruta_archivo and os.path.exists(ruta_archivo):
            os.remove(ruta_archivo)

        if ruta_csv_limpio and os.path.exists(ruta_csv_limpio):
            os.remove(ruta_csv_limpio)

        if ruta_csv_malicioso and os.path.exists(ruta_csv_malicioso):
            os.remove(ruta_csv_malicioso)

        flash("Análisis eliminado correctamente", "success")

        if session.get("user_role") == "ADMIN":
            return redirect(url_for("auth.dashboard_global"))

        return redirect(url_for("auth.analizar"))

    except Exception as e:
        connection.rollback()
        connection.close()
        flash(f"Error al eliminar el análisis: {str(e)}", "danger")
        return redirect(url_for("auth.dashboard_global" if session.get("user_role") == "ADMIN" else "auth.analizar"))


# =========================
# DESCARGAR PDF RESULTADO
# =========================

@auth_bp.route("/resultado/<int:analisis_id>/pdf")
def descargar_pdf(analisis_id):
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    connection = get_connection()
    cursor = connection.cursor()

    if session.get("user_role") == "ADMIN":
        cursor.execute("""
            SELECT Analisis.id,
                   ArchivoCSV.nombre,
                   ArchivoCSV.ruta,
                   Analisis.estado,
                   Resultado.clasificacion,
                   Resultado.probabilidad,
                   COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
                   Analisis.fecha,
                   Usuario.nombre,
                   Resultado.total_muestras,
                   Resultado.benignos,
                   Resultado.maliciosos,
                   Resultado.porcentaje_benigno,
                   Resultado.porcentaje_malicioso,
                   Resultado.filas_originales
            FROM Analisis
            JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
            LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
            JOIN Usuario ON Analisis.id_usuario = Usuario.id
            WHERE Analisis.id = %s
        """, (analisis_id,))
    else:
        cursor.execute("""
            SELECT Analisis.id,
                   ArchivoCSV.nombre,
                   ArchivoCSV.ruta,
                   Analisis.estado,
                   Resultado.clasificacion,
                   Resultado.probabilidad,
                   COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
                   Analisis.fecha,
                   Usuario.nombre,
                   Resultado.total_muestras,
                   Resultado.benignos,
                   Resultado.maliciosos,
                   Resultado.porcentaje_benigno,
                   Resultado.porcentaje_malicioso,
                   Resultado.filas_originales
            FROM Analisis
            JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
            LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
            JOIN Usuario ON Analisis.id_usuario = Usuario.id
            WHERE Analisis.id = %s AND Analisis.id_usuario = %s
        """, (analisis_id, session["user_id"]))

    resultado = cursor.fetchone()
    connection.close()

    if not resultado:
        flash("Análisis no encontrado", "danger")
        return redirect(url_for("auth.dashboard_global" if session.get("user_role") == "ADMIN" else "auth.analizar"))

    if resultado[3] != "Finalizado":
        flash("Solo se puede generar PDF de análisis finalizados.", "warning")
        return redirect(url_for("auth.dashboard_global" if session.get("user_role") == "ADMIN" else "auth.analizar"))

    try:
        modelo_usado = resultado[6] if resultado[6] else "random_forest"
        modelo_nombre = obtener_nombre_modelo(modelo_usado)

        if resultado[9] is None or resultado[9] == 0:
            resultado_ml = actualizar_resultado_guardado(
                analisis_id=resultado[0],
                ruta_csv=resultado[2],
                modelo_usado=modelo_usado
            )

            total = resultado_ml["total"]
            benignos = resultado_ml["benignos"]
            maliciosos = resultado_ml["maliciosos"]
            porcentaje_benigno = resultado_ml["porcentaje_benigno"]
            porcentaje_malicioso = resultado_ml["porcentaje_malicioso"]
            filas_originales = resultado_ml["filas_originales"]

        else:
            total = resultado[9]
            benignos = resultado[10]
            maliciosos = resultado[11]
            porcentaje_benigno = resultado[12]
            porcentaje_malicioso = resultado[13]
            filas_originales = resultado[14]

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)

        ancho, alto = A4
        y = alto - 2 * cm

        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(2 * cm, y, "Informe de analisis de trafico de red")

        y -= 1.2 * cm
        pdf.setFont("Helvetica", 11)
        pdf.drawString(2 * cm, y, "Sistema IDS basado en Machine Learning")

        y -= 1.4 * cm
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(2 * cm, y, "Datos generales del analisis")

        y -= 0.8 * cm
        pdf.setFont("Helvetica", 10)
        pdf.drawString(2 * cm, y, f"ID del analisis: {resultado[0]}")

        y -= 0.6 * cm
        pdf.drawString(2 * cm, y, f"Usuario: {resultado[8]}")

        y -= 0.6 * cm
        pdf.drawString(2 * cm, y, f"Archivo analizado: {resultado[1]}")

        y -= 0.6 * cm
        pdf.drawString(2 * cm, y, f"Fecha: {resultado[7]}")

        y -= 0.6 * cm
        pdf.drawString(2 * cm, y, f"Estado: {resultado[3]}")

        y -= 0.6 * cm
        pdf.drawString(2 * cm, y, f"Modelo utilizado: {modelo_nombre}")

        y -= 1.1 * cm
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(2 * cm, y, "Resultado")

        y -= 0.8 * cm
        pdf.setFont("Helvetica", 10)
        pdf.drawString(2 * cm, y, f"Clasificacion general: {resultado[4]}")

        y -= 0.6 * cm
        pdf.drawString(2 * cm, y, f"Probabilidad de trafico malicioso: {porcentaje_malicioso}%")

        y -= 0.6 * cm
        pdf.drawString(2 * cm, y, f"Total de muestras analizadas: {total}")

        if filas_originales and filas_originales > total:
            y -= 0.6 * cm
            pdf.drawString(2 * cm, y, f"Filas originales del CSV: {filas_originales}")

        y -= 0.6 * cm
        pdf.drawString(2 * cm, y, f"Muestras benignas: {benignos} ({porcentaje_benigno}%)")

        y -= 0.6 * cm
        pdf.drawString(2 * cm, y, f"Muestras DDoS: {maliciosos} ({porcentaje_malicioso}%)")

        y -= 1.1 * cm
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(2 * cm, y, "Interpretacion")

        y -= 0.8 * cm
        pdf.setFont("Helvetica", 10)

        texto = [
            "El archivo CSV ha sido procesado utilizando un modelo de aprendizaje automatico",
            "entrenado con el dataset CIC-IDS2017. Cada fila analizada representa una",
            "muestra de trafico de red. Las muestras clasificadas como BENIGN se consideran",
            "trafico normal, mientras que las muestras clasificadas como DDoS representan",
            "trafico potencialmente malicioso."
        ]

        for linea in texto:
            pdf.drawString(2 * cm, y, linea)
            y -= 0.5 * cm

        pdf.showPage()
        pdf.save()

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"informe_analisis_{analisis_id}.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        flash(f"No se pudo generar el PDF: {str(e)}", "danger")
        return redirect(url_for("auth.ver_resultado", analisis_id=analisis_id))


# =========================
# DESCARGAR CSV BENIGNO
# =========================

@auth_bp.route("/resultado/<int:analisis_id>/csv_benigno")
def descargar_csv_filtrado(analisis_id):
    return descargar_csv_por_tipo(analisis_id, tipo="benigno")


# =========================
# DESCARGAR CSV MALICIOSO
# =========================

@auth_bp.route("/resultado/<int:analisis_id>/csv_malicioso")
def descargar_csv_malicioso(analisis_id):
    return descargar_csv_por_tipo(analisis_id, tipo="malicioso")


def descargar_csv_por_tipo(analisis_id, tipo="benigno"):
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    connection = get_connection()
    cursor = connection.cursor()

    if session.get("user_role") == "ADMIN":
        cursor.execute("""
            SELECT Resultado.ruta_csv_limpio,
                   Resultado.ruta_csv_malicioso,
                   COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
                   ArchivoCSV.ruta,
                   Resultado.benignos,
                   Resultado.maliciosos
            FROM Analisis
            JOIN Resultado ON Analisis.id = Resultado.id_analisis
            JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
            WHERE Analisis.id = %s
        """, (analisis_id,))
    else:
        cursor.execute("""
            SELECT Resultado.ruta_csv_limpio,
                   Resultado.ruta_csv_malicioso,
                   COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
                   ArchivoCSV.ruta,
                   Resultado.benignos,
                   Resultado.maliciosos
            FROM Analisis
            JOIN Resultado ON Analisis.id = Resultado.id_analisis
            JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
            WHERE Analisis.id = %s AND Analisis.id_usuario = %s
        """, (analisis_id, session["user_id"]))

    resultado = cursor.fetchone()
    connection.close()

    if not resultado:
        flash("Análisis no encontrado", "danger")
        return redirect(url_for("auth.dashboard_global" if session.get("user_role") == "ADMIN" else "auth.analizar"))

    ruta_csv_benigno = resultado[0]
    ruta_csv_malicioso = resultado[1]
    modelo_usado = resultado[2] if resultado[2] else "random_forest"
    ruta_csv_original = resultado[3]
    benignos = resultado[4] if resultado[4] is not None else 0
    maliciosos = resultado[5] if resultado[5] is not None else 0

    if tipo == "benigno":
        if benignos == 0:
            flash(
                "No se puede generar un CSV benigno porque todas las muestras han sido clasificadas como maliciosas.",
                "warning"
            )
            return redirect(url_for("auth.ver_resultado", analisis_id=analisis_id))

        ruta_descarga = ruta_csv_benigno
        nombre_descarga = f"csv_benigno_{modelo_usado}_analisis_{analisis_id}.csv"

    else:
        if maliciosos == 0:
            flash(
                "No se puede generar un CSV malicioso porque no existen muestras clasificadas como maliciosas.",
                "warning"
            )
            return redirect(url_for("auth.ver_resultado", analisis_id=analisis_id))

        ruta_descarga = ruta_csv_malicioso
        nombre_descarga = f"csv_malicioso_{modelo_usado}_analisis_{analisis_id}.csv"

    if not ruta_descarga or not os.path.exists(ruta_descarga):
        try:
            actualizar_resultado_guardado(
                analisis_id=analisis_id,
                ruta_csv=ruta_csv_original,
                modelo_usado=modelo_usado
            )

            connection = get_connection()
            cursor = connection.cursor()
            cursor.execute("""
                SELECT ruta_csv_limpio,
                       ruta_csv_malicioso,
                       benignos,
                       maliciosos
                FROM Resultado
                WHERE id_analisis = %s
            """, (analisis_id,))
            nuevo_resultado = cursor.fetchone()
            connection.close()

            if nuevo_resultado:
                ruta_csv_benigno = nuevo_resultado[0]
                ruta_csv_malicioso = nuevo_resultado[1]
                benignos = nuevo_resultado[2] if nuevo_resultado[2] is not None else 0
                maliciosos = nuevo_resultado[3] if nuevo_resultado[3] is not None else 0

            if tipo == "benigno":
                if benignos == 0:
                    flash(
                        "No se puede generar un CSV benigno porque no existen muestras clasificadas como benignas.",
                        "warning"
                    )
                    return redirect(url_for("auth.ver_resultado", analisis_id=analisis_id))
                ruta_descarga = ruta_csv_benigno
            else:
                if maliciosos == 0:
                    flash(
                        "No se puede generar un CSV malicioso porque no existen muestras clasificadas como maliciosas.",
                        "warning"
                    )
                    return redirect(url_for("auth.ver_resultado", analisis_id=analisis_id))
                ruta_descarga = ruta_csv_malicioso

        except Exception as e:
            flash(f"No se pudo generar el CSV {tipo}: {str(e)}", "danger")
            return redirect(url_for("auth.dashboard_global" if session.get("user_role") == "ADMIN" else "auth.analizar"))

    if not ruta_descarga or not os.path.exists(ruta_descarga):
        flash(f"No existe CSV {tipo} para este análisis", "warning")
        return redirect(url_for("auth.ver_resultado", analisis_id=analisis_id))

    return send_file(
        ruta_descarga,
        as_attachment=True,
        download_name=nombre_descarga,
        mimetype="text/csv"
    )


# =========================
# COMPATIBILIDAD CON RUTA ANTIGUA
# =========================

@auth_bp.route("/resultado/<int:analisis_id>/csv_filtrado")
def descargar_csv_filtrado_antiguo(analisis_id):
    return descargar_csv_por_tipo(analisis_id, tipo="benigno")


# =========================
# DESCARGAR PDF ESTADÍSTICAS ADMIN
# =========================

@auth_bp.route("/dashboard_global/pdf")
def descargar_pdf_dashboard():
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    if session.get("user_role") != "ADMIN":
        flash("Acceso restringido solo para administradores", "danger")
        return redirect(url_for("auth.analizar"))

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM Analisis")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM Resultado
        WHERE clasificacion = 'Normal'
    """)
    normales = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM Resultado
        WHERE clasificacion = 'Malicioso'
    """)
    maliciosos = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM Analisis
        WHERE estado = 'Pendiente'
    """)
    pendientes = cursor.fetchone()[0]

    porcentaje_malicioso = 0
    if total > 0:
        porcentaje_malicioso = round((maliciosos / total) * 100, 2)

    cursor.execute("""
        SELECT Usuario.id,
               Usuario.nombre,
               Usuario.email,
               COUNT(Analisis.id) AS total_usuario,
               SUM(CASE WHEN Resultado.clasificacion = 'Normal' THEN 1 ELSE 0 END) AS normales_usuario,
               SUM(CASE WHEN Resultado.clasificacion = 'Malicioso' THEN 1 ELSE 0 END) AS maliciosos_usuario,
               SUM(CASE WHEN Analisis.estado = 'Pendiente' THEN 1 ELSE 0 END) AS pendientes_usuario
        FROM Usuario
        LEFT JOIN Analisis ON Usuario.id = Analisis.id_usuario
        LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
        GROUP BY Usuario.id, Usuario.nombre, Usuario.email
        ORDER BY total_usuario DESC
    """)
    usuarios_estadisticas = cursor.fetchall()

    cursor.execute("""
        SELECT COALESCE(modelo, 'random_forest') AS modelo,
               COUNT(*) AS total_modelo,
               SUM(CASE WHEN clasificacion = 'Normal' THEN 1 ELSE 0 END) AS normales_modelo,
               SUM(CASE WHEN clasificacion = 'Malicioso' THEN 1 ELSE 0 END) AS maliciosos_modelo,
               AVG(probabilidad) AS probabilidad_media
        FROM Resultado
        GROUP BY COALESCE(modelo, 'random_forest')
        ORDER BY modelo
    """)
    estadisticas_modelos = cursor.fetchall()

    connection.close()

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    ancho, alto = A4
    y = alto - 2 * cm

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(2 * cm, y, "Informe estadistico global del sistema IDS")

    y -= 1.1 * cm
    pdf.setFont("Helvetica", 11)
    pdf.drawString(2 * cm, y, "Sistema de deteccion de trafico malicioso basado en Machine Learning")

    y -= 1.3 * cm
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(2 * cm, y, "Resumen global")

    y -= 0.8 * cm
    pdf.setFont("Helvetica", 10)
    pdf.drawString(2 * cm, y, f"Total de analisis: {total}")

    y -= 0.55 * cm
    pdf.drawString(2 * cm, y, f"Analisis normales: {normales}")

    y -= 0.55 * cm
    pdf.drawString(2 * cm, y, f"Analisis maliciosos: {maliciosos}")

    y -= 0.55 * cm
    pdf.drawString(2 * cm, y, f"Analisis pendientes: {pendientes}")

    y -= 0.55 * cm
    pdf.drawString(2 * cm, y, f"Porcentaje malicioso global: {porcentaje_malicioso}%")

    y -= 1.2 * cm
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(2 * cm, y, "Estadisticas por usuario")

    y -= 0.8 * cm
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(2 * cm, y, "Usuario")
    pdf.drawString(6 * cm, y, "Email")
    pdf.drawString(11 * cm, y, "Total")
    pdf.drawString(13 * cm, y, "Normal")
    pdf.drawString(15 * cm, y, "Malic.")

    y -= 0.4 * cm
    pdf.setFont("Helvetica", 8)

    for u in usuarios_estadisticas:
        if y < 2.5 * cm:
            pdf.showPage()
            y = alto - 2 * cm
            pdf.setFont("Helvetica-Bold", 13)
            pdf.drawString(2 * cm, y, "Estadisticas por usuario")
            y -= 0.8 * cm
            pdf.setFont("Helvetica", 8)

        total_usuario = u[3] if u[3] is not None else 0
        normales_usuario = u[4] if u[4] is not None else 0
        maliciosos_usuario = u[5] if u[5] is not None else 0

        pdf.drawString(2 * cm, y, str(u[1])[:20])
        pdf.drawString(6 * cm, y, str(u[2])[:28])
        pdf.drawString(11 * cm, y, str(total_usuario))
        pdf.drawString(13 * cm, y, str(normales_usuario))
        pdf.drawString(15 * cm, y, str(maliciosos_usuario))

        y -= 0.45 * cm

    y -= 0.8 * cm

    if y < 5 * cm:
        pdf.showPage()
        y = alto - 2 * cm

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(2 * cm, y, "Estadisticas por modelo")

    y -= 0.8 * cm
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(2 * cm, y, "Modelo")
    pdf.drawString(7 * cm, y, "Total")
    pdf.drawString(9 * cm, y, "Normal")
    pdf.drawString(11 * cm, y, "Malicioso")
    pdf.drawString(14 * cm, y, "Prob. media")

    y -= 0.45 * cm
    pdf.setFont("Helvetica", 8)

    for e in estadisticas_modelos:
        modelo_clave = e[0] if e[0] else "random_forest"
        nombre_modelo = obtener_nombre_modelo(modelo_clave)

        total_modelo = e[1] if e[1] is not None else 0
        normales_modelo = e[2] if e[2] is not None else 0
        maliciosos_modelo = e[3] if e[3] is not None else 0
        probabilidad_media = e[4] if e[4] is not None else 0

        pdf.drawString(2 * cm, y, nombre_modelo)
        pdf.drawString(7 * cm, y, str(total_modelo))
        pdf.drawString(9 * cm, y, str(normales_modelo))
        pdf.drawString(11 * cm, y, str(maliciosos_modelo))
        pdf.drawString(14 * cm, y, f"{round(probabilidad_media * 100, 2)}%")

        y -= 0.5 * cm

    y -= 1 * cm
    pdf.setFont("Helvetica", 9)
    pdf.drawString(2 * cm, y, "Informe generado automaticamente desde el dashboard global del administrador.")

    pdf.showPage()
    pdf.save()

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="informe_estadisticas_globales.pdf",
        mimetype="application/pdf"
    )


# =========================
# MÉTRICAS DEL MODELO
# =========================

@auth_bp.route("/metricas")
def ver_metricas():
    try:
        metricas_modelos = cargar_metricas_modelos()

        return render_template(
            "metricas.html",
            m=metricas_modelos.get("random_forest"),
            metricas_modelos=metricas_modelos
        )

    except Exception as e:
        flash(f"No se pudieron cargar las métricas del modelo: {str(e)}", "danger")
        return redirect(url_for("auth.analizar"))


# =========================
# DASHBOARD PERSONAL
# =========================

@auth_bp.route("/dashboard_personal")
def dashboard_personal():
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    return cargar_dashboard_usuario(session["user_id"], modo_admin=False)


@auth_bp.route("/admin/dashboard_usuario/<int:user_id>")
def dashboard_usuario_admin(user_id):
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    if session.get("user_role") != "ADMIN":
        flash("Acceso restringido solo para administradores", "danger")
        return redirect(url_for("auth.analizar"))

    return cargar_dashboard_usuario(user_id, modo_admin=True)


def cargar_dashboard_usuario(user_id, modo_admin=False):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, nombre, email, rol, activo
        FROM Usuario
        WHERE id = %s
    """, (user_id,))

    usuario = cursor.fetchone()

    if not usuario:
        connection.close()
        flash("Usuario no encontrado", "danger")
        return redirect(url_for("auth.dashboard_global"))

    cursor.execute("""
        SELECT COUNT(*)
        FROM Analisis
        WHERE id_usuario = %s
    """, (user_id,))
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM Analisis
        JOIN Resultado ON Analisis.id = Resultado.id_analisis
        WHERE Analisis.id_usuario = %s
        AND Resultado.clasificacion = 'Normal'
    """, (user_id,))
    normales = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM Analisis
        JOIN Resultado ON Analisis.id = Resultado.id_analisis
        WHERE Analisis.id_usuario = %s
        AND Resultado.clasificacion = 'Malicioso'
    """, (user_id,))
    maliciosos = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM Analisis
        WHERE id_usuario = %s
        AND estado = 'Pendiente'
    """, (user_id,))
    pendientes = cursor.fetchone()[0]

    porcentaje_malicioso = 0
    if total > 0:
        porcentaje_malicioso = round((maliciosos / total) * 100, 2)

    cursor.execute("""
        SELECT COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
               COUNT(*) AS total_modelo
        FROM Analisis
        LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
        WHERE Analisis.id_usuario = %s
        GROUP BY COALESCE(Resultado.modelo, Analisis.modelo)
        ORDER BY total_modelo DESC
    """, (user_id,))
    uso_modelos = cursor.fetchall()

    cursor.execute("""
        SELECT Analisis.id,
               ArchivoCSV.nombre,
               Analisis.estado,
               Resultado.clasificacion,
               Resultado.probabilidad,
               COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
               Analisis.fecha
        FROM Analisis
        JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
        LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
        WHERE Analisis.id_usuario = %s
        ORDER BY Analisis.id DESC
        LIMIT 10
    """, (user_id,))
    ultimos_analisis = cursor.fetchall()

    fecha_desde = request.args.get("fecha_desde", "").strip()
    fecha_hasta = request.args.get("fecha_hasta", "").strip()
    modelo_filtro = request.args.get("modelo", "").strip()
    clasificacion_filtro = request.args.get("clasificacion", "").strip()
    estado_filtro = request.args.get("estado", "").strip()
    ver_todos = request.args.get("ver_todos", "").strip()

    filtros = {
        "fecha_desde": fecha_desde,
        "fecha_hasta": fecha_hasta,
        "modelo": modelo_filtro,
        "clasificacion": clasificacion_filtro,
        "estado": estado_filtro,
        "ver_todos": ver_todos
    }

    condiciones = ["Analisis.id_usuario = %s"]
    parametros = [user_id]

    if fecha_desde:
        condiciones.append("DATE(Analisis.fecha) >= %s")
        parametros.append(fecha_desde)

    if fecha_hasta:
        condiciones.append("DATE(Analisis.fecha) <= %s")
        parametros.append(fecha_hasta)

    if modelo_filtro:
        condiciones.append("COALESCE(Resultado.modelo, Analisis.modelo) = %s")
        parametros.append(modelo_filtro)

    if clasificacion_filtro:
        condiciones.append("Resultado.clasificacion = %s")
        parametros.append(clasificacion_filtro)

    if estado_filtro:
        condiciones.append("Analisis.estado = %s")
        parametros.append(estado_filtro)

    hay_busqueda = (
        fecha_desde or
        fecha_hasta or
        modelo_filtro or
        clasificacion_filtro or
        estado_filtro or
        ver_todos == "1"
    )

    analisis_filtrados = []

    if hay_busqueda:
        where_sql = " AND ".join(condiciones)

        consulta_filtrada = f"""
            SELECT Analisis.id,
                   ArchivoCSV.nombre,
                   Analisis.estado,
                   Resultado.clasificacion,
                   Resultado.probabilidad,
                   COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
                   Analisis.fecha
            FROM Analisis
            JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
            LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
            WHERE {where_sql}
            ORDER BY Analisis.id DESC
        """

        cursor.execute(consulta_filtrada, tuple(parametros))
        analisis_filtrados = cursor.fetchall()

    connection.close()

    return render_template(
        "dashboard_personal.html",
        usuario=usuario,
        total=total,
        normales=normales,
        maliciosos=maliciosos,
        pendientes=pendientes,
        porcentaje=porcentaje_malicioso,
        uso_modelos=uso_modelos,
        ultimos_analisis=ultimos_analisis,
        analisis_filtrados=analisis_filtrados,
        filtros=filtros,
        hay_busqueda=hay_busqueda,
        modelos=MODELOS_DISPONIBLES,
        modo_admin=modo_admin
    )


# =========================
# DASHBOARD GLOBAL ADMIN
# =========================

@auth_bp.route("/dashboard_global")
def dashboard_global():
    if "user_id" not in session:
        flash("Acceso no autorizado", "danger")
        return redirect(url_for("auth.login"))

    if session.get("user_role") != "ADMIN":
        flash("Acceso restringido solo para administradores", "danger")
        return redirect(url_for("auth.analizar"))

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM Analisis")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) 
        FROM Resultado 
        WHERE clasificacion = 'Normal'
    """)
    normales = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) 
        FROM Resultado 
        WHERE clasificacion = 'Malicioso'
    """)
    maliciosos = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM Analisis
        WHERE estado = 'Pendiente'
    """)
    pendientes = cursor.fetchone()[0]

    porcentaje_malicioso = 0
    if total > 0:
        porcentaje_malicioso = round((maliciosos / total) * 100, 2)

    cursor.execute("""
        SELECT Analisis.id,
               Usuario.nombre,
               ArchivoCSV.nombre,
               Analisis.estado,
               Resultado.clasificacion,
               Resultado.probabilidad,
               COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
               Analisis.fecha
        FROM Analisis
        JOIN Usuario ON Analisis.id_usuario = Usuario.id
        JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
        LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
        ORDER BY Analisis.id DESC
        LIMIT 10
    """)
    ultimos_analisis = cursor.fetchall()

    cursor.execute("""
        SELECT Usuario.id,
               Usuario.nombre,
               Usuario.email,
               COUNT(Analisis.id) AS total_usuario,
               SUM(CASE WHEN Resultado.clasificacion = 'Normal' THEN 1 ELSE 0 END) AS normales_usuario,
               SUM(CASE WHEN Resultado.clasificacion = 'Malicioso' THEN 1 ELSE 0 END) AS maliciosos_usuario,
               SUM(CASE WHEN Analisis.estado = 'Pendiente' THEN 1 ELSE 0 END) AS pendientes_usuario
        FROM Usuario
        LEFT JOIN Analisis ON Usuario.id = Analisis.id_usuario
        LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
        GROUP BY Usuario.id, Usuario.nombre, Usuario.email
        ORDER BY total_usuario DESC
    """)
    usuarios_estadisticas = cursor.fetchall()

    cursor.execute("""
        SELECT id, nombre, email
        FROM Usuario
        ORDER BY nombre ASC
    """)
    usuarios_lista = cursor.fetchall()

    cursor.execute("""
        SELECT COALESCE(modelo, 'random_forest') AS modelo,
               COUNT(*) AS total_modelo
        FROM Resultado
        GROUP BY COALESCE(modelo, 'random_forest')
        ORDER BY total_modelo DESC
    """)
    uso_modelos = cursor.fetchall()

    cursor.execute("""
        SELECT COALESCE(modelo, 'random_forest') AS modelo,
               COUNT(*) AS total_modelo,
               SUM(CASE WHEN clasificacion = 'Normal' THEN 1 ELSE 0 END) AS normales_modelo,
               SUM(CASE WHEN clasificacion = 'Malicioso' THEN 1 ELSE 0 END) AS maliciosos_modelo,
               AVG(probabilidad) AS probabilidad_media
        FROM Resultado
        GROUP BY COALESCE(modelo, 'random_forest')
        ORDER BY modelo
    """)
    estadisticas_modelos = cursor.fetchall()

    usuario_filtro = request.args.get("usuario", "").strip()
    fecha_desde = request.args.get("fecha_desde", "").strip()
    fecha_hasta = request.args.get("fecha_hasta", "").strip()
    modelo_filtro = request.args.get("modelo", "").strip()
    clasificacion_filtro = request.args.get("clasificacion", "").strip()
    estado_filtro = request.args.get("estado", "").strip()
    ver_todos = request.args.get("ver_todos", "").strip()

    filtros_global = {
        "usuario": usuario_filtro,
        "fecha_desde": fecha_desde,
        "fecha_hasta": fecha_hasta,
        "modelo": modelo_filtro,
        "clasificacion": clasificacion_filtro,
        "estado": estado_filtro,
        "ver_todos": ver_todos
    }

    condiciones = ["1 = 1"]
    parametros = []

    if usuario_filtro:
        condiciones.append("Usuario.id = %s")
        parametros.append(usuario_filtro)

    if fecha_desde:
        condiciones.append("DATE(Analisis.fecha) >= %s")
        parametros.append(fecha_desde)

    if fecha_hasta:
        condiciones.append("DATE(Analisis.fecha) <= %s")
        parametros.append(fecha_hasta)

    if modelo_filtro:
        condiciones.append("COALESCE(Resultado.modelo, Analisis.modelo) = %s")
        parametros.append(modelo_filtro)

    if clasificacion_filtro:
        condiciones.append("Resultado.clasificacion = %s")
        parametros.append(clasificacion_filtro)

    if estado_filtro:
        condiciones.append("Analisis.estado = %s")
        parametros.append(estado_filtro)

    hay_busqueda_global = (
        usuario_filtro or
        fecha_desde or
        fecha_hasta or
        modelo_filtro or
        clasificacion_filtro or
        estado_filtro or
        ver_todos == "1"
    )

    analisis_filtrados_global = []

    if hay_busqueda_global:
        where_sql = " AND ".join(condiciones)

        consulta_global = f"""
            SELECT Analisis.id,
                   Usuario.nombre,
                   Usuario.email,
                   ArchivoCSV.nombre,
                   Analisis.estado,
                   Resultado.clasificacion,
                   Resultado.probabilidad,
                   COALESCE(Resultado.modelo, Analisis.modelo) AS modelo,
                   Analisis.fecha
            FROM Analisis
            JOIN Usuario ON Analisis.id_usuario = Usuario.id
            JOIN ArchivoCSV ON Analisis.id = ArchivoCSV.id_analisis
            LEFT JOIN Resultado ON Analisis.id = Resultado.id_analisis
            WHERE {where_sql}
            ORDER BY Analisis.id DESC
        """

        cursor.execute(consulta_global, tuple(parametros))
        analisis_filtrados_global = cursor.fetchall()

    connection.close()

    return render_template(
        "dashboard_global.html",
        total=total,
        normales=normales,
        maliciosos=maliciosos,
        pendientes=pendientes,
        porcentaje=porcentaje_malicioso,
        ultimos_analisis=ultimos_analisis,
        usuarios_estadisticas=usuarios_estadisticas,
        usuarios_lista=usuarios_lista,
        uso_modelos=uso_modelos,
        estadisticas_modelos=estadisticas_modelos,
        filtros_global=filtros_global,
        hay_busqueda_global=hay_busqueda_global,
        analisis_filtrados_global=analisis_filtrados_global,
        modelos=MODELOS_DISPONIBLES
    )