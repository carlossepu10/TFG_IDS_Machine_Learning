from flask import Flask, session
from models.database import get_connection, init_db
from routes.auth import auth_bp
from routes.ml import ml_bp

app = Flask(__name__)
app.secret_key = "clave_super_secreta"
app.config["UPLOAD_FOLDER"] = "uploads"

app.register_blueprint(auth_bp)
app.register_blueprint(ml_bp)


@app.route("/")
def home():
    return "Backend funcionando correctamente"


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return "Acceso no autorizado ❌"

    return f"Bienvenido {session['user_name']} al dashboard"


if __name__ == "__main__":
    init_db()
    app.run(debug=True)