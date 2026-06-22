from flask import Blueprint, render_template
import pickle
import os

ml_bp = Blueprint("ml", __name__)


@ml_bp.route("/metricas")
def metricas():
    ruta_metricas = os.path.join("ml", "metrics.pkl")

    if not os.path.exists(ruta_metricas):
        return render_template(
            "metricas.html",
            error="Todavía no se han generado métricas. Entrena el modelo primero.",
            accuracy=None,
            matriz=None,
            reporte=None
        )

    with open(ruta_metricas, "rb") as f:
        metricas_modelo = pickle.load(f)

    accuracy = metricas_modelo.get("accuracy")
    matriz = metricas_modelo.get("confusion_matrix")
    reporte = metricas_modelo.get("classification_report")

    return render_template(
        "metricas.html",
        error=None,
        accuracy=accuracy,
        matriz=matriz,
        reporte=reporte
    )