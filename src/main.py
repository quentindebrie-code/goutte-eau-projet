"""
main.py — API FastAPI — Projet Goutte d'Eau MVP
Bloc 2 — Compétence C13

Endpoints :
    GET /predict?date=YYYY-MM-DD  → estimation du risque de pluie
    GET /health                   → statut de l'API et du modèle
    GET /metrics                  → métriques d'évaluation du modèle
    GET /docs                     → documentation Swagger (auto-générée)

Lancement :
    uvicorn main:app --reload
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Chemins ─────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
EVAL_PATH  = BASE_DIR / "model" / "evaluation.json"
DB_PATH    = BASE_DIR / "data" / "weather.db"

FEATURES = [
    "temp_max", "temp_min", "humidity_avg",
    "pressure_avg", "wind_avg", "cloud_cover",
    "temp_range", "month", "day_of_year",
]


# ─── Application FastAPI ──────────────────────────────────────────────────────

app = FastAPI(
    title="Projet Goutte d'Eau — API de prévision du risque de pluie",
    description=(
        "MVP développé dans le cadre du Mastère Management de la Transformation Digitale en IA — "
        "Institut Léonard de Vinci.\n\n"
        "Cette API retourne une estimation du risque de pluie pour Paris (75) "
        "en fonction d'une date donnée, à partir d'un modèle Random Forest entraîné "
        "sur les données historiques SYNOP / Open-Meteo.\n\n"
        "**Limitations** : modèle entraîné sur Paris uniquement. "
        "Résultats à interpréter avec précaution pour des dates futures éloignées."
    ),
    version="1.0.0",
    contact={"name": "Quentin Debrie", "email": "contact@exemple.fr"},
    license_info={"name": "MIT"},
)

# CORS (pour permettre les appels depuis Streamlit en développement local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ─── Chargement du modèle au démarrage ───────────────────────────────────────

_model = None
_eval_metrics = None


@app.on_event("startup")
def load_model():
    """Charge le modèle .pkl en mémoire au démarrage de l'API."""
    global _model, _eval_metrics

    if not MODEL_PATH.exists():
        print(f"[WARN] Modèle introuvable : {MODEL_PATH}")
        print("       Lancez d'abord : python train.py")
        return

    _model = joblib.load(MODEL_PATH)
    print(f"[OK] Modèle chargé depuis {MODEL_PATH}")

    if EVAL_PATH.exists():
        with open(EVAL_PATH, encoding="utf-8") as f:
            _eval_metrics = json.load(f)
        print(f"[OK] Métriques chargées depuis {EVAL_PATH}")


# ─── Schémas Pydantic ─────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    date: str
    risk_level: str
    risk_label: str
    probability: float
    confidence: str
    advice: str
    model_version: str = "1.0.0"
    disclaimer: str = (
        "Estimation basée sur les données historiques de Paris (75). "
        "Résultat indicatif, ne se substitue pas à une prévision météorologique officielle."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_available: bool
    api_version: str = "1.0.0"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_db_features_for_date(target_date: date) -> Optional[np.ndarray]:
    """
    Construit le vecteur de features pour une date donnée.
    
    Pour les dates passées : utilise les données réelles de la base.
    Pour les dates futures : utilise les moyennes saisonnières de la base
                             comme proxy (acceptable pour un MVP).
    """
    if not DB_PATH.exists():
        return None

    conn = sqlite3.connect(DB_PATH)
    target_str = target_date.strftime("%Y-%m-%d")

    # Tentative 1 : données réelles (date passée dans la base)
    row = conn.execute(
        "SELECT temp_max, temp_min, humidity_avg, pressure_avg, wind_avg, cloud_cover "
        "FROM observations WHERE date = ?",
        (target_str,)
    ).fetchone()

    if row:
        conn.close()
        temp_max, temp_min, humidity_avg, pressure_avg, wind_avg, cloud_cover = row
    else:
        # Tentative 2 : moyennes du même mois sur les années disponibles
        month = target_date.month
        avg_row = conn.execute(
            """SELECT 
                AVG(temp_max), AVG(temp_min), AVG(humidity_avg),
                AVG(pressure_avg), AVG(wind_avg), AVG(cloud_cover)
               FROM observations
               WHERE CAST(strftime('%m', date) AS INTEGER) = ?""",
            (month,)
        ).fetchone()
        conn.close()

        if not avg_row or avg_row[0] is None:
            return None

        temp_max, temp_min, humidity_avg, pressure_avg, wind_avg, cloud_cover = avg_row

    # Feature engineering (même logique que train.py)
    temp_range  = temp_max - temp_min
    month_num   = target_date.month
    day_of_year = target_date.timetuple().tm_yday

    features = np.array([[
        temp_max, temp_min, humidity_avg,
        pressure_avg, wind_avg, cloud_cover,
        temp_range, month_num, day_of_year,
    ]])

    return features


def _probability_to_risk(probability: float) -> tuple[str, str, str, str]:
    """
    Convertit une probabilité en niveau de risque verbalisé.
    
    Returns:
        (risk_level, risk_label, confidence, advice)
    """
    if probability < 0.35:
        return (
            "faible",
            "Risque faible de pluie",
            "haute" if probability < 0.20 else "modérée",
            "Conditions favorables pour des interventions agricoles. "
            "Risque de précipitations limité pour demain."
        )
    elif probability < 0.60:
        return (
            "modere",
            "Risque modéré de pluie",
            "modérée",
            "Incertitude sur les précipitations de demain. "
            "Prévoir un plan de secours pour les interventions critiques."
        )
    else:
        return (
            "eleve",
            "Risque élevé de pluie",
            "haute" if probability > 0.80 else "modérée",
            "Probabilité élevée de précipitations demain. "
            "Déconseillé de planifier des traitements ou récoltes sensibles à l'eau."
        )


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Projet Goutte d'Eau — API opérationnelle",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Statut de l'API",
    tags=["Monitoring"],
)
def health():
    """Vérifie que l'API, le modèle et la base de données sont disponibles."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        db_available=DB_PATH.exists(),
    )


@app.get(
    "/metrics",
    summary="Métriques d'évaluation du modèle",
    tags=["Modèle"],
)
def get_metrics():
    """
    Retourne les métriques d'évaluation du modèle entraîné.
    Transparence sur les performances et les limites du modèle.
    """
    if _eval_metrics is None:
        raise HTTPException(
            status_code=404,
            detail="Métriques non disponibles. Lancez d'abord : python train.py"
        )
    return _eval_metrics


@app.get(
    "/predict",
    response_model=PredictionResponse,
    summary="Estimation du risque de pluie",
    tags=["Prévision"],
)
def predict(
    date: str = Query(
        ...,
        description="Date pour laquelle estimer le risque de pluie le lendemain (format : YYYY-MM-DD)",
        example="2024-07-14",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    )
):
    """
    Retourne une estimation du risque de pluie pour le jour suivant la date fournie.
    
    - **date** : date au format YYYY-MM-DD
    - **risk_level** : faible | modere | eleve
    - **probability** : probabilité de pluie (0.0 – 1.0)
    - **confidence** : niveau de confiance de l'estimation
    - **advice** : conseil opérationnel adapté au niveau de risque
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Lancez d'abord : python train.py"
        )

    # Validation de la date
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Format de date invalide. Utilisez YYYY-MM-DD (ex: 2024-07-14)"
        )

    # Construction des features
    features = _get_db_features_for_date(target_date)
    if features is None:
        raise HTTPException(
            status_code=503,
            detail="Données météo indisponibles pour cette date. "
                   "Vérifiez que la base de données est peuplée (python collect.py)."
        )

    # Prédiction
    probability = float(_model.predict_proba(features)[0][1])
    risk_level, risk_label, confidence, advice = _probability_to_risk(probability)

    return PredictionResponse(
        date=date,
        risk_level=risk_level,
        risk_label=risk_label,
        probability=round(probability, 3),
        confidence=confidence,
        advice=advice,
    )
