"""
tests/test_api.py — Tests unitaires de l'API FastAPI
Projet Goutte d'Eau — MVP BLOC 2

Lancement :
    pytest tests/ -v
    pytest tests/ -v --cov=src
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# Fixtures

@pytest.fixture
def temp_db():
    """Crée une base SQLite temporaire peuplée pour les tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE observations (
            date TEXT PRIMARY KEY, temp_max REAL, temp_min REAL,
            humidity_avg REAL, pressure_avg REAL, wind_avg REAL,
            cloud_cover REAL, rain_tomorrow INTEGER
        )
    """)
    # Insertion de données de test
    test_data = [
        ("2024-01-15", 8.5, 2.1, 75.0, 1012.0, 15.0, 60.0, 1),
        ("2024-06-15", 25.0, 15.0, 45.0, 1018.0, 10.0, 20.0, 0),
        ("2024-10-10", 16.0, 8.0, 80.0, 1005.0, 25.0, 85.0, 1),
    ]
    conn.executemany(
        "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?)", test_data
    )
    conn.commit()
    conn.close()
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def mock_model():
    """Mock du modèle scikit-learn."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture
def client(temp_db, mock_model, tmp_path):
    """Client de test FastAPI avec modèle et DB mockés."""
    import importlib
    import sys

    # Patch des chemins dans main.py
    eval_data = {
        "accuracy": 0.72, "f1_score": 0.68, "roc_auc": 0.75,
        "confusion_matrix": [[80, 20], [15, 85]],
        "feature_importance": {"temp_max": 0.25, "humidity_avg": 0.20}
    }
    eval_path = tmp_path / "evaluation.json"
    eval_path.write_text(json.dumps(eval_data))

    with patch("src.main.MODEL_PATH", tmp_path / "model.pkl"), \
         patch("src.main.EVAL_PATH", eval_path), \
         patch("src.main.DB_PATH", temp_db), \
         patch("src.main._model", mock_model), \
         patch("src.main._eval_metrics", eval_data):
        from src.main import app
        yield TestClient(app)


# Tests endpoint /health

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_structure(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "db_available" in data

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


# Tests endpoint /predict

class TestPredict:
    def test_predict_valid_date(self, client):
        """Date présente dans la base de test."""
        response = client.get("/predict", params={"date": "2024-01-15"})
        assert response.status_code == 200

    def test_predict_response_structure(self, client):
        """Vérification de la structure de la réponse."""
        data = client.get("/predict", params={"date": "2024-01-15"}).json()
        assert "date" in data
        assert "risk_level" in data
        assert "probability" in data
        assert "advice" in data
        assert "disclaimer" in data

    def test_predict_risk_level_values(self, client):
        """Le niveau de risque doit appartenir à l'ensemble défini."""
        data = client.get("/predict", params={"date": "2024-01-15"}).json()
        assert data["risk_level"] in {"faible", "modere", "eleve"}

    def test_predict_probability_range(self, client):
        """La probabilité doit être entre 0 et 1."""
        data = client.get("/predict", params={"date": "2024-01-15"}).json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_invalid_date_format(self, client):
        """Date dans un format invalide → 422."""
        response = client.get("/predict", params={"date": "15/01/2024"})
        assert response.status_code == 422

    def test_predict_missing_date(self, client):
        """Paramètre date absent → 422."""
        response = client.get("/predict")
        assert response.status_code == 422


# Tests endpoint /metrics

class TestMetrics:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_accuracy(self, client):
        data = client.get("/metrics").json()
        assert "accuracy" in data
        assert isinstance(data["accuracy"], float)


# Tests de la logique de conversion risque

class TestRiskConversion:
    """Tests unitaires de la fonction _probability_to_risk."""

    def test_low_probability_is_faible(self):
        from src.main import _probability_to_risk
        risk_level, _, _, _ = _probability_to_risk(0.15)
        assert risk_level == "faible"

    def test_medium_probability_is_modere(self):
        from src.main import _probability_to_risk
        risk_level, _, _, _ = _probability_to_risk(0.50)
        assert risk_level == "modere"

    def test_high_probability_is_eleve(self):
        from src.main import _probability_to_risk
        risk_level, _, _, _ = _probability_to_risk(0.85)
        assert risk_level == "eleve"

    def test_boundary_35_is_modere(self):
        from src.main import _probability_to_risk
        risk_level, _, _, _ = _probability_to_risk(0.35)
        assert risk_level == "modere"

    def test_returns_non_empty_advice(self):
        from src.main import _probability_to_risk
        for proba in [0.1, 0.45, 0.80]:
            _, _, _, advice = _probability_to_risk(proba)
            assert len(advice) > 10
