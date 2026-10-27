# KidneyAI Backend Tests
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import io
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# ─────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────
def test_health_returns_200():
    response = client.get("/api/health")
    assert response.status_code == 200

def test_health_returns_status_healthy():
    response = client.get("/api/health")
    data = response.json()
    assert data["status"] == "healthy"

def test_health_returns_version():
    response = client.get("/api/health")
    data = response.json()
    assert "version" in data

def test_health_returns_models_loaded():
    response = client.get("/api/health")
    data = response.json()
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], dict)


# ─────────────────────────────────────────────
# Model info endpoint
# ─────────────────────────────────────────────
def test_model_info_returns_200():
    response = client.get("/api/model-info")
    assert response.status_code == 200

def test_model_info_returns_models_list():
    response = client.get("/api/model-info")
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) == 3

def test_model_info_has_required_fields():
    response = client.get("/api/model-info")
    models = response.json()["models"]
    for model in models:
        assert "name" in model
        assert "status" in model
        assert "framework" in model
        assert "input_size" in model
        assert "num_classes" in model


# ─────────────────────────────────────────────
# History endpoint
# ─────────────────────────────────────────────
def test_history_returns_200():
    response = client.get("/api/history")
    assert response.status_code == 200

def test_history_returns_correct_shape():
    response = client.get("/api/history")
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)

def test_history_pagination_params():
    response = client.get("/api/history?skip=0&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) <= 5

def test_history_filter_by_prediction():
    response = client.get("/api/history?prediction=Cyst")
    assert response.status_code == 200


# ─────────────────────────────────────────────
# Stats endpoint
# ─────────────────────────────────────────────
def test_stats_returns_200():
    response = client.get("/api/stats")
    assert response.status_code == 200

def test_stats_returns_total_and_by_class():
    response = client.get("/api/stats")
    data = response.json()
    assert "total_analyses" in data
    assert "by_class" in data


# ─────────────────────────────────────────────
# Predict endpoint — validation tests (no model needed)
# ─────────────────────────────────────────────
def test_predict_rejects_non_image_file():
    fake_file = io.BytesIO(b"this is not an image")
    fake_file.name = "test.txt"
    response = client.post(
        "/api/predict",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400

def test_predict_rejects_wrong_extension():
    fake_file = io.BytesIO(b"fake gif data")
    response = client.post(
        "/api/predict",
        files={"file": ("test.gif", fake_file, "image/gif")},
    )
    assert response.status_code == 400

def test_predict_requires_file():
    response = client.post("/api/predict")
    assert response.status_code == 422  # Unprocessable entity — missing field


# ─────────────────────────────────────────────
# Root endpoint
# ─────────────────────────────────────────────
def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200

def test_root_returns_api_info():
    response = client.get("/")
    data = response.json()
    assert "message" in data
