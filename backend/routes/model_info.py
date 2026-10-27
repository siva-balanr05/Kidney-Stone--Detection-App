"""
backend/routes/model_info.py — Model info and health endpoints.
"""
import json
import os
import logging
from typing import List
from fastapi import APIRouter
from backend.schemas.prediction_schema import ModelInfo, ModelInfoResponse, HealthResponse
from backend.services.model_manager import model_manager
from backend.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


def _load_metrics(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load metrics from {path}: {e}")
    return {}


@router.get("/api/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def get_model_info():
    """Return availability status and metadata for all three models."""
    status = model_manager.model_status()

    cnn_metrics = _load_metrics(settings.CNN_METRICS_PATH)
    resnet_metrics = _load_metrics(settings.RESNET_METRICS_PATH)
    rf_metrics = _load_metrics(settings.RF_METRICS_PATH)

    models_list = [
        ModelInfo(
            name="Custom CNN",
            model_type="Convolutional Neural Network",
            framework="TensorFlow 2.15 / Keras",
            input_size="150 x 150 x 3",
            num_classes=len(settings.CLASS_NAMES),
            status="loaded" if status["cnn"] else "not_trained",
            version=model_manager.versions["cnn"],
            metrics_available=bool(cnn_metrics),
            accuracy=cnn_metrics.get("accuracy"),
        ),
        ModelInfo(
            name="ResNet50 Transfer Learning",
            model_type="Deep Transfer Learning",
            framework="TensorFlow 2.15 / Keras",
            input_size="224 x 224 x 3",
            num_classes=len(settings.CLASS_NAMES),
            status="loaded" if status["resnet"] else "not_trained",
            version=model_manager.versions["resnet"],
            metrics_available=bool(resnet_metrics),
            accuracy=resnet_metrics.get("accuracy"),
        ),
        ModelInfo(
            name="ResNet50 + Random Forest",
            model_type="Hybrid Deep-ML Ensemble",
            framework="scikit-learn 1.8",
            input_size="2048-dim (ResNet50 features)",
            num_classes=len(settings.CLASS_NAMES),
            status="loaded" if status["rf"] else "not_trained",
            version=model_manager.versions["rf"],
            metrics_available=bool(rf_metrics),
            accuracy=rf_metrics.get("accuracy"),
        ),
    ]
    return ModelInfoResponse(models=models_list, any_available=model_manager.any_available())


@router.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check — returns API status and model availability."""
    return HealthResponse(
        status="healthy",
        models_loaded=model_manager.model_status(),
        version="2.0.0",
    )


@router.get("/api/stats", tags=["Stats"])
def get_stats():
    """Return aggregate statistics from the database."""
    try:
        from backend.database import SessionLocal, Analysis
        from sqlalchemy import func
        db = SessionLocal()
        total = db.query(func.count(Analysis.id)).scalar()
        by_class = (
            db.query(Analysis.prediction, func.count(Analysis.id))
            .group_by(Analysis.prediction)
            .all()
        )
        db.close()
        return {
            "total_analyses": total,
            "by_class": {row[0]: row[1] for row in by_class},
        }
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        return {"total_analyses": 0, "by_class": {}}
