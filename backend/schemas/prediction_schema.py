"""
backend/schemas/prediction_schema.py — Pydantic schemas for API responses.
"""
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime


class PredictionResponse(BaseModel):
    analysis_id: str
    prediction: str
    probability: float
    model_name: str
    model_version: str
    all_probabilities: Dict[str, float]
    gradcam_available: bool
    gradcam_image_b64: Optional[str] = None
    warnings: List[str] = []
    disclaimer: str

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ModelInfo(BaseModel):
    name: str
    model_type: str
    framework: str
    input_size: str
    num_classes: int
    status: str
    version: str
    metrics_available: bool = False
    accuracy: Optional[float] = None


class ModelInfoResponse(BaseModel):
    models: List[ModelInfo]
    any_available: bool


class HistoryItem(BaseModel):
    analysis_id: str
    created_at: datetime
    prediction: str
    probability: float
    model_name: str
    image_filename: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AnalysisList(BaseModel):
    total: int
    items: List[HistoryItem]


class AnalysisDetail(BaseModel):
    analysis_id: str
    created_at: datetime
    prediction: str
    probability: float
    model_name: str
    model_version: str
    explanation_available: bool
    all_probabilities: Optional[Dict[str, float]] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    version: str = "2.0.0"
