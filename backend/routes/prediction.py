"""
backend/routes/prediction.py — POST /api/predict endpoint.
"""
import os
import json
import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db, Analysis
from backend.schemas.prediction_schema import PredictionResponse
from backend.services.prediction_service import run_predictions
from backend.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@router.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_image(
    file: UploadFile = File(..., description="Kidney scan image (JPG or PNG, max 10MB)"),
    db: Session = Depends(get_db),
):
    """
    Upload a kidney scan image and receive a classification prediction.
    
    Returns the predicted class (Cyst / Normal / Tumor) with probability scores,
    model information, Grad-CAM visualization (when available), and a disclaimer.
    
    This endpoint is for research and educational purposes only.
    """
    # File type validation
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG."
        )

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {ext}. Allowed: .jpg, .jpeg, .png"
        )

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file.")

    # Run inference
    response = run_predictions(image_bytes)

    # Save image to disk
    try:
        filename = f"{response.analysis_id}{ext}"
        filepath = os.path.join(settings.UPLOAD_DIR, filename)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        with open(filepath, "wb") as fh:
            fh.write(image_bytes)
    except Exception as e:
        logger.warning(f"Could not save image to disk: {e}")
        filename = file.filename or "unknown"

    # Persist to database
    try:
        db_record = Analysis(
            analysis_id=response.analysis_id,
            prediction=response.prediction,
            probability=response.probability,
            model_name=response.model_name,
            model_version=response.model_version,
            image_filename=filename,
            explanation_available=response.gradcam_available,
            all_probabilities_json=json.dumps(response.all_probabilities),
        )
        db.add(db_record)
        db.commit()
        logger.info(f"Analysis {response.analysis_id} saved. Prediction: {response.prediction}")
    except Exception as e:
        logger.error(f"DB write failed: {e}")
        # Do not crash the API response if DB fails

    return response
