"""
backend/services/prediction_service.py — Orchestrates model inference pipeline.
"""
import io
import uuid
import logging
import numpy as np
from PIL import Image
from fastapi import HTTPException
from backend.config import settings
from backend.schemas.prediction_schema import PredictionResponse
from backend.services.model_manager import model_manager
from backend.services.preprocessing_service import (
    preprocess_image_for_cnn, preprocess_image_for_resnet, validate_and_decode
)
from backend.services.gradcam_service import generate_gradcam

logger = logging.getLogger(__name__)

# Layer names for Grad-CAM
CNN_GRADCAM_LAYER = "last_conv"
RESNET_GRADCAM_LAYER = "conv5_block3_out"


def run_predictions(image_bytes: bytes) -> PredictionResponse:
    """
    Run all available models on the uploaded image.
    Returns a PredictionResponse with the primary prediction and all model outputs.
    
    Primary model priority: CNN > ResNet > RF.
    Grad-CAM is generated for the CNN if available.
    """
    status = model_manager.model_status()
    if not any(status.values()):
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained models are currently available. "
                "Please train the models using the ML scripts first. "
                "See README.md for instructions."
            )
        )

    analysis_id = str(uuid.uuid4())
    warnings: list[str] = []

    # Decode raw image once for overlay
    raw_img = validate_and_decode(image_bytes)

    # --- CNN prediction ---
    cnn_probs = None
    if status["cnn"]:
        try:
            img_cnn = preprocess_image_for_cnn(image_bytes)
            cnn_probs = model_manager.cnn_model.predict(img_cnn, verbose=0)[0]
        except Exception as e:
            logger.error(f"CNN inference error: {e}")
            warnings.append("CNN inference failed.")

    # --- ResNet prediction ---
    resnet_probs = None
    img_resnet = None
    if status["resnet"]:
        try:
            img_resnet = preprocess_image_for_resnet(image_bytes)
            resnet_probs = model_manager.resnet_model.predict(img_resnet, verbose=0)[0]
        except Exception as e:
            logger.error(f"ResNet inference error: {e}")
            warnings.append("ResNet inference failed.")

    # --- RF prediction ---
    rf_probs = None
    if status["rf"]:
        try:
            if img_resnet is None:
                img_resnet = preprocess_image_for_resnet(image_bytes)
            features = model_manager.resnet_feature_extractor.predict(img_resnet, verbose=0)
            features_scaled = model_manager.scaler.transform(features)
            rf_probs_raw = model_manager.rf_model.predict_proba(features_scaled)[0]
            rf_probs = np.array(rf_probs_raw)
        except Exception as e:
            logger.error(f"RF inference error: {e}")
            warnings.append("Random Forest inference failed.")

    if cnn_probs is None and resnet_probs is None and rf_probs is None:
        raise HTTPException(status_code=500, detail="All model inferences failed.")

    # Select primary model (CNN preferred, then ResNet, then RF)
    if cnn_probs is not None:
        primary_probs = cnn_probs
        primary_model = "Custom CNN"
        primary_version = model_manager.versions["cnn"]
    elif resnet_probs is not None:
        primary_probs = resnet_probs
        primary_model = "ResNet50"
        primary_version = model_manager.versions["resnet"]
    else:
        primary_probs = rf_probs
        primary_model = "ResNet50 + Random Forest"
        primary_version = model_manager.versions["rf"]

    class_idx = int(np.argmax(primary_probs))
    prediction_label = settings.CLASS_NAMES[class_idx]
    probability = float(primary_probs[class_idx])
    all_probs = {settings.CLASS_NAMES[i]: float(primary_probs[i]) for i in range(len(settings.CLASS_NAMES))}

    # --- Grad-CAM (CNN first, then ResNet) ---
    gradcam_b64 = None
    if cnn_probs is not None and model_manager.cnn_model is not None:
        try:
            img_cnn = preprocess_image_for_cnn(image_bytes)
            gradcam_b64 = generate_gradcam(
                model=model_manager.cnn_model,
                img_array=img_cnn,
                class_idx=class_idx,
                last_conv_layer_name=CNN_GRADCAM_LAYER,
                original_img_array=raw_img,
            )
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")
    elif resnet_probs is not None and model_manager.resnet_model is not None:
        try:
            if img_resnet is None:
                img_resnet = preprocess_image_for_resnet(image_bytes)
            gradcam_b64 = generate_gradcam(
                model=model_manager.resnet_model,
                img_array=img_resnet,
                class_idx=class_idx,
                last_conv_layer_name=RESNET_GRADCAM_LAYER,
                original_img_array=raw_img,
            )
        except Exception as e:
            logger.warning(f"ResNet Grad-CAM failed: {e}")

    if gradcam_b64 is None and any(status.values()):
        warnings.append("Grad-CAM visualization could not be generated for this image.")

    return PredictionResponse(
        analysis_id=analysis_id,
        prediction=prediction_label,
        probability=probability,
        model_name=primary_model,
        model_version=primary_version,
        all_probabilities=all_probs,
        gradcam_available=(gradcam_b64 is not None),
        gradcam_image_b64=gradcam_b64,
        warnings=warnings,
        disclaimer=settings.DISCLAIMER,
    )
