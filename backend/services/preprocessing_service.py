"""
backend/services/preprocessing_service.py — Image validation and preprocessing.
"""
import io
import logging
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from fastapi import HTTPException
from backend.config import settings

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


def validate_and_decode(image_bytes: bytes) -> np.ndarray:
    """Validate image bytes and return an RGB numpy array."""
    if len(image_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit."
        )
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        arr = np.array(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Image must have 3 channels")
        return arr
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Image decode failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")


def preprocess_image_for_cnn(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for custom CNN: 150x150, float32, /255."""
    arr = validate_and_decode(image_bytes)
    resized = cv2.resize(arr, settings.IMG_SIZE_CNN)  # (150, 150)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)  # (1, 150, 150, 3)


def preprocess_image_for_resnet(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for ResNet50: 224x224, resnet50 preprocess_input."""
    arr = validate_and_decode(image_bytes)
    resized = cv2.resize(arr, settings.IMG_SIZE_RESNET)  # (224, 224)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(resized.astype(np.float32))
    return np.expand_dims(preprocessed, axis=0)  # (1, 224, 224, 3)
