"""
backend/services/model_manager.py — Singleton model manager.
Loads models once at startup; handles missing files gracefully.
"""
import os
import json
import logging
import joblib
import numpy as np
import tensorflow as tf
from backend.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.cnn_model = None
            cls._instance.resnet_model = None
            cls._instance.rf_model = None
            cls._instance.scaler = None
            cls._instance.resnet_feature_extractor = None
            cls._instance.versions = {"cnn": "unavailable", "resnet": "unavailable", "rf": "unavailable"}
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_all()

    def _get_version(self, subdir: str) -> str:
        path = os.path.join(settings.MODEL_ARTIFACTS_DIR, subdir, "version.json")
        try:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f).get("version", "unknown")
        except Exception as e:
            logger.warning(f"Could not read version for {subdir}: {e}")
        return "unknown"

    def _load_all(self):
        # CNN
        if os.path.exists(settings.CNN_MODEL_PATH):
            try:
                self.cnn_model = tf.keras.models.load_model(settings.CNN_MODEL_PATH)
                self.versions["cnn"] = self._get_version("cnn")
                logger.info(f"CNN model loaded: {settings.CNN_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to load CNN: {e}")
        else:
            logger.warning(f"CNN model not found at {settings.CNN_MODEL_PATH}. Train first.")

        # ResNet
        if os.path.exists(settings.RESNET_MODEL_PATH):
            try:
                self.resnet_model = tf.keras.models.load_model(settings.RESNET_MODEL_PATH)
                self.versions["resnet"] = self._get_version("resnet")
                # Build feature extractor sharing ResNet base weights
                try:
                    gap_output = self.resnet_model.get_layer("global_average_pooling2d").output
                    self.resnet_feature_extractor = tf.keras.models.Model(
                        inputs=self.resnet_model.input, outputs=gap_output,
                        name="ResNet50_FeatureExtractor"
                    )
                except Exception:
                    base_output = self.resnet_model.get_layer("conv5_block3_out").output
                    pool = tf.keras.layers.GlobalAveragePooling2D()(base_output)
                    self.resnet_feature_extractor = tf.keras.models.Model(
                        inputs=self.resnet_model.input, outputs=pool,
                        name="ResNet50_FeatureExtractor"
                    )
                logger.info(f"ResNet50 model loaded: {settings.RESNET_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to load ResNet: {e}")
                # Fallback: build fresh extractor with ImageNet weights
                try:
                    base = tf.keras.applications.ResNet50(
                        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
                    )
                    pool = tf.keras.layers.GlobalAveragePooling2D()(base.output)
                    self.resnet_feature_extractor = tf.keras.models.Model(
                        inputs=base.input, outputs=pool
                    )
                    logger.info("ResNet feature extractor (ImageNet only) loaded as fallback.")
                except Exception as e2:
                    logger.error(f"Fallback ResNet extractor also failed: {e2}")
        else:
            logger.warning(f"ResNet model not found at {settings.RESNET_MODEL_PATH}. Train first.")

        # RF + Scaler
        if os.path.exists(settings.RF_MODEL_PATH) and os.path.exists(settings.SCALER_PATH):
            try:
                self.rf_model = joblib.load(settings.RF_MODEL_PATH)
                self.scaler = joblib.load(settings.SCALER_PATH)
                self.versions["rf"] = self._get_version("random_forest")
                logger.info(f"Random Forest loaded: {settings.RF_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to load RF: {e}")
        else:
            logger.warning("RF model or scaler not found. Run extract_resnet_features.py and train_random_forest.py first.")

    def model_status(self) -> dict:
        if not self._initialized:
            self.initialize()
        return {
            "cnn": self.cnn_model is not None,
            "resnet": self.resnet_model is not None,
            "rf": self.rf_model is not None and self.scaler is not None,
        }

    def any_available(self) -> bool:
        return any(self.model_status().values())


model_manager = ModelManager()
