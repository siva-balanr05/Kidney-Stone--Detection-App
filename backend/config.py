"""
backend/config.py — Application settings for KidneyAI FastAPI backend.
"""
import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Model artifact paths
    MODEL_ARTIFACTS_DIR: str = r"D:\Kidney-Stone--Detection-App\model_artifacts"

    @property
    def CNN_MODEL_PATH(self) -> str:
        return os.path.join(self.MODEL_ARTIFACTS_DIR, "cnn", "kidney_cnn.keras")

    @property
    def RESNET_MODEL_PATH(self) -> str:
        return os.path.join(self.MODEL_ARTIFACTS_DIR, "resnet", "kidney_resnet.keras")

    @property
    def RF_MODEL_PATH(self) -> str:
        return os.path.join(self.MODEL_ARTIFACTS_DIR, "random_forest", "rf_model.joblib")

    @property
    def SCALER_PATH(self) -> str:
        return os.path.join(self.MODEL_ARTIFACTS_DIR, "random_forest", "scaler.joblib")

    @property
    def CNN_METRICS_PATH(self) -> str:
        return os.path.join(self.MODEL_ARTIFACTS_DIR, "cnn", "cnn_metrics.json")

    @property
    def RESNET_METRICS_PATH(self) -> str:
        return os.path.join(self.MODEL_ARTIFACTS_DIR, "resnet", "resnet_metrics.json")

    @property
    def RF_METRICS_PATH(self) -> str:
        return os.path.join(self.MODEL_ARTIFACTS_DIR, "random_forest", "rf_metrics.json")

    # Dataset classes (alphabetical = Keras flow_from_directory order)
    CLASS_NAMES: List[str] = ["Cyst", "Normal", "Tumor"]

    # Image sizes
    IMG_SIZE_CNN: tuple = (150, 150)
    IMG_SIZE_RESNET: tuple = (224, 224)

    # Upload
    MAX_UPLOAD_SIZE_MB: int = 10
    UPLOAD_DIR: str = r"D:\Kidney-Stone--Detection-App\backend\uploads"

    # Database
    DB_URL: str = "sqlite:///D:/Kidney-Stone--Detection-App/backend/kidneyai.db"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # Disclaimer
    DISCLAIMER: str = (
        "This result is for research and educational purposes only. "
        "It is not a substitute for professional medical diagnosis or clinical decision-making."
    )

    class Config:
        env_file = ".env"


settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
