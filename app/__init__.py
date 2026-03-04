"""
Deepfake Detector Package

AI-powered deepfake detection system built with FastAPI and TensorFlow.
"""

__version__ = "1.0.0"
__author__ = "Deepfake Detector Team"
__description__ = "AI-powered deepfake detection system"

from app.main import app
from app.model import predict_deepfake_probability, initialize_models, get_available_models
from app.config import AppConfig

__all__ = [
    "app",
    "predict_deepfake_probability",
    "initialize_models",
    "get_available_models",
    "AppConfig"
]

