import numpy as np
from PIL import Image
from typing import Optional, Tuple

from app.config import AppConfig
from app.models_manager import ModelManager

# Инициализация менеджера моделей
_model_manager = ModelManager()


def initialize_models() -> bool:
    """Инициализирует и загружает все доступные модели."""
    try:
        results = _model_manager.load_all_models_from_directory()
        if not results:
            return False
        return sum(1 for v in results.values() if v) > 0
    except Exception:
        return False


def get_available_models() -> list:
    """Возвращает список успешно загруженных моделей."""
    models = _model_manager.get_loaded_models_info()
    return [m["name"] for m in models if m["status"] == "loaded"]


def preprocess_image(image_path: str, target_size: Tuple[int, int] = AppConfig.TARGET_IMAGE_SIZE) -> np.ndarray:
    """Загружает и подготавливает изображение к подаче в модель."""
    try:
        with Image.open(image_path).convert("RGB") as im:
            im = im.resize(target_size)
            arr = np.asarray(im, dtype="float32") / 255.0
        # Добавляем размер батча: (1, H, W, C)
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise ValueError(f"Cannot process image: {str(e)}")


def predict_deepfake_probability(
    image_path: str,
    model_name: Optional[str] = None,
    threshold: float = 0.5
) -> dict:
    """Возвращает результат детекции дипфейка."""
    # Определяем модель для использования
    available_models = get_available_models()
    if not available_models:
        raise ValueError("No models loaded")

    selected_model = model_name or available_models[0]
    model = _model_manager.get_model(selected_model)
    if model is None:
        raise ValueError(f"Model '{selected_model}' not found or not loaded")

    # Предобработка изображения
    img_tensor = preprocess_image(image_path)

    # Выполнение предсказания
    preds = model.predict(img_tensor, verbose=0)
    prob = float(preds[0][0])
    # На всякий случай ограничим диапазон
    prob = max(0.0, min(1.0, prob))

    confidence = prob if prob >= threshold else (1.0 - prob)
    label = "deepfake" if prob >= threshold else "real"

    return {
        "probability": prob,
        "label": label,
        "confidence": confidence,
        "model_used": selected_model,
        "threshold": threshold,
    }
