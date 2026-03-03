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


def _extract_probability(preds: np.ndarray) -> float:
    """Извлекает вероятность deepfake из предсказания модели в разных форматах."""
    arr = np.asarray(preds, dtype="float32")

    if arr.size == 0:
        raise ValueError("Model returned empty prediction")

    # Сводим предсказание к одномерному вектору для одного примера
    sample = arr[0] if arr.ndim > 1 else arr
    flat = np.ravel(sample)

    if flat.size == 1:
        prob = float(flat[0])
    else:
        # Для бинарной softmax-модели считаем deepfake вторым классом
        prob = float(flat[1]) if flat.size >= 2 else float(flat[0])

    return max(0.0, min(1.0, prob))


def _build_model_inputs(model, image_tensor: np.ndarray):
    """Готовит входы для моделей с 1+ входами (включая вспомогательные входы)."""
    inputs = getattr(model, "inputs", None) or []
    if len(inputs) <= 1:
        return image_tensor

    prepared_inputs = []
    batch_size = image_tensor.shape[0]

    for model_input in inputs:
        shape = tuple(model_input.shape)

        # Типичный вход изображения: (None, H, W, C)
        if len(shape) == 4:
            prepared_inputs.append(image_tensor)
            continue

        # Типичный вспомогательный вектор признаков: (None, N)
        if len(shape) == 2:
            features = int(shape[1]) if shape[1] is not None else 1
            prepared_inputs.append(np.zeros((batch_size, features), dtype="float32"))
            continue

        # Универсальный fallback для редких форматов входа
        fallback_shape = [batch_size]
        for dim in shape[1:]:
            fallback_shape.append(int(dim) if dim is not None else 1)
        prepared_inputs.append(np.zeros(tuple(fallback_shape), dtype="float32"))

    return prepared_inputs


def predict_deepfake_probability(
    image_path: str,
    model_name: Optional[str] = None,
    threshold: float = 0.5
) -> dict:
    """Возвращает результат детекции дипфейка."""
    available_models = get_available_models()
    if not available_models:
        raise ValueError("No models loaded")

    default_model = AppConfig.DEFAULT_MODEL if AppConfig.DEFAULT_MODEL in available_models else available_models[0]
    selected_model = model_name or default_model

    model = _model_manager.get_model(selected_model)
    if model is None:
        raise ValueError(f"Model '{selected_model}' not found or not loaded")

    img_tensor = preprocess_image(image_path)
    model_inputs = _build_model_inputs(model, img_tensor)
    preds = model.predict(model_inputs, verbose=0)
    prob = _extract_probability(preds)

    confidence = prob if prob >= threshold else (1.0 - prob)
    label = "deepfake" if prob >= threshold else "real"

    return {
        "probability": prob,
        "percent": round(prob * 100.0, 4),
        "label": label,
        "confidence": confidence,
        "model_used": selected_model,
        "threshold": threshold,
    }
