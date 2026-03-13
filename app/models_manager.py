from typing import Dict, Optional, List
import os
import tensorflow as tf

from app.config import AppConfig
from app.audio_custom_objects import get_audio_custom_objects


class ModelManager:
    """Менеджер для загрузки и управления ML моделями."""

    _instance = None
    _models: Dict[str, tf.keras.Model] = {}
    _model_configs: Dict[str, dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Инициализация менеджера моделей."""
        self.models_dir = AppConfig.MODELS_DIR

    def load_model(self, model_name: str, model_path: str, content_type: str) -> bool:
        """
        Загружает модель в память.

        Args:
            model_name: Уникальное имя модели
            model_path: Полный путь к файлу модели

        Returns:
            True если успешно загружена, False иначе
        """
        if not os.path.exists(model_path):
            self._model_configs[model_name] = {
                "path": model_path,
                "content_type": content_type,
                "loaded": False,
                "error": "Model file not found",
            }
            return False

        errors = []
        audio_custom_objects = get_audio_custom_objects() if content_type == "audio" else None

        if audio_custom_objects:
            load_attempts = (
                {"compile": False, "custom_objects": audio_custom_objects, "safe_mode": False},
                {"compile": False, "custom_objects": audio_custom_objects},
                {"custom_objects": audio_custom_objects, "safe_mode": False},
                {"custom_objects": audio_custom_objects},
            )
        else:
            load_attempts = (
                {"compile": False},
                {},
            )

        for kwargs in load_attempts:
            try:
                model = tf.keras.models.load_model(model_path, **kwargs)
                self._models[model_name] = model
                self._model_configs[model_name] = {
                    "path": model_path,
                    "content_type": content_type,
                    "loaded": True,
                    "error": None,
                }
                return True
            except Exception as exc:
                print(exc)
                mode = "compile=False" if kwargs else "default"
                errors.append(f"{mode}: {exc}")

        self._model_configs[model_name] = {
            "path": model_path,
            "content_type": content_type,
            "loaded": False,
            "error": " | ".join(errors),
        }
        return False

    def load_all_models_from_directory(self) -> Dict[str, bool]:
        """
        Загружает только поддерживаемые .keras модели из директории models/.

        Returns:
            Dict с результатами загрузки каждой модели
        """
        results = {}
        self._models.clear()
        self._model_configs.clear()

        if not os.path.exists(self.models_dir):
            return results

        for model_name, spec in AppConfig.MODEL_REGISTRY.items():
            model_filename = str(spec["file_name"])
            model_path = os.path.join(self.models_dir, model_filename)
            content_type = str(spec["content_type"])
            results[model_name] = self.load_model(model_name, model_path, content_type)

        return results

    def get_model(self, model_name: str) -> Optional[tf.keras.Model]:
        """Получает загруженную модель по имени."""
        return self._models.get(model_name)

    def get_all_models(self) -> Dict[str, tf.keras.Model]:
        """Получает все загруженные модели."""
        return self._models.copy()

    def get_loaded_models_info(self) -> List[Dict]:
        """Получает информацию о всех загруженных моделях."""
        return [
            {
                "name": name,
                "path": config["path"],
                "content_type": config.get("content_type"),
                "status": "loaded" if config["loaded"] else "error",
                "error": config.get("error"),
            }
            for name, config in self._model_configs.items()
        ]

    def is_model_loaded(self, model_name: str) -> bool:
        """Проверяет, загружена ли модель."""
        return model_name in self._models

    def get_loaded_count(self) -> int:
        """Возвращает количество загруженных моделей."""
        return len(self._models)
