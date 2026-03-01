from typing import Dict, Optional, List
import os
import tensorflow as tf


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
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models"
        )

    def load_model(self, model_name: str, model_path: str) -> bool:
        """
        Загружает модель в память.

        Args:
            model_name: Уникальное имя модели
            model_path: Полный путь к файлу модели

        Returns:
            True если успешно загружена, False иначе
        """
        try:
            if not os.path.exists(model_path):
                return False

            model = tf.keras.models.load_model(model_path)
            self._models[model_name] = model
            self._model_configs[model_name] = {
                "path": model_path,
                "loaded": True
            }
            return True

        except Exception as e:
            return False

    def load_all_models_from_directory(self) -> Dict[str, bool]:
        """
        Загружает все .keras модели из директории models/.

        Returns:
            Dict с результатами загрузки каждой модели
        """
        results = {}

        if not os.path.exists(self.models_dir):
            return results

        for filename in os.listdir(self.models_dir):
            if filename.endswith(".keras"):
                model_name = filename.replace(".keras", "")
                model_path = os.path.join(self.models_dir, filename)
                results[model_name] = self.load_model(model_name, model_path)

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
                "status": "loaded" if config["loaded"] else "error"
            }
            for name, config in self._model_configs.items()
        ]

    def is_model_loaded(self, model_name: str) -> bool:
        """Проверяет, загружена ли модель."""
        return model_name in self._models

    def get_loaded_count(self) -> int:
        """Возвращает количество загруженных моделей."""
        return len(self._models)
