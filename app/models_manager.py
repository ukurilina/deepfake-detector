from typing import Dict, Optional, List
import os
import json
import zipfile
import tempfile
import tensorflow as tf
import keras

from app.config import AppConfig
from app.audio_custom_objects import get_audio_custom_objects


def _sanitize_keras_config_node(node):
    if isinstance(node, dict):
        module = node.get("module")
        config = node.get("config")

        if isinstance(config, dict):
            if "quantization_config" in config:
                config.pop("quantization_config", None)

            if isinstance(module, str) and module.startswith("keras.src.ops"):
                # Older runtime op classes often reject serialized `name` in ctor.
                config.pop("name", None)

        for value in node.values():
            _sanitize_keras_config_node(value)
        return

    if isinstance(node, list):
        for item in node:
            _sanitize_keras_config_node(item)


def _build_sanitized_keras_archive(model_path: str) -> str:
    with zipfile.ZipFile(model_path, "r") as src_zip:
        config_raw = src_zip.read("config.json")
        config_obj = json.loads(config_raw.decode("utf-8"))
        _sanitize_keras_config_node(config_obj)
        new_config_raw = json.dumps(config_obj, ensure_ascii=True).encode("utf-8")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_file:
            sanitized_path = tmp_file.name

        with zipfile.ZipFile(sanitized_path, "w") as dst_zip:
            for info in src_zip.infolist():
                if info.filename == "config.json":
                    dst_zip.writestr("config.json", new_config_raw)
                else:
                    dst_zip.writestr(info, src_zip.read(info.filename))

    return sanitized_path


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
        audio_custom_objects = get_audio_custom_objects(include_ops=True) if content_type == "audio" else None

        if audio_custom_objects:
            load_attempts = (
                (keras.models.load_model, {"compile": False, "custom_objects": audio_custom_objects, "safe_mode": False}),
                (keras.models.load_model, {"compile": False, "custom_objects": audio_custom_objects}),
                (tf.keras.models.load_model, {"compile": False, "custom_objects": audio_custom_objects, "safe_mode": False}),
                (tf.keras.models.load_model, {"compile": False, "custom_objects": audio_custom_objects}),
                (keras.models.load_model, {"custom_objects": audio_custom_objects, "safe_mode": False}),
                (tf.keras.models.load_model, {"custom_objects": audio_custom_objects, "safe_mode": False}),
            )
        else:
            load_attempts = (
                (tf.keras.models.load_model, {"compile": False}),
                (tf.keras.models.load_model, {}),
            )

        for loader, kwargs in load_attempts:
            try:
                model = loader(model_path, **kwargs)
                self._models[model_name] = model
                self._model_configs[model_name] = {
                    "path": model_path,
                    "content_type": content_type,
                    "loaded": True,
                    "error": None,
                }
                return True
            except Exception as exc:
                mode = f"{loader.__module__}.{loader.__name__}({kwargs if kwargs else 'default'})"
                errors.append(f"{mode}: {exc}")

        if audio_custom_objects:
            sanitized_path = None
            try:
                sanitized_path = _build_sanitized_keras_archive(model_path)
                sanitized_custom_objects = get_audio_custom_objects(include_ops=False)
                sanitized_attempts = (
                    (keras.models.load_model, {"compile": False, "custom_objects": sanitized_custom_objects, "safe_mode": False}),
                    (keras.models.load_model, {"compile": False, "custom_objects": sanitized_custom_objects}),
                    (tf.keras.models.load_model, {"compile": False, "custom_objects": sanitized_custom_objects, "safe_mode": False}),
                    (tf.keras.models.load_model, {"compile": False, "custom_objects": sanitized_custom_objects}),
                )

                for loader, kwargs in sanitized_attempts:
                    try:
                        model = loader(sanitized_path, **kwargs)
                        self._models[model_name] = model
                        self._model_configs[model_name] = {
                            "path": model_path,
                            "content_type": content_type,
                            "loaded": True,
                            "error": None,
                        }
                        return True
                    except Exception as exc:
                        mode = f"sanitized:{loader.__module__}.{loader.__name__}({kwargs if kwargs else 'default'})"
                        errors.append(f"{mode}: {exc}")
            finally:
                if sanitized_path and os.path.exists(sanitized_path):
                    try:
                        os.unlink(sanitized_path)
                    except OSError:
                        pass

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
