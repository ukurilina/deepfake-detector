import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

from app.config import AppConfig
from app.models_manager import ModelManager
from app.video_utils import get_key_frames

# Инициализация менеджера моделей
_model_manager = ModelManager()

AUDIO_SAMPLE_RATE = 16000
AUDIO_DEFAULT_SEGMENT_LEN = 3 * AUDIO_SAMPLE_RATE
AUDIO_DEFAULT_HOP_LEN = int(1.5 * AUDIO_SAMPLE_RATE)


def compute_fft(img: np.ndarray, crop_ratio: Optional[float] = 0.4) -> np.ndarray:
    """
    Вычисляет FFT (быстрое преобразование Фурье) изображения.

    Args:
        img: Нормализованное изображение (H, W, 3) в формате RGB, значения 0-1

    Returns:
        FFT-преобразованное изображение (104, 104, 3) с каналами:
        [magnitude, phase_sin, phase_cos]
    """
    # Конвертируем в тензор TensorFlow
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # Конвертируем в градации серого
    gray = tf.image.rgb_to_grayscale(img_tensor)
    gray = tf.squeeze(gray, axis=-1)

    # Преобразуем в комплексный формат
    complex_img = tf.cast(gray, tf.complex64)

    # Применяем 2D FFT и центрируем
    fft = tf.signal.fft2d(complex_img)
    fft = tf.signal.fftshift(fft)

    # Вычисляем магнитуду (логарифмическая шкала и нормализация)
    mag = tf.math.log(tf.abs(fft) + 1e-8)
    mag = (mag - tf.reduce_mean(mag)) / (tf.math.reduce_std(mag) + 1e-8)

    # Вычисляем фазу
    phase = tf.math.angle(fft)
    phase_sin = tf.sin(phase)
    phase_cos = tf.cos(phase)

    # Объединяем в 3-канальное изображение
    fft_img = tf.stack([mag, phase_sin, phase_cos], axis=-1)

    # Для фото-ветки обычно используется центральный кроп, для видео-ветки - полный размер.
    if crop_ratio is not None:
        fft_img = tf.image.central_crop(fft_img, crop_ratio)

    return fft_img.numpy()


def initialize_models() -> bool:
    """Инициализирует и загружает все доступные модели."""
    try:
        results = _model_manager.load_all_models_from_directory()
        if not results:
            return False
        return sum(1 for v in results.values() if v) > 0
    except Exception:
        return False


def get_available_models(content_type: Optional[str] = None) -> list:
    """Возвращает список успешно загруженных моделей (опционально по типу контента)."""
    models = _model_manager.get_loaded_models_info()
    filtered = [m for m in models if m["status"] == "loaded"]
    if content_type:
        filtered = [m for m in filtered if m.get("content_type") == content_type]
    return [m["name"] for m in filtered]


def _to_int_dim(dim: Any, fallback: int) -> int:
    if dim is None:
        return fallback
    try:
        return int(dim)
    except Exception:
        return fallback


def _infer_photo_model_specs(model) -> Dict[str, Any]:
    inputs = getattr(model, "inputs", None) or []
    image_like = []
    for idx, inp in enumerate(inputs):
        shape = tuple(inp.shape)
        if len(shape) != 4:
            continue
        h = _to_int_dim(shape[1], AppConfig.TARGET_IMAGE_SIZE[0])
        w = _to_int_dim(shape[2], AppConfig.TARGET_IMAGE_SIZE[1])
        c = _to_int_dim(shape[3], 3)
        image_like.append({"index": idx, "size": (h, w), "channels": c})

    if not image_like:
        return {
            "rgb_index": 0,
            "fft_index": None,
            "rgb_size": AppConfig.TARGET_IMAGE_SIZE,
            "fft_size": AppConfig.TARGET_IMAGE_SIZE,
        }

    if len(image_like) == 1:
        return {
            "rgb_index": image_like[0]["index"],
            "fft_index": None,
            "rgb_size": image_like[0]["size"],
            "fft_size": image_like[0]["size"],
        }

    image_like_sorted = sorted(image_like, key=lambda item: item["size"][0] * item["size"][1], reverse=True)
    rgb_spec = image_like_sorted[0]
    fft_spec = image_like_sorted[1]
    return {
        "rgb_index": rgb_spec["index"],
        "fft_index": fft_spec["index"],
        "rgb_size": rgb_spec["size"],
        "fft_size": fft_spec["size"],
    }


def _resize_array_hwc(arr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    h, w = target_size
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    resized = tf.image.resize(arr, [h, w], method="bilinear")
    return tf.cast(resized, tf.float32).numpy()


def _normalize_heatmap_to_uint8(heatmap: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    arr = np.asarray(heatmap, dtype="float32")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if target_size is not None:
        arr = tf.image.resize(arr[..., np.newaxis], [target_size[0], target_size[1]], method="bilinear").numpy()[..., 0]

    min_v = float(np.min(arr)) if arr.size else 0.0
    max_v = float(np.max(arr)) if arr.size else 0.0
    if max_v - min_v < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)

    norm = (arr - min_v) / (max_v - min_v)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def _serialize_heatmap(heatmap_uint8: np.ndarray) -> List[List[int]]:
    return heatmap_uint8.astype(np.uint8).tolist()


def preprocess_image(image_path: str, target_size: Tuple[int, int] = AppConfig.TARGET_IMAGE_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает и подготавливает изображение к подаче в модель.

    Args:
        image_path: Путь к изображению
        target_size: Размер для изменения изображения (по умолчанию 256x256)

    Returns:
        Кортеж из двух массивов:
        - Оригинальное изображение (1, 256, 256, 3)
        - FFT-преобразованное изображение (1, 104, 104, 3)
    """
    try:
        with Image.open(image_path).convert("RGB") as im:
            im = im.resize(target_size)
            arr = np.asarray(im, dtype="float32") / 255.0

        # Вычисляем FFT для изображения
        fft_arr = compute_fft(arr)

        # Добавляем размер батча: (1, H, W, C)
        img_batch = np.expand_dims(arr, axis=0)
        fft_batch = np.expand_dims(fft_arr, axis=0)

        return img_batch, fft_batch
    except Exception as e:
        raise ValueError(f"Cannot process image: {str(e)}")


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = AppConfig.TARGET_IMAGE_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Подготавливает RGB-кадр видео к подаче в модель."""
    try:
        pil_frame = Image.fromarray(frame.astype("uint8"), mode="RGB").resize(target_size)
        arr = np.asarray(pil_frame, dtype="float32") / 255.0
        fft_arr = compute_fft(arr)
        return np.expand_dims(arr, axis=0), np.expand_dims(fft_arr, axis=0)
    except Exception as e:
        raise ValueError(f"Cannot process video frame: {str(e)}")


def _preprocess_photo_for_model(image_path: str, model) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    specs = _infer_photo_model_specs(model)
    rgb_size = specs["rgb_size"]
    fft_size = specs["fft_size"]

    try:
        with Image.open(image_path).convert("RGB") as im:
            im = im.resize((rgb_size[1], rgb_size[0]))
            arr = np.asarray(im, dtype="float32") / 255.0
    except Exception as e:
        raise ValueError(f"Cannot process image: {str(e)}")

    fft_arr = None
    if specs["fft_index"] is not None:
        use_crop = fft_size[0] < rgb_size[0] or fft_size[1] < rgb_size[1]
        fft_arr = compute_fft(arr, crop_ratio=0.4 if use_crop else None)
        fft_arr = _resize_array_hwc(fft_arr, fft_size)

    img_batch = np.expand_dims(arr, axis=0)
    fft_batch = np.expand_dims(fft_arr, axis=0) if fft_arr is not None else None
    return img_batch, fft_batch, specs


def _infer_video_model_specs(model, fallback_num_frames: int) -> Dict[str, Any]:
    inputs = getattr(model, "inputs", None) or []
    sequence_like = []
    for idx, inp in enumerate(inputs):
        shape = tuple(inp.shape)
        if len(shape) != 5:
            continue
        t = _to_int_dim(shape[1], fallback_num_frames)
        h = _to_int_dim(shape[2], 112)
        w = _to_int_dim(shape[3], 112)
        c = _to_int_dim(shape[4], 3)
        sequence_like.append({"index": idx, "num_frames": t, "size": (h, w), "channels": c})

    if not sequence_like:
        raise ValueError("Loaded video model does not expose sequence inputs")

    if len(sequence_like) == 1:
        spec = sequence_like[0]
        return {
            "rgb_index": spec["index"],
            "fft_index": None,
            "num_frames": spec["num_frames"],
            "rgb_size": spec["size"],
            "fft_size": spec["size"],
        }

    # По контракту обучения RGB-вход идет первым, FFT-вход вторым.
    first, second = sequence_like[0], sequence_like[1]
    return {
        "rgb_index": first["index"],
        "fft_index": second["index"],
        "num_frames": first["num_frames"],
        "rgb_size": first["size"],
        "fft_size": second["size"],
    }


def _extract_video_inputs(video_path: str, model, max_frames: int) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any], int]:
    specs = _infer_video_model_specs(model, fallback_num_frames=max_frames)
    target_frames = max(1, min(specs["num_frames"], max_frames))

    frames = get_key_frames(video_path, num_frames=target_frames)
    if not frames:
        raise ValueError("No frames extracted from video")

    rgb_frames: List[np.ndarray] = []
    fft_frames: List[np.ndarray] = []
    use_fft = specs["fft_index"] is not None
    use_crop = specs["fft_size"][0] < specs["rgb_size"][0] or specs["fft_size"][1] < specs["rgb_size"][1]

    for frame in frames:
        pil_frame = Image.fromarray(frame.astype("uint8"), mode="RGB")

        rgb_pil = pil_frame.resize((specs["rgb_size"][1], specs["rgb_size"][0]))
        rgb_arr = np.asarray(rgb_pil, dtype="float32") / 255.0
        rgb_frames.append(rgb_arr)

        if use_fft:
            fft_arr = compute_fft(rgb_arr, crop_ratio=0.4 if use_crop else None)
            fft_arr = _resize_array_hwc(fft_arr, specs["fft_size"])
            fft_frames.append(fft_arr)

    rgb_batch = np.expand_dims(np.asarray(rgb_frames, dtype="float32"), axis=0)
    fft_batch = np.expand_dims(np.asarray(fft_frames, dtype="float32"), axis=0) if use_fft else None
    return rgb_batch, fft_batch, specs, len(rgb_frames)


def _compose_model_inputs(model, rgb_tensor: np.ndarray, fft_tensor: Optional[np.ndarray], rgb_index: int, fft_index: Optional[int]):
    inputs = getattr(model, "inputs", None) or []
    if len(inputs) <= 1:
        return rgb_tensor

    prepared_inputs: List[np.ndarray] = []
    for idx, model_input in enumerate(inputs):
        if idx == rgb_index:
            prepared_inputs.append(rgb_tensor)
            continue
        if fft_index is not None and idx == fft_index and fft_tensor is not None:
            prepared_inputs.append(fft_tensor)
            continue

        shape = tuple(model_input.shape)
        fallback_shape = [rgb_tensor.shape[0]]
        for dim in shape[1:]:
            fallback_shape.append(_to_int_dim(dim, 1))
        prepared_inputs.append(np.zeros(tuple(fallback_shape), dtype="float32"))

    return prepared_inputs


def _find_photo_rgb_conv_layer(model) -> Optional[str]:
    candidates = []
    conv_markers = ("Conv2D", "SeparableConv2D", "DepthwiseConv2D")

    for idx, layer in enumerate(model.layers):
        class_name = layer.__class__.__name__
        if not any(marker in class_name for marker in conv_markers):
            continue

        try:
            shape = tuple(layer.output.shape)
        except Exception:
            continue

        if len(shape) != 4:
            continue

        h = _to_int_dim(shape[1], 0)
        w = _to_int_dim(shape[2], 0)
        c = _to_int_dim(shape[3], 0)
        if h <= 0 or w <= 0 or c <= 0:
            continue

        candidates.append((c, h * w, idx, layer.name))

    if not candidates:
        return None

    # Выбираем глубокий сверточный слой RGB-ветки: максимум каналов, затем большая карта.
    candidates.sort(reverse=True)
    return candidates[0][3]


def _generate_photo_gradcam(model, model_inputs, target_size: Tuple[int, int]) -> Optional[List[List[int]]]:
    target_layer_name = _find_photo_rgb_conv_layer(model)
    if not target_layer_name:
        return None

    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(model_inputs, training=False)
        score = preds[:, 0] if preds.shape.rank and preds.shape.rank > 1 else preds

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    conv_map = conv_outputs[0]
    cam = tf.reduce_sum(conv_map * pooled_grads[0], axis=-1)
    cam = tf.nn.relu(cam).numpy()

    heatmap_uint8 = _normalize_heatmap_to_uint8(cam, target_size=target_size)
    return _serialize_heatmap(heatmap_uint8)


def _generate_video_input_saliency_heatmaps(model, model_inputs, rgb_index: int, target_size: Tuple[int, int]) -> List[List[List[int]]]:
    if isinstance(model_inputs, list):
        rgb_tensor = tf.convert_to_tensor(model_inputs[rgb_index], dtype=tf.float32)
        wrapped_inputs = list(model_inputs)
        wrapped_inputs[rgb_index] = rgb_tensor
        wrapped_inputs = [tf.convert_to_tensor(inp, dtype=tf.float32) for inp in wrapped_inputs]
    else:
        rgb_tensor = tf.convert_to_tensor(model_inputs, dtype=tf.float32)
        wrapped_inputs = rgb_tensor

    with tf.GradientTape() as tape:
        tape.watch(rgb_tensor)
        preds = model(wrapped_inputs, training=False)
        score = preds[:, 0] if preds.shape.rank and preds.shape.rank > 1 else preds

    grads = tape.gradient(score, rgb_tensor)
    if grads is None:
        return []

    saliency = tf.reduce_mean(tf.abs(grads), axis=-1).numpy()
    if saliency.ndim != 4:
        return []

    frame_saliency = saliency[0]
    heatmaps = []
    for frame_map in frame_saliency:
        heatmap_uint8 = _normalize_heatmap_to_uint8(frame_map, target_size=target_size)
        heatmaps.append(_serialize_heatmap(heatmap_uint8))

    return heatmaps


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


def _build_model_inputs(model, image_tensor: np.ndarray, fft_tensor: np.ndarray):
    """
    Готовит входы для моделей с несколькими входами.

    Args:
        model: Модель Keras
        image_tensor: Тензор оригинального изображения (1, 256, 256, 3)
        fft_tensor: Тензор FFT изображения (1, 104, 104, 3)

    Returns:
        Список входных тензоров в правильном порядке
    """
    inputs = getattr(model, "inputs", None) or []

    # Если модель принимает один вход, возвращаем только изображение
    if len(inputs) <= 1:
        return image_tensor

    # Для модели с двумя входами (изображение + FFT)
    if len(inputs) == 2:
        # Определяем порядок входов по их размерам
        input_shapes = [tuple(inp.shape) for inp in inputs]

        prepared_inputs = []
        for shape in input_shapes:
            # Вход с большим размером - это оригинальное изображение (256x256)
            if len(shape) == 4 and shape[1] is not None and shape[1] > 150:
                prepared_inputs.append(image_tensor)
            # Вход с меньшим размером - это FFT (104x104)
            elif len(shape) == 4 and shape[1] is not None and shape[1] < 150:
                prepared_inputs.append(fft_tensor)
            else:
                # Fallback: добавляем нулевой тензор нужной формы
                batch_size = image_tensor.shape[0]
                fallback_shape = [batch_size]
                for dim in shape[1:]:
                    fallback_shape.append(int(dim) if dim is not None else 1)
                prepared_inputs.append(np.zeros(tuple(fallback_shape), dtype="float32"))

        return prepared_inputs

    # Универсальный fallback для моделей с более чем 2 входами
    prepared_inputs = []
    batch_size = image_tensor.shape[0]

    for model_input in inputs:
        shape = tuple(model_input.shape)

        # Типичный вход изображения: (None, H, W, C)
        if len(shape) == 4:
            if shape[1] is not None and shape[1] > 150:
                prepared_inputs.append(image_tensor)
            else:
                prepared_inputs.append(fft_tensor)
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


def _select_model_for_content(content_type: str, model_name: Optional[str]) -> str:
    available_models = get_available_models(content_type=content_type)
    if not available_models:
        raise ValueError(f"No {content_type} models loaded")

    if model_name:
        if model_name not in available_models:
            raise ValueError(
                f"Model '{model_name}' is not available for content type '{content_type}'. "
                f"Available: {', '.join(available_models)}"
            )
        return model_name

    default_model = AppConfig.DEFAULT_MODEL_BY_CONTENT.get(content_type)
    if default_model and default_model in available_models:
        return default_model

    return available_models[0]


def _infer_audio_model_specs(model) -> Dict[str, int]:
    input_shape = getattr(model, "input_shape", None)

    segment_len = AUDIO_DEFAULT_SEGMENT_LEN
    if isinstance(input_shape, tuple) and len(input_shape) >= 3:
        inferred = _to_int_dim(input_shape[-1], AUDIO_DEFAULT_SEGMENT_LEN)
        segment_len = inferred if inferred > 0 else AUDIO_DEFAULT_SEGMENT_LEN
    elif isinstance(input_shape, list) and input_shape:
        first = input_shape[0]
        if isinstance(first, tuple) and len(first) >= 3:
            inferred = _to_int_dim(first[-1], AUDIO_DEFAULT_SEGMENT_LEN)
            segment_len = inferred if inferred > 0 else AUDIO_DEFAULT_SEGMENT_LEN

    hop_len = max(1, int(segment_len * 0.5))

    return {
        "sample_rate": AUDIO_SAMPLE_RATE,
        "segment_len": segment_len,
        "hop_len": hop_len,
    }


def _split_audio_segments(audio: np.ndarray, segment_len: int, hop_len: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((1, segment_len), dtype="float32")

    if audio.shape[0] < segment_len:
        padded = np.pad(audio, (0, segment_len - audio.shape[0]))
        return np.expand_dims(padded.astype("float32"), axis=0)

    segments = []
    start = 0
    while start + segment_len <= audio.shape[0]:
        segments.append(audio[start:start + segment_len])
        start += hop_len

    if not segments:
        segments = [audio[:segment_len]]

    return np.asarray(segments, dtype="float32")


def preprocess_audio_for_model(audio_path: str, model) -> Tuple[np.ndarray, Dict[str, float]]:
    if librosa is None:
        raise ValueError("Audio dependencies are not installed. Add librosa and soundfile.")

    specs = _infer_audio_model_specs(model)
    sample_rate = specs["sample_rate"]
    segment_len = specs["segment_len"]
    hop_len = specs["hop_len"]

    try:
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise ValueError(f"Cannot process audio: {str(e)}")

    audio = np.asarray(audio, dtype="float32")
    if audio.size == 0:
        raise ValueError("Cannot process audio: empty waveform")

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-9:
        audio = audio / peak

    segments = _split_audio_segments(audio, segment_len=segment_len, hop_len=hop_len)
    batch = np.expand_dims(segments, axis=0)

    return batch, {
        "sample_rate": sample_rate,
        "segment_len": segment_len,
        "segments_count": int(segments.shape[0]),
        "duration_seconds": float(audio.shape[0]) / float(sample_rate),
    }


def _predict_with_loaded_model(model, image_tensor: np.ndarray, fft_tensor: np.ndarray) -> float:
    model_inputs = _build_model_inputs(model, image_tensor, fft_tensor)
    preds = model.predict(model_inputs, verbose=0)
    return _extract_probability(preds)


def predict_deepfake_probability(
    image_path: str,
    model_name: Optional[str] = None,
    threshold: float = 0.5,
    include_heatmap: bool = True,
) -> dict:
    """Возвращает результат детекции дипфейка."""
    selected_model = _select_model_for_content("photo", model_name)

    model = _model_manager.get_model(selected_model)
    if model is None:
        raise ValueError(f"Model '{selected_model}' not found or not loaded")

    img_tensor, fft_tensor, specs = _preprocess_photo_for_model(image_path, model)
    model_inputs = _compose_model_inputs(
        model,
        rgb_tensor=img_tensor,
        fft_tensor=fft_tensor,
        rgb_index=specs["rgb_index"],
        fft_index=specs["fft_index"],
    )
    preds = model.predict(model_inputs, verbose=0)
    prob = _extract_probability(preds)

    confidence = prob if prob >= threshold else (1.0 - prob)
    label = "deepfake" if prob >= threshold else "real"

    result = {
        "probability": prob,
        "percent": round(prob * 100.0, 4),
        "label": label,
        "confidence": confidence,
        "model_used": selected_model,
        "threshold": threshold,
    }

    if include_heatmap:
        heatmap = _generate_photo_gradcam(model, model_inputs, target_size=specs["rgb_size"])
        result["heatmap"] = heatmap
    return result


def predict_video_deepfake_probability(
    video_path: str,
    model_name: Optional[str] = None,
    threshold: float = 0.5,
    max_frames: int = 32,
    include_heatmap: bool = True,
) -> dict:
    """Возвращает результат детекции дипфейка для видео."""
    selected_model = _select_model_for_content("video", model_name)

    model = _model_manager.get_model(selected_model)
    if model is None:
        raise ValueError(f"Model '{selected_model}' not found or not loaded")

    rgb_batch, fft_batch, specs, frames_count = _extract_video_inputs(video_path, model, max_frames=max_frames)
    model_inputs = _compose_model_inputs(
        model,
        rgb_tensor=rgb_batch,
        fft_tensor=fft_batch,
        rgb_index=specs["rgb_index"],
        fft_index=specs["fft_index"],
    )

    preds = model.predict(model_inputs, verbose=0)
    prob = _extract_probability(preds)
    confidence = prob if prob >= threshold else (1.0 - prob)
    label = "deepfake" if prob >= threshold else "real"

    result = {
        "probability": prob,
        "percent": round(prob * 100.0, 4),
        "label": label,
        "confidence": confidence,
        "model_used": selected_model,
        "threshold": threshold,
        "frames_analyzed": frames_count,
    }

    if include_heatmap:
        result["frame_heatmaps"] = _generate_video_input_saliency_heatmaps(
            model,
            model_inputs,
            rgb_index=specs["rgb_index"],
            target_size=specs["rgb_size"],
        )

    return result


def predict_audio_deepfake_probability(
    audio_path: str,
    model_name: Optional[str] = None,
    threshold: float = 0.5,
) -> dict:
    """Возвращает результат детекции дипфейка для аудио."""
    selected_model = _select_model_for_content("audio", model_name)

    model = _model_manager.get_model(selected_model)
    if model is None:
        raise ValueError(f"Model '{selected_model}' not found or not loaded")

    audio_batch, meta = preprocess_audio_for_model(audio_path, model)
    preds = model.predict(audio_batch, verbose=0)
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
        "sample_rate": meta["sample_rate"],
        "duration_seconds": round(meta["duration_seconds"], 4),
        "segments_analyzed": meta["segments_count"],
        "segment_length_seconds": round(meta["segment_len"] / meta["sample_rate"], 4),
    }


