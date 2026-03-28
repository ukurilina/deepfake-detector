import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import io
import base64

import keras
from keras import layers

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover
    FaceAnalysis = None

from app.config import AppConfig
from app.models_manager import ModelManager
from app.video_utils import get_key_frames

# Foolbox is optional; protection endpoints will validate availability.
try:
    import foolbox as fb
except Exception:  # pragma: no cover
    fb = None

# Инициализация менеджера моделей
_model_manager = ModelManager()

AUDIO_SAMPLE_RATE = 16000
AUDIO_DEFAULT_SEGMENT_LEN = 3 * AUDIO_SAMPLE_RATE
AUDIO_DEFAULT_HOP_LEN = int(1.5 * AUDIO_SAMPLE_RATE)

_face_analyzer = None
_face_analyzer_failed = False
_VIDEO_ALIGN_TEMPLATE = np.array(
    [
        [38.3, 51.7],
        [73.5, 51.5],
        [56.0, 71.7],
        [41.5, 92.4],
        [70.7, 92.2],
    ],
    dtype=np.float32,
)


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


def prepare_image_for_model(image_path: str, model) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Prepares image inputs for the given Keras model.

    Returns:
        (model_inputs, meta) where model_inputs is a list aligned with model.inputs.
    """
    specs = _infer_photo_model_specs(model)
    img = Image.open(image_path).convert("RGB")
    rgb_uint8 = np.asarray(img, dtype=np.uint8)

    # Center face for photo (as requested earlier).
    rgb_uint8 = _center_face_crop_with_insightface(rgb_uint8)

    rgb_uint8 = _resize_rgb_uint8(rgb_uint8, specs["rgb_size"])
    rgb = rgb_uint8.astype(np.float32) / 255.0

    inputs = [None] * max(1, len(getattr(model, "inputs", []) or [0]))
    if specs["fft_index"] is None:
        inputs[specs["rgb_index"]] = rgb[None, ...]
    else:
        fft = compute_fft(rgb, crop_ratio=0.4)
        fft = _resize_array_hwc(fft, specs["fft_size"])
        inputs[specs["rgb_index"]] = rgb[None, ...]
        inputs[specs["fft_index"]] = fft[None, ...]

    meta = {
        **specs,
        "rgb_preview_uint8": rgb_uint8,
    }
    return inputs, meta


def prepare_video_frames_for_model(
    video_path: str,
    model,
    max_frames: int = 24,
    frame_stride: int = 1,
    return_frames_rgb_uint8: bool = False,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Prepares video tensors exactly like inference.

    Uses existing key-frame extraction and InsightFace alignment.
    """
    # NOTE: video model expected input sizes are inferred similarly to photo models, but with TimeDistributed.
    inputs = getattr(model, "inputs", None) or []
    if len(inputs) < 1:
        raise ValueError("Video model has no inputs")

    # Infer per-frame size from first input (B, T, H, W, C)
    shape = tuple(inputs[0].shape)
    h = _to_int_dim(shape[2] if len(shape) > 2 else None, 112)
    w = _to_int_dim(shape[3] if len(shape) > 3 else None, 112)
    target_size = (h, w)

    # Grab frames
    frames = get_key_frames(video_path, num_frames=max_frames)
    # frames returned as RGB uint8
    if not frames:
        raise ValueError("Could not extract frames from video")

    # Subsample stride
    frames = frames[:: max(1, int(frame_stride))]

    # Align each frame similarly to neuro.py (InsightFace).
    aligned_frames = []
    last_transform = None
    for fr in frames:
        if AppConfig.ENABLE_FACE_ALIGN_FOR_VIDEO:
            aligned, last_transform = _align_face_frame_with_insightface(
                fr,
                target_size=target_size,
                fallback_transform=last_transform if AppConfig.REUSE_LAST_VIDEO_FACE_TRANSFORM else None,
            )
        else:
            aligned = _resize_rgb_uint8(fr, target_size)
        aligned_frames.append(aligned)

    rgb_uint8 = np.stack(aligned_frames, axis=0)
    rgb = rgb_uint8.astype(np.float32) / 255.0

    # Model in this project expects [video_rgb, video_fft]
    # FFT branch is computed from RGB.
    fft_frames = np.stack([compute_fft(f, crop_ratio=None) for f in rgb], axis=0)
    fft_frames = np.asarray(fft_frames, dtype=np.float32)

    meta = {
        "rgb_index": 0,
        "fft_index": 1 if len(inputs) > 1 else None,
        "rgb_size": target_size,
        "frames": int(rgb.shape[0]),
    }

    model_inputs = [None] * len(inputs)
    model_inputs[0] = rgb
    if meta["fft_index"] is not None:
        model_inputs[1] = fft_frames
    if return_frames_rgb_uint8:
        meta["frames_rgb_uint8"] = rgb_uint8
    return model_inputs, meta


def _build_photo_foolbox_wrapper(model, model_name: str):
    """Creates a Keras model that takes only RGB (B,H,W,3) and forwards to the original model.

    If the original model has an FFT input, we compute it via tf ops (no external deps).
    """
    specs = _infer_photo_model_specs(model)
    rgb_input = keras.Input(shape=(specs["rgb_size"][0], specs["rgb_size"][1], 3), name=f"{model_name}_rgb")

    if specs["fft_index"] is None:
        out = model(rgb_input)
    else:
        # Pure TF approximation of compute_fft wrapped into Keras layers (Keras 3 requirement).
        gray = layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x), name=f"{model_name}_fft_gray")(rgb_input)
        gray = layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name=f"{model_name}_fft_squeeze")(gray)

        def _fft2d_layer(x):
            x = tf.cast(x, tf.complex64)
            f = tf.signal.fft2d(x)
            return tf.signal.fftshift(f)

        fft2d = layers.Lambda(_fft2d_layer, name=f"{model_name}_fft2d")(gray)

        def _fft_features(f):
            mag = tf.math.log(tf.abs(f) + 1e-8)
            mag = (mag - tf.reduce_mean(mag, axis=[1, 2], keepdims=True)) / (
                tf.math.reduce_std(mag, axis=[1, 2], keepdims=True) + 1e-8
            )
            phase = tf.math.angle(f)
            return tf.stack([mag, tf.sin(phase), tf.cos(phase)], axis=-1)

        fft = layers.Lambda(_fft_features, name=f"{model_name}_fft_features")(fft2d)
        fft = layers.Lambda(lambda x: tf.image.central_crop(x, 0.4), name=f"{model_name}_fft_crop")(fft)
        fft = layers.Resizing(specs["fft_size"][0], specs["fft_size"][1], interpolation="bilinear", name=f"{model_name}_fft_resize")(fft)

        inputs = [None] * len(model.inputs)
        inputs[specs["rgb_index"]] = rgb_input
        inputs[specs["fft_index"]] = fft
        out = model(inputs)

    return keras.Model(rgb_input, out, name=f"{model_name}_foolbox")


def _build_video_foolbox_wrapper(model, model_name: str):
    # video: (B,T,H,W,3)
    inp0 = getattr(model, "inputs", None)[0]
    shape = tuple(inp0.shape)
    t_fixed = _to_int_dim(shape[1] if len(shape) > 1 else None, 0)
    h = _to_int_dim(shape[2] if len(shape) > 2 else None, 112)
    w = _to_int_dim(shape[3] if len(shape) > 3 else None, 112)
    rgb_input = keras.Input(shape=((t_fixed or None), h, w, 3), name=f"{model_name}_video_rgb")

    if len(model.inputs) < 2:
        out = model(rgb_input)
        return keras.Model(rgb_input, out, name=f"{model_name}_video_foolbox")

    # Pure TF approximation of compute_fft for a video sequence (wrapped as Keras layers).
    def _flatten_bt(x):
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        return tf.reshape(x, (b * t, h, w, 3))

    flat = layers.Lambda(
        _flatten_bt,
        output_shape=(h, w, 3),
        name=f"{model_name}_vfft_flat",
    )(rgb_input)
    gray = layers.Lambda(
        lambda x: tf.image.rgb_to_grayscale(x),
        output_shape=(h, w, 1),
        name=f"{model_name}_vfft_gray",
    )(flat)
    gray = layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1),
        output_shape=(h, w),
        name=f"{model_name}_vfft_squeeze",
    )(gray)

    def _fft2d_layer(x):
        x = tf.cast(x, tf.complex64)
        f = tf.signal.fft2d(x)
        return tf.signal.fftshift(f)

    fft2d = layers.Lambda(
        _fft2d_layer,
        output_shape=(h, w),
        name=f"{model_name}_vfft2d",
    )(gray)

    def _fft_features(f):
        mag = tf.math.log(tf.abs(f) + 1e-8)
        mag = (mag - tf.reduce_mean(mag, axis=[1, 2], keepdims=True)) / (
            tf.math.reduce_std(mag, axis=[1, 2], keepdims=True) + 1e-8
        )
        phase = tf.math.angle(f)
        return tf.stack([mag, tf.sin(phase), tf.cos(phase)], axis=-1)

    fft_flat = layers.Lambda(
        _fft_features,
        output_shape=(h, w, 3),
        name=f"{model_name}_vfft_features",
    )(fft2d)

    def _unflatten_bt(x):
        b = tf.shape(rgb_input)[0]
        t = tf.shape(rgb_input)[1]
        return tf.reshape(x, (b, t, h, w, 3))

    fft = layers.Lambda(
        _unflatten_bt,
        output_shape=((t_fixed or None), h, w, 3),
        name=f"{model_name}_vfft_unflat",
    )(fft_flat)
    out = model([rgb_input, fft])
    return keras.Model(rgb_input, out, name=f"{model_name}_video_foolbox")


_FOOLBOX_WRAPPERS: Dict[str, keras.Model] = {}


def get_foolbox_wrapper_model(model_name: str) -> keras.Model:
    """Returns (and caches) a wrapper model suitable for Foolbox attacks."""
    if model_name in _FOOLBOX_WRAPPERS:
        return _FOOLBOX_WRAPPERS[model_name]

    model = _model_manager.get_model(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' is not loaded")
    content_type = AppConfig.MODEL_REGISTRY.get(model_name, {}).get("content_type")
    if content_type == "video":
        wrapper = _build_video_foolbox_wrapper(model, model_name)
    else:
        wrapper = _build_photo_foolbox_wrapper(model, model_name)
    _FOOLBOX_WRAPPERS[model_name] = wrapper
    return wrapper


def protect_with_foolbox(
    content_type: str,
    input_path: str,
    model_name: str,
    attack: str,
    eps: float,
    steps: int,
    frame_stride: int = 1,
    max_frames: int = 24,
) -> Tuple[str, Dict[str, Any]]:
    """Applies Foolbox-based protection and returns (output_path, meta)."""
    if fb is None:
        raise RuntimeError("Foolbox is not installed")

    wrapper = get_foolbox_wrapper_model(model_name)
    fmodel = fb.TensorFlowModel(wrapper, bounds=(0.0, 1.0))

    if attack == "pgd":
        attacker = fb.attacks.LinfPGD(steps=max(1, int(steps)))
    else:
        attacker = fb.attacks.FGSM()

    # Prepare input for wrapper (only RGB)
    if content_type == "video":
        model_inputs, meta = prepare_video_frames_for_model(
            input_path,
            _model_manager.get_model(model_name),
            max_frames=max_frames,
            frame_stride=frame_stride,
        )
        x = np.asarray(model_inputs[0], dtype=np.float32)
    else:
        # Keep original size for output: protection must not change image resolution.
        original_pil = Image.open(input_path).convert("RGB")
        original_size = original_pil.size  # (W, H)

        model_inputs, meta = prepare_image_for_model(input_path, _model_manager.get_model(model_name))
        rgb_index = int(meta.get("rgb_index", 0))
        x = np.asarray(model_inputs[rgb_index], dtype=np.float32)
        meta["original_size"] = (int(original_size[1]), int(original_size[0]))  # (H, W)

    if x.ndim == 3:
        x = x[None, ...]

    # Foolbox/EagerPy chooses backend based on the tensor type.
    # Use TF tensors to enable gradients.
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    labels = tf.zeros((tf.shape(x_tf)[0],), dtype=tf.int64)
    # Foolbox may return (raw_advs, clipped_advs, success) or just advs depending on attack.
    adv = attacker(fmodel, x_tf, labels, epsilons=float(eps))
    # Many Foolbox attacks return (raw_advs, clipped_advs, success)
    if isinstance(adv, (tuple, list)):
        if len(adv) >= 2:
            adv = adv[1]
        else:
            adv = adv[0]

    # Convert output to TF tensor reliably.
    try:
        import eagerpy as ep
        adv_ep = ep.astensor(adv)
        adv_tf = adv_ep.raw
    except Exception:
        adv_tf = adv
        if hasattr(adv_tf, "raw"):
            adv_tf = adv_tf.raw

    adv_tf = tf.convert_to_tensor(adv_tf, dtype=tf.float32)
    adv_tf = tf.clip_by_value(adv_tf, 0.0, 1.0)
    adv_np = adv_tf.numpy()

    # Save output
    from app.protection import _new_protect_id, _create_protect_dir, _write_json  # local import to avoid cycles
    protect_id = _new_protect_id()
    out_dir = _create_protect_dir(protect_id)

    if content_type == "video":
        if cv2 is None:
            raise RuntimeError("OpenCV is required for video protection")
        frames_uint8 = (adv_np * 255.0).round().astype(np.uint8)
        h, w = frames_uint8.shape[1:3]
        out_path = os.path.join(out_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 15.0, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Could not open VideoWriter")
        try:
            for frame in frames_uint8:
                writer.write(frame[:, :, ::-1])
        finally:
            writer.release()

        meta_out = {
            "created_at": time.time(),
            "content_type": "video",
            "model": model_name,
            "attack": attack,
            "eps": float(eps),
            "steps": int(steps),
            "frame_stride": int(frame_stride),
            "max_frames": int(max_frames),
            "frames": int(frames_uint8.shape[0]),
            "output_file": "output.mp4",
        }
    else:
        out_path = os.path.join(out_dir, "output.png")
        # Ensure HWC
        img_f = adv_np[0] if adv_np.ndim == 4 else adv_np
        if img_f.ndim != 3 or img_f.shape[-1] != 3:
            raise RuntimeError(f"Unexpected adversarial image shape: {img_f.shape}")
        img_uint8 = (img_f * 255.0).round().astype(np.uint8)

        out_img = Image.fromarray(img_uint8, mode="RGB")
        # Resize back to original resolution.
        original_hw = meta.get("original_size")
        if original_hw and isinstance(original_hw, (tuple, list)) and len(original_hw) == 2:
            oh, ow = int(original_hw[0]), int(original_hw[1])
            if oh > 0 and ow > 0 and (out_img.size[0] != ow or out_img.size[1] != oh):
                out_img = out_img.resize((ow, oh), resample=Image.BILINEAR)

        out_img.save(out_path)
        meta_out = {
            "created_at": time.time(),
            "content_type": "photo",
            "model": model_name,
            "attack": attack,
            "eps": float(eps),
            "steps": int(steps),
            "output_file": "output.png",
            "original_size": list(original_hw) if original_hw else None,
        }

    _write_json(os.path.join(out_dir, "meta.json"), meta_out)
    meta_out["protect_id"] = protect_id
    meta_out["output_path"] = out_path
    return out_path, meta_out


def _get_face_analyzer():
    global _face_analyzer, _face_analyzer_failed

    if _face_analyzer is not None:
        return _face_analyzer
    if _face_analyzer_failed or FaceAnalysis is None:
        return None

    try:
        analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        analyzer.prepare(ctx_id=-1, det_size=(640, 640))
        _face_analyzer = analyzer
        return _face_analyzer
    except Exception:
        _face_analyzer_failed = True
        return None


def _center_face_crop_with_insightface(image_rgb: np.ndarray) -> np.ndarray:
    analyzer = _get_face_analyzer()
    if analyzer is None:
        return image_rgb

    try:
        faces = analyzer.get(image_rgb[:, :, ::-1])
    except Exception:
        return image_rgb

    if not faces:
        return image_rgb

    best_bbox = None
    best_area = -1.0
    for face in faces:
        bbox = getattr(face, "bbox", None)
        if bbox is None and isinstance(face, dict):
            bbox = face.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best_bbox = (x1, y1, x2, y2)

    if best_bbox is None:
        return image_rgb

    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = best_bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    face_side = max(x2 - x1, y2 - y1)
    crop_side = max(32.0, face_side * 1.55)
    half = 0.5 * crop_side

    left = int(max(0, round(cx - half)))
    right = int(min(w, round(cx + half)))
    top = int(max(0, round(cy - half)))
    bottom = int(min(h, round(cy + half)))

    if right - left < 8 or bottom - top < 8:
        return image_rgb

    return image_rgb[top:bottom, left:right]


def _resize_rgb_uint8(image_rgb: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image_rgb.astype("uint8"), mode="RGB")
    return np.asarray(pil.resize((target_size[1], target_size[0])), dtype="uint8")


def _align_face_frame_with_insightface(
    image_rgb: np.ndarray,
    target_size: Tuple[int, int],
    fallback_transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Aligns a face to the training template used in neuro.py, with safe fallbacks."""
    if cv2 is None:
        return _resize_rgb_uint8(image_rgb, target_size), None

    analyzer = _get_face_analyzer()
    best_transform = None

    if analyzer is not None:
        try:
            faces = analyzer.get(image_rgb[:, :, ::-1])
        except Exception:
            faces = []

        if faces:
            best_face = None
            best_area = -1.0
            for face in faces:
                bbox = getattr(face, "bbox", None)
                if bbox is None and isinstance(face, dict):
                    bbox = face.get("bbox")
                if bbox is None or len(bbox) < 4:
                    continue

                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area > best_area:
                    best_area = area
                    best_face = face

            if best_face is not None:
                kps = getattr(best_face, "kps", None)
                if kps is None and isinstance(best_face, dict):
                    kps = best_face.get("kps")

                if kps is not None:
                    try:
                        src = np.asarray(kps, dtype=np.float32)
                        if src.shape[0] >= 5:
                            src = src[:5]
                        target_h, target_w = target_size
                        scale = np.array([target_w / 112.0, target_h / 112.0], dtype=np.float32)
                        dst = _VIDEO_ALIGN_TEMPLATE * scale
                        best_transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
                    except Exception:
                        best_transform = None

    if best_transform is None and fallback_transform is not None:
        best_transform = fallback_transform

    if best_transform is not None:
        try:
            target_h, target_w = target_size
            aligned = cv2.warpAffine(
                image_rgb,
                best_transform,
                (target_w, target_h),
                borderValue=0,
            )
            return aligned.astype("uint8"), best_transform
        except Exception:
            pass

    return _resize_rgb_uint8(image_rgb, target_size), None


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


def _encode_rgb_uint8_to_data_url(image_rgb: np.ndarray) -> str:
    try:
        img = Image.fromarray(image_rgb.astype("uint8"), mode="RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""


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
        frame_uint8 = frame.astype("uint8")
        if AppConfig.ENABLE_FACE_ALIGN_FOR_VIDEO:
            aligned_uint8, _ = _align_face_frame_with_insightface(frame_uint8, target_size=target_size)
        else:
            aligned_uint8 = _resize_rgb_uint8(frame_uint8, target_size=target_size)

        arr = np.asarray(aligned_uint8, dtype="float32") / 255.0
        fft_arr = compute_fft(arr, crop_ratio=None)
        return np.expand_dims(arr, axis=0), np.expand_dims(fft_arr, axis=0)
    except Exception as e:
        raise ValueError(f"Cannot process video frame: {str(e)}")


def _preprocess_photo_for_model(image_path: str, model) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any], str]:
    specs = _infer_photo_model_specs(model)
    rgb_size = specs["rgb_size"]
    fft_size = specs["fft_size"]

    try:
        with Image.open(image_path).convert("RGB") as im:
            arr_rgb = np.asarray(im, dtype="uint8")
            arr_rgb = _center_face_crop_with_insightface(arr_rgb)
            im_face = Image.fromarray(arr_rgb, mode="RGB").resize((rgb_size[1], rgb_size[0]))
            arr = np.asarray(im_face, dtype="float32") / 255.0
    except Exception as e:
        raise ValueError(f"Cannot process image: {str(e)}")

    fft_arr = None
    if specs["fft_index"] is not None:
        use_crop = fft_size[0] < rgb_size[0] or fft_size[1] < rgb_size[1]
        fft_arr = compute_fft(arr, crop_ratio=0.4 if use_crop else None)
        fft_arr = _resize_array_hwc(fft_arr, fft_size)

    preview_uint8 = np.clip(arr * 255.0, 0, 255).astype("uint8")
    source_image_data_url = _encode_rgb_uint8_to_data_url(preview_uint8)

    img_batch = np.expand_dims(arr, axis=0)
    fft_batch = np.expand_dims(fft_arr, axis=0) if fft_arr is not None else None
    return img_batch, fft_batch, specs, source_image_data_url


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
    last_transform = None

    for frame in frames:
        frame_uint8 = frame.astype("uint8")
        if AppConfig.ENABLE_FACE_ALIGN_FOR_VIDEO:
            fallback_transform = last_transform if AppConfig.REUSE_LAST_VIDEO_FACE_TRANSFORM else None
            aligned_uint8, used_transform = _align_face_frame_with_insightface(
                frame_uint8,
                target_size=specs["rgb_size"],
                fallback_transform=fallback_transform,
            )
            if used_transform is not None:
                last_transform = used_transform
        else:
            aligned_uint8 = _resize_rgb_uint8(frame_uint8, target_size=specs["rgb_size"])

        rgb_arr = np.asarray(aligned_uint8, dtype="float32") / 255.0
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


def _calibrate_probability(raw_prob: float, model_name: str) -> float:
    pivot = float(AppConfig.CALIBRATION_PIVOT_BY_MODEL.get(model_name, 0.5))
    pivot = max(1e-6, min(1.0 - 1e-6, pivot))

    prob = max(0.0, min(1.0, float(raw_prob)))
    if prob >= pivot:
        scaled = 0.5 + 0.5 * ((prob - pivot) / (1.0 - pivot))
    else:
        scaled = 0.5 * (prob / pivot)

    return max(0.0, min(1.0, float(scaled)))


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

    img_tensor, fft_tensor, specs, source_image_data_url = _preprocess_photo_for_model(image_path, model)
    model_inputs = _compose_model_inputs(
        model,
        rgb_tensor=img_tensor,
        fft_tensor=fft_tensor,
        rgb_index=specs["rgb_index"],
        fft_index=specs["fft_index"],
    )
    preds = model.predict(model_inputs, verbose=0)
    raw_prob = _extract_probability(preds)
    prob = _calibrate_probability(raw_prob, selected_model)
    label = "deepfake" if prob >= threshold else "real"

    result = {
        "probability": prob,
        "percent": round(prob * 100.0, 4),
        "label": label,
        "model_used": selected_model,
        "threshold": threshold,
    }

    if include_heatmap:
        heatmap = _generate_photo_gradcam(model, model_inputs, target_size=specs["rgb_size"])
        result["heatmap"] = heatmap

    if source_image_data_url:
        # Frontend uses this preview for region overlay, keeping coordinates aligned with heatmap space.
        result["source_image_data_url"] = source_image_data_url
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
    raw_prob = _extract_probability(preds)
    prob = _calibrate_probability(raw_prob, selected_model)
    label = "deepfake" if prob >= threshold else "real"

    result = {
        "probability": prob,
        "percent": round(prob * 100.0, 4),
        "label": label,
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
    raw_prob = _extract_probability(preds)
    prob = _calibrate_probability(raw_prob, selected_model)
    label = "deepfake" if prob >= threshold else "real"

    return {
        "probability": prob,
        "percent": round(prob * 100.0, 4),
        "label": label,
        "model_used": selected_model,
        "threshold": threshold,
        "sample_rate": meta["sample_rate"],
        "duration_seconds": round(meta["duration_seconds"], 4),
        "segments_analyzed": meta["segments_count"],
        "segment_length_seconds": round(meta["segment_len"] / meta["sample_rate"], 4),
    }


