"""
Утилиты для обработки видео-файлов.
Требует: opencv-python (cv2)
"""

import os
import tempfile
from typing import Optional, List, Tuple
import numpy as np

try:
    import cv2
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
FRAME_EXTRACTION_INTERVAL = 5  # Извлекаем каждый 5-й кадр


def is_video_file(file_path: str) -> bool:
    """Проверяет, является ли файл видео."""
    if not os.path.exists(file_path):
        return False

    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_VIDEO_EXTENSIONS


def extract_frames(
    video_path: str,
    interval: int = FRAME_EXTRACTION_INTERVAL,
    max_frames: Optional[int] = None
) -> List[np.ndarray]:
    """
    Извлекает фреймы из видеофайла.

    Args:
        video_path: Путь к видеофайлу
        interval: Интервал между извлекаемыми фреймами (каждый N-й кадр)
        max_frames: Максимальное количество фреймов для извлечения

    Returns:
        Список массивов NumPy с фреймами

    Raises:
        ValueError: Если видеофайл не может быть открыт
    """

    if not VIDEO_SUPPORT:
        raise ValueError("Video support not available. Install opencv-python")

    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    frames = []

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Извлекаем каждый interval-й кадр
            if frame_count % interval == 0:
                # Преобразуем BGR в RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        cap.release()

        if not frames:
            raise ValueError("No frames could be extracted from the video")

        return frames

    except Exception as e:
        raise ValueError(f"Failed to extract frames: {str(e)}")


def get_key_frames(
    video_path: str,
    num_frames: int = 5
) -> List[np.ndarray]:
    """
    Извлекает ключевые фреймы из видео (первый, последний и промежуточные).

    Args:
        video_path: Путь к видеофайлу
        num_frames: Количество ключевых фреймов для извлечения

    Returns:
        Список ключевых фреймов
    """

    if not VIDEO_SUPPORT:
        raise ValueError("Video support not available. Install opencv-python")

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        key_frames = []

        # Определяем индексы ключевых фреймов
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                key_frames.append(frame_rgb)

        cap.release()

        if not key_frames:
            raise ValueError("No key frames could be extracted")

        return key_frames

    except Exception as e:
        raise ValueError(f"Failed to extract key frames: {str(e)}")


def save_frame_to_temp_file(frame: np.ndarray) -> str:
    """
    Сохраняет фрейм в временный файл.

    Args:
        frame: Массив NumPy с фреймом (RGB)

    Returns:
        Путь к временному файлу
    """
    try:
        # Преобразуем RGB обратно в BGR для OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, frame_bgr)
            return tmp.name

    except Exception as e:
        raise ValueError(f"Failed to save frame: {str(e)}")


def get_video_info(video_path: str) -> dict:
    """
    Получает информацию о видеофайле.

    Args:
        video_path: Путь к видеофайлу

    Returns:
        Dict с информацией о видео
    """

    if not VIDEO_SUPPORT:
        return {"error": "Video support not available"}

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        info = {
            "path": video_path,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }

        cap.release()
        return info

    except Exception as e:
        return {"error": str(e)}
