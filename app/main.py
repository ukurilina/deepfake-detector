import os
import tempfile
import mimetypes
import base64
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Literal
import urllib.request
import re
from html import unescape
import shutil

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, AnyHttpUrl, Field

from app.config import (
    AppConfig,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from app.model import (
    predict_deepfake_probability,
    predict_video_deepfake_probability,
    predict_audio_deepfake_probability,
    initialize_models,
    get_available_models,
)
from app.models_manager import ModelManager

# Инициализация FastAPI приложения
app = FastAPI(
    title=AppConfig.TITLE,
    version=AppConfig.VERSION,
    description=AppConfig.DESCRIPTION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Добавляем CORS middleware для поддержки кросс-доменных запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _is_temp_cleanup_candidate(path: Path) -> bool:
    """Определяет, можно ли удалять артефакт из temp автоматически."""
    if path.name in {".gitkeep", "git_report.txt"}:
        return False

    if path.is_dir():
        return path.name.startswith("url_")

    if not path.is_file():
        return False

    if path.name.startswith("tmp") or path.name.startswith("direct_media"):
        return True

    return path.suffix.lower() in (SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS)


def _is_allowed_content_type(content_type: str) -> bool:
    return content_type in AppConfig.CONTENT_TYPES


def _resolve_content_type(
    explicit_content_type: Optional[str],
    file_content_type: Optional[str],
    extension: str,
) -> str:
    if explicit_content_type:
        if not _is_allowed_content_type(explicit_content_type):
            raise HTTPException(status_code=400, detail=f"Unsupported content_type '{explicit_content_type}'")
        return explicit_content_type

    if extension in SUPPORTED_IMAGE_EXTENSIONS or (file_content_type or "").startswith("image/"):
        return "photo"
    if extension in SUPPORTED_VIDEO_EXTENSIONS or (file_content_type or "").startswith("video/"):
        return "video"
    if extension in SUPPORTED_AUDIO_EXTENSIONS or (file_content_type or "").startswith("audio/"):
        return "audio"

    raise HTTPException(status_code=415, detail="Cannot detect content type for uploaded file")


def _ensure_extension_matches_content_type(content_type: str, extension: str) -> None:
    if not extension:
        return

    if content_type == "photo" and extension not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension '{extension}'. Allowed: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}",
        )


def _resolve_threshold(content_type: str, threshold: Optional[float]) -> float:
    if threshold is None:
        return float(AppConfig.DEFAULT_THRESHOLD_BY_CONTENT.get(content_type, 0.5))
    return float(threshold)

    if content_type == "video" and extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video extension '{extension}'. Allowed: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
        )

    if content_type == "audio" and extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio extension '{extension}'. Allowed: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}",
        )


def cleanup_temp_dir() -> int:
    """Удаляет накопившиеся временные артефакты и возвращает количество удаленных объектов."""
    removed = 0
    temp_path = Path(AppConfig.TEMP_DIR)

    if not temp_path.exists() or not temp_path.is_dir():
        return removed

    for item in temp_path.iterdir():
        if not _is_temp_cleanup_candidate(item):
            continue

        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue

    return removed


@app.on_event("startup")
async def startup_event():
    """Выполняется при старте приложения."""
    if AppConfig.TEMP_CLEANUP_ON_STARTUP:
        cleanup_temp_dir()

    # Инициализируем модели
    success = initialize_models()
    if not success:
        pass


@app.on_event("shutdown")
async def shutdown_event():
    """Выполняется при остановке приложения."""
    pass


@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Health check endpoint.
    Проверяет здоровье приложения и доступность загруженных моделей.
    """
    try:
        available_models = get_available_models()
        model_info = ModelManager().get_loaded_models_info()
        failed_models = [m for m in model_info if m.get("status") == "error"]

        health_status = {
            "status": "healthy" if available_models else "degraded",
            "models_loaded": len(available_models),
            "available_models": available_models,
            "failed_models": failed_models,
            "version": AppConfig.VERSION,
            "gpu_enabled": AppConfig.USE_GPU,
        }

        if not available_models:
            return JSONResponse(health_status, status_code=503)

        return health_status

    except Exception as e:
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)},
            status_code=500
        )


@app.get("/models", response_class=JSONResponse)
async def list_models(content_type: Optional[str] = Query(None, description="Filter models by content type")):
    """
    Получить список доступных моделей.
    """
    try:
        if content_type and not _is_allowed_content_type(content_type):
            raise HTTPException(status_code=400, detail=f"Unsupported content_type '{content_type}'")

        available_models = get_available_models(content_type=content_type)
        default_model = None
        if content_type:
            configured_default = AppConfig.DEFAULT_MODEL_BY_CONTENT.get(content_type)
            default_model = configured_default if configured_default in available_models else (available_models[0] if available_models else None)

        return {
            "models": available_models,
            "count": len(available_models),
            "default_model": default_model,
            "content_type": content_type,
            "models_by_content": {
                ct: get_available_models(content_type=ct)
                for ct in AppConfig.CONTENT_TYPES
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", response_class=JSONResponse)
@app.post("/predict/image", response_class=JSONResponse)
async def detect(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Model name to use for detection"),
    content_type: Optional[str] = Query(None, description="Content type: photo|video|audio"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Classification threshold")
):
    """
    Обнаружить дипфейк в загруженном файле.

    Args:
        file: Файл изображения для анализа
        model: Имя модели (если не указано, используется первая доступная)
        threshold: Порог для классификации (0.5 по умолчанию)

    Returns:
        JSON с результатами детекции
    """

    filename = file.filename or "image"
    suffix = os.path.splitext(filename)[1].lower()
    resolved_content_type = _resolve_content_type(content_type, file.content_type, suffix)
    resolved_threshold = _resolve_threshold(resolved_content_type, threshold)

    _ensure_extension_matches_content_type(resolved_content_type, suffix)

    available_models = get_available_models(content_type=resolved_content_type)
    if not available_models:
        raise HTTPException(status_code=503, detail=f"No '{resolved_content_type}' models loaded. Service unavailable.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > AppConfig.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {AppConfig.MAX_FILE_SIZE / 1024 / 1024:.0f} MB"
        )

    if model and model not in available_models:
        raise HTTPException(status_code=404, detail=f"Model '{model}' is not available for '{resolved_content_type}'")

    try:
        default_suffix = ".jpg"
        if resolved_content_type == "video":
            default_suffix = ".mp4"
        elif resolved_content_type == "audio":
            default_suffix = ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or default_suffix, dir=AppConfig.TEMP_DIR) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name

        try:
            if resolved_content_type == "photo":
                return predict_deepfake_probability(
                    tmp_path,
                    model_name=model,
                    threshold=resolved_threshold,
                )

            if resolved_content_type == "video":
                return predict_video_deepfake_probability(
                    tmp_path,
                    model_name=model,
                    threshold=resolved_threshold,
                )

            if resolved_content_type == "audio":
                return predict_audio_deepfake_probability(
                    tmp_path,
                    model_name=model,
                    threshold=resolved_threshold,
                )

            raise HTTPException(status_code=415, detail="Unsupported content type")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(exc)}")


class UrlDetectRequest(BaseModel):
    url: AnyHttpUrl
    content_type: Literal["photo", "video", "audio"]
    model: Optional[str] = None
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


def _validate_model_selection(model: Optional[str], content_type: str) -> list:
    available_models = get_available_models(content_type=content_type)
    if not available_models:
        raise HTTPException(status_code=503, detail=f"No '{content_type}' models loaded. Service unavailable.")

    if model and model not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not available for '{content_type}'"
        )

    return available_models


def _detect_image_from_path(image_path: str, model: Optional[str], threshold: float) -> dict:
    _validate_model_selection(model, content_type="photo")

    try:
        return predict_deepfake_probability(
            image_path,
            model_name=model,
            threshold=threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(exc)}")


def _detect_video_from_path(video_path: str, model: Optional[str], threshold: float) -> dict:
    _validate_model_selection(model, content_type="video")

    try:
        return predict_video_deepfake_probability(
            video_path,
            model_name=model,
            threshold=threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(exc)}")


def _detect_audio_from_path(audio_path: str, model: Optional[str], threshold: float) -> dict:
    _validate_model_selection(model, content_type="audio")

    try:
        return predict_audio_deepfake_probability(
            audio_path,
            model_name=model,
            threshold=threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(exc)}")


def _is_supported_by_ytdlp(target_url: str) -> bool:
    try:
        from yt_dlp.extractor import gen_extractors
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"yt-dlp is not available: {exc}")

    parsed = urlparse(target_url)
    if parsed.scheme not in {"http", "https"}:
        return False

    for extractor in gen_extractors():
        ie_name = getattr(extractor, "IE_NAME", "")
        if ie_name == "generic":
            continue

        suitable = getattr(extractor, "suitable", None)
        if callable(suitable) and suitable(target_url):
            return True

    return False


def _extract_image_url_from_info(info: dict) -> Optional[str]:
    if not isinstance(info, dict):
        return None

    entries = info.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            candidate = _extract_image_url_from_info(entry)
            if candidate:
                return candidate

    direct_url = info.get("url")
    if isinstance(direct_url, str) and direct_url.startswith(("http://", "https://")):
        return direct_url

    thumbnails = info.get("thumbnails") or []
    if isinstance(thumbnails, list) and thumbnails:
        thumb_url = thumbnails[-1].get("url")
        if isinstance(thumb_url, str) and thumb_url.startswith(("http://", "https://")):
            return thumb_url

    return None


def _extract_image_url_from_html(page_url: str) -> Optional[str]:
    request = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(charset, errors="ignore")

    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
    ]

    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            candidate = unescape(match.group(1)).strip()
            if candidate.startswith(("http://", "https://")):
                return candidate

    return None


def _download_direct_media(media_url: str, work_dir: str) -> str:
    request = urllib.request.Request(
        media_url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get_content_type()
        ext = mimetypes.guess_extension(content_type or "") or ".jpg"
        safe_ext = ext if len(ext) <= 6 else ".jpg"
        file_path = os.path.join(work_dir, f"direct_media{safe_ext}")
        with open(file_path, "wb") as output_file:
            output_file.write(response.read())

    return file_path


def _download_url_with_ytdlp(target_url: str) -> str:
    try:
        import yt_dlp
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"yt-dlp is not available: {exc}")

    os.makedirs(AppConfig.TEMP_DIR, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="url_", dir=AppConfig.TEMP_DIR)

    ydl_opts = {
        "outtmpl": os.path.join(work_dir, "%(id)s.%(ext)s"),
        "format": "best/bestvideo+bestaudio/bestaudio",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(target_url, download=True)

        candidates = [
            p for p in Path(work_dir).glob("**/*")
            if p.is_file() and p.suffix not in {".part", ".ytdl"}
        ]
        if not candidates:
            raise HTTPException(status_code=400, detail="yt-dlp could not download a file from the provided URL")

        downloaded = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(downloaded)

    except HTTPException:
        raise
    except Exception as exc:
        error_text = str(exc)
        if "No video formats found" in error_text:
            # Some pages (e.g. Pinterest image pins) expose only image metadata.
            try:
                with yt_dlp.YoutubeDL({**ydl_opts, "skip_download": True}) as ydl:
                    info = ydl.extract_info(target_url, download=False)

                image_url = _extract_image_url_from_info(info)
                if image_url:
                    return _download_direct_media(image_url, work_dir)
            except Exception:
                pass

            try:
                image_url = _extract_image_url_from_html(target_url)
                if image_url:
                    return _download_direct_media(image_url, work_dir)
            except Exception:
                pass

            raise HTTPException(
                status_code=415,
                detail="The URL is reachable, but downloadable media formats were not found",
            )

        raise HTTPException(status_code=400, detail=f"Failed to download URL with yt-dlp: {exc}")


def _cleanup_downloaded_file(file_path: str) -> None:
    try:
        path = Path(file_path)
        parent = path.parent
        if path.exists():
            path.unlink()
        if parent.exists() and parent.is_dir() and parent.name.startswith("url_"):
            shutil.rmtree(parent, ignore_errors=True)
    except Exception:
        pass


def _build_image_data_url(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        return None

    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    try:
        with open(file_path, "rb") as src:
            raw = src.read()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return None


@app.post("/predict/url", response_class=JSONResponse)
async def detect_by_url(payload: UrlDetectRequest):
    """
    Download a file by URL using yt-dlp and route it to detector by selected content type.
    """
    target_url = str(payload.url)

    if not _is_supported_by_ytdlp(target_url):
        raise HTTPException(
            status_code=400,
            detail=(
                "URL from this site is not supported by yt-dlp. "
                "See supported sites: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md"
            ),
        )

    downloaded_file = _download_url_with_ytdlp(target_url)
    guessed_mime, _ = mimetypes.guess_type(downloaded_file)
    extension = Path(downloaded_file).suffix.lower()

    try:
        if payload.content_type == "photo":
            if not ((guessed_mime or "").startswith("image/") or extension in SUPPORTED_IMAGE_EXTENSIONS):
                raise HTTPException(status_code=415, detail="Downloaded content is not an image")
            resolved_threshold = _resolve_threshold("photo", payload.threshold)
            result = _detect_image_from_path(
                downloaded_file,
                model=payload.model,
                threshold=resolved_threshold,
            )
            source_data_url = _build_image_data_url(downloaded_file)
            if source_data_url:
                result["source_image_data_url"] = source_data_url
            return result

        if payload.content_type == "video":
            if not ((guessed_mime or "").startswith("video/") or extension in SUPPORTED_VIDEO_EXTENSIONS):
                raise HTTPException(status_code=415, detail="Downloaded content is not a video")
            resolved_threshold = _resolve_threshold("video", payload.threshold)
            return _detect_video_from_path(
                downloaded_file,
                model=payload.model,
                threshold=resolved_threshold,
            )

        if payload.content_type == "audio":
            if not ((guessed_mime or "").startswith("audio/") or extension in SUPPORTED_AUDIO_EXTENSIONS):
                raise HTTPException(status_code=415, detail="Downloaded content is not an audio file")
            resolved_threshold = _resolve_threshold("audio", payload.threshold)
            return _detect_audio_from_path(
                downloaded_file,
                model=payload.model,
                threshold=resolved_threshold,
            )

        raise HTTPException(status_code=415, detail="This file type is not supported for analysis")
    finally:
        _cleanup_downloaded_file(downloaded_file)


@app.get("/info", response_class=JSONResponse)
async def info():
    """
    Получить информацию о приложении.
    """
    return {
        "title": AppConfig.TITLE,
        "version": AppConfig.VERSION,
        "description": AppConfig.DESCRIPTION,
        "debug": AppConfig.DEBUG,
        "config": AppConfig.get_config()
    }


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint с базовой информацией."""
    return {
        "message": "Deepfake Detector API",
        "version": AppConfig.VERSION,
        "endpoints": {
            "health": "/health",
            "docs": "/api/docs",
            "models": "/models",
            "detect": "/detect",
            "predict_image": "/predict/image",
            "detect_by_url": "/predict/url",
            "info": "/info"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        reload=AppConfig.RELOAD
    )
