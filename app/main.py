import os
import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import AppConfig, SUPPORTED_IMAGE_EXTENSIONS
from app.model import predict_deepfake_probability, initialize_models, get_available_models

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


@app.on_event("startup")
async def startup_event():
    """Выполняется при старте приложения."""
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

        health_status = {
            "status": "healthy" if available_models else "degraded",
            "models_loaded": len(available_models),
            "available_models": available_models,
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
async def list_models():
    """
    Получить список доступных моделей.
    """
    try:
        available_models = get_available_models()
        return {
            "models": available_models,
            "count": len(available_models),
            "default_model": available_models[0] if available_models else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", response_class=JSONResponse)
async def detect(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Model name to use for detection"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Classification threshold")
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

    available_models = get_available_models()
    if not available_models:
        raise HTTPException(status_code=503, detail="No models loaded. Service unavailable.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    filename = file.filename or "image"
    suffix = os.path.splitext(filename)[1].lower()
    if suffix and suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > AppConfig.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {AppConfig.MAX_FILE_SIZE / 1024 / 1024:.0f} MB"
        )

    if model and model not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found. Available models: {', '.join(available_models)}"
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".jpg", dir=AppConfig.TEMP_DIR) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name

        try:
            return predict_deepfake_probability(
                tmp_path,
                model_name=model,
                threshold=threshold
            )
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
