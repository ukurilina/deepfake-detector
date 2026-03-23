# Deepfake Detector API

AI-powered deepfake detection system built with FastAPI and TensorFlow.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Performance Tuning](#performance-tuning)
- [Contributing](#contributing)

---

## ✨ Features

- **Multi-model Support**: Load and use multiple deepfake detection models simultaneously
- **RESTful API**: Easy-to-use FastAPI endpoints with automatic documentation
- **Media Processing**: Automatic preprocessing for photos and videos
- **Error Handling**: Comprehensive error handling and validation
- **Health Checks**: Built-in health check endpoints for monitoring
- **Logging**: Detailed logging for debugging and monitoring
- **Deployable**: Can be containerized if needed (not included here)
- **Configurable**: Flexible configuration through environment variables
- **Content-Type Routing**: Models are mapped to `photo`/`video`/`audio` types
- **Audio Placeholder**: Audio flow is exposed in API/UI as `coming soon`

---

## 🏗️ Architecture

```
deepfake_detector/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── model.py             # ML model interface
│   ├── config.py            # Configuration management
│   ├── models_manager.py    # Model loading and management
│   └── video_utils.py       # Video processing utilities
├── models/
│   ├── model_photo.keras
│   └── model_video.keras
├── temp/                    # Temporary files
├── requirements.txt         # Python dependencies
└── README.md              # This file
```

### Two-service architecture (Python 3.12 + Python 3.10)

Audio protection via **AntiFake** is isolated into a separate web-service.
The main backend runs on **Python 3.12**, while AntiFake stack requires **Python 3.10**.

Services:

- **deepfake-api** (Python 3.12): main backend.
- **antifake-service** (Python 3.10): audio protection wrapper for `thirdparty/AntiFake`.

Main backend delegates audio protection to the service via `ANTIFAKE_SERVICE_URL`.

#### Run with Docker Compose

> Requires Docker Desktop / Docker Engine running.

```bash
docker compose up --build
```

In compose, the backend is configured with:

- `ANTIFAKE_SERVICE_URL=http://antifake-service:8010`

#### Run locally without Docker

1) Run `antifake-service` using Python 3.10:

```bash
cd services/antifake_service
python -m venv .venv310
# activate the venv and then:
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8010
```

2) Run `deepfake-api` using Python 3.12 and point it to the service:

```bash
set ANTIFAKE_SERVICE_URL=http://localhost:8010
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Component Overview

- **main.py**: FastAPI application with endpoints
- **model.py**: Model inference interface
- **models_manager.py**: Singleton manager for loading/managing models
- **config.py**: Centralized configuration
- **video_utils.py**: Video frame extraction utilities

---

## 🚀 Quick Start

### Local Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd deepfake_detector

# 2. Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access API

Once running, visit:
- **API Docs**: http://localhost:8000/api/docs
- **Alternative Docs**: http://localhost:8000/api/redoc
- **Health Check**: http://localhost:8000/health

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "available_models": [
    "model_photo",
    "model_video"
  ],
  "version": "1.0.0",
  "gpu_enabled": false
}
```

#### 2. List Models
```http
GET /models
GET /models?content_type=photo
```

**Response:**
```json
{
  "models": ["model_photo"],
  "count": 1,
  "default_model": "model_photo",
  "content_type": "photo",
  "models_by_content": {
    "photo": ["model_photo"],
    "video": ["model_video"],
    "audio": []
  }
}
```

#### 3. Detect Deepfake
```http
POST /detect
Content-Type: multipart/form-data

Parameters:
- file: Media file (photo/video)
- content_type: photo|video|audio (optional, inferred when possible)
- model: Model name (optional, filtered by content type)
- threshold: Classification threshold 0.0-1.0 (optional, default: 0.5)
```

**Response:**
```json
{
  "probability": 0.8523,
  "label": "deepfake",
  "confidence": 0.8523,
  "percent": 85.23,
  "model_used": "model_photo",
  "threshold": 0.5
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid file type or request
- `415` - Uploaded/downloaded media does not match selected type
- `413` - File too large
- `503` - No models available
- `500` - Server error

#### 4. Detect by URL
```http
POST /predict/url
Content-Type: application/json
```

**Request body:**
```json
{
  "url": "https://example.com/file.mp4",
  "content_type": "video",
  "model": "model_video",
  "threshold": 0.5
}
```

`content_type` is required and must be one of `photo`, `video`, `audio`.
For `audio`, API currently returns `501` (`coming soon`).

#### 5. Application Info
```http
GET /info
```

**Response:**
```json
{
  "title": "Deepfake Detector API",
  "version": "1.0.0",
  "description": "AI-powered deepfake detection system",
  "debug": false,
  "config": { ... }
}
```

#### 6. Root Endpoint
```http
GET /
```

**Response:**
```json
{
  "message": "Deepfake Detector API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "docs": "/api/docs",
    "models": "/models",
    "detect": "/detect",
    "info": "/info"
  }
}
```

---

## 💻 Installation

### Requirements

- Python 3.9+
- TensorFlow 2.13+
- 8GB RAM minimum (GPU recommended for production)

### Setup Steps

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Video support dependency is included in requirements
# (opencv-python-headless)

# 4. Create necessary directories
mkdir temp
```

### Environment Variables

```bash
# Server configuration
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Model configuration
MAX_FILE_SIZE=20971520  # 20 MB
USE_GPU=false
# Default mapping is configured in app/config.py (DEFAULT_MODEL_BY_CONTENT)

# Features
ENABLE_VIDEO_SUPPORT=false
ENABLE_CACHING=false
ENABLE_RATE_LIMITING=false

# CORS
CORS_ORIGINS=*
```

---

## 📖 Usage Examples

### Python Client

```python
import requests

# Check health
response = requests.get('http://localhost:8000/health')
print(response.json())

# List models
response = requests.get('http://localhost:8000/models')
print(response.json())

# Detect deepfake
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    params = {
        'model': 'model_photo',
        'threshold': 0.5
    }
    response = requests.post(
        'http://localhost:8000/detect',
        files=files,
        params=params
    )
    print(response.json())
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Detect deepfake
curl -X POST -F "file=@image.jpg" \
  "http://localhost:8000/detect?threshold=0.5"
```

---

## ⚙️ Configuration

### config.py Overview

Configuration is managed through `app/config.py`:

```python
from app.config import AppConfig

# Access configuration
print(AppConfig.TITLE)           # Application title
print(AppConfig.VERSION)         # API version
print(AppConfig.MAX_FILE_SIZE)   # Max upload size
print(AppConfig.USE_GPU)         # GPU enabled
print(AppConfig.MODELS_DIR)      # Models directory
```

### Environment-based Configuration

Set environment variables before running:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export USE_GPU=true
python -m uvicorn app.main:app --reload
```

---

## 🚄 Performance Tuning

### GPU Support

To enable GPU support:

```bash
# Install GPU dependencies
pip install tensorflow[and-cuda]

# Set environment variable
export USE_GPU=true

# Run application
python -m uvicorn app.main:app
```

### Model Optimization

1. **Model Quantization**: Reduce model size for faster inference
2. **Model Pruning**: Remove unnecessary weights
3. **ONNX Runtime**: Use ONNX Runtime for faster CPU inference

### Caching

Enable result caching with Redis:

```bash
# Install Redis
pip install redis

# Start Redis
redis-server

# Enable in environment
export ENABLE_CACHING=true
```

---

## 📈 Scaling

### Short-term (Weeks 1-2)
- Fix critical bugs ✓
- Add logging ✓
- Health checks ✓

### Medium-term (Weeks 3-4)
- Video support
- Batch processing
- Result caching
- Metrics/monitoring

### Long-term (Weeks 5+)
- Model optimization
- Async processing (Celery)
- Kubernetes deployment
- Database integration

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🆘 Troubleshooting

### Models not loading
- Check `models/` directory exists
- Verify `.keras` file extensions

### API not starting
- Ensure port 8000 is available
- Check Python version (3.9+)
- Verify dependencies installed: `pip install -r requirements.txt`

### Out of memory
- Reduce `MAX_FILE_SIZE`
- Enable GPU: `USE_GPU=true`
- Use model quantization

### Slow inference
- Enable GPU support
- Check CPU usage with `top` or Task Manager
- Consider batch processing

---

## 📞 Support

For issues and questions:
1. Check API health: `curl http://localhost:8000/health`

---

## 🐳 Docker Compose

```bash
# Build and start the API
Docker compose up --build
```

```bash
# Stop the API
Docker compose down
```

Notes:
- Models are mounted from `./models` (read-only).
- Temporary files are written to `./temp`.
- Configure environment via `docker-compose.yml` or a `.env` file.
