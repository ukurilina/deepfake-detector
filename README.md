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
- **Image Processing**: Automatic image preprocessing and validation
- **Error Handling**: Comprehensive error handling and validation
- **Health Checks**: Built-in health check endpoints for monitoring
- **Logging**: Detailed logging for debugging and monitoring
- **Deployable**: Can be containerized if needed (not included here)
- **Configurable**: Flexible configuration through environment variables
- **Extensible**: Ready for video support and additional features

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
│   ├── best_model3 (3)_acc.keras
│   └── best_model3 (4)_loss.keras
├── temp/                    # Temporary files
├── requirements.txt         # Python dependencies
└── README.md              # This file
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
    "best_model3 (3)_acc",
    "best_model3 (4)_loss"
  ],
  "version": "1.0.0",
  "gpu_enabled": false
}
```

#### 2. List Models
```http
GET /models
```

**Response:**
```json
{
  "models": [
    "best_model3 (3)_acc",
    "best_model3 (4)_loss"
  ],
  "count": 2,
  "default_model": "best_model3 (3)_acc"
}
```

#### 3. Detect Deepfake
```http
POST /detect
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- model: Model name (optional, defaults to first loaded)
- threshold: Classification threshold 0.0-1.0 (optional, default: 0.5)
```

**Response:**
```json
{
  "probability": 0.8523,
  "label": "deepfake",
  "confidence": 0.8523,
  "percent": 85.23,
  "model_used": "best_model3 (3)_acc",
  "threshold": 0.5
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid file type or request
- `413` - File too large
- `503` - No models available
- `500` - Server error

#### 4. Application Info
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

#### 5. Root Endpoint
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

# 3. (Optional) Install video support
pip install opencv-python

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
DEFAULT_MODEL_NAME=best_model3 (3)_acc

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
        'model': 'best_model3 (3)_acc',
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
