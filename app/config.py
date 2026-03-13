import os

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Deepfake Detector API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "AI-powered deepfake detection system"

# Model Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 20 * 1024 * 1024))  # 20 MB default
TARGET_IMAGE_SIZE = (256, 256)
CONTENT_TYPES = ("photo", "video", "audio")

# Each model is tied to one specific content type.
MODEL_REGISTRY = {
    "model_photo": {
        "file_name": "model_photo.keras",
        "content_type": "photo",
    },
    "model_video": {
        "file_name": "model_video.keras",
        "content_type": "video",
    },
    "model_audio": {
        "file_name": "model_audio.keras",
        "content_type": "audio",
    },
}

DEFAULT_MODEL_BY_CONTENT = {
    "photo": "model_photo",
    "video": "model_video",
    "audio": "model_audio",
}

DEFAULT_THRESHOLD_BY_CONTENT = {
    "photo": 0.6515938,
    "video": 0.52268463,
    "audio": 0.6681531,
}

ALLOWED_MODEL_FILES = tuple(spec["file_name"] for spec in MODEL_REGISTRY.values())

# Supported file types
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}

# GPU Configuration
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

# Cache Configuration
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "false").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 1 hour default

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# Rate Limiting
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "false").lower() == "true"
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", 100))

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Temp cleanup configuration
TEMP_CLEANUP_ON_STARTUP = os.getenv("TEMP_CLEANUP_ON_STARTUP", "true").lower() == "true"

# Create necessary directories
for directory in [TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)


class AppConfig:
    """Application configuration class."""

    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    TESTING = os.getenv("TESTING", "false").lower() == "true"

    # FastAPI settings
    TITLE = API_TITLE
    VERSION = API_VERSION
    DESCRIPTION = API_DESCRIPTION

    # Server settings
    HOST = API_HOST
    PORT = API_PORT
    RELOAD = DEBUG

    # Model settings
    MAX_FILE_SIZE = MAX_FILE_SIZE
    TARGET_IMAGE_SIZE = TARGET_IMAGE_SIZE
    MODEL_REGISTRY = MODEL_REGISTRY
    CONTENT_TYPES = CONTENT_TYPES
    DEFAULT_MODEL_BY_CONTENT = DEFAULT_MODEL_BY_CONTENT
    DEFAULT_THRESHOLD_BY_CONTENT = DEFAULT_THRESHOLD_BY_CONTENT
    ALLOWED_MODEL_FILES = ALLOWED_MODEL_FILES
    USE_GPU = USE_GPU

    # Feature flags
    ENABLE_VIDEO_SUPPORT = os.getenv("ENABLE_VIDEO_SUPPORT", "false").lower() == "true"
    ENABLE_CACHING = ENABLE_CACHING
    ENABLE_RATE_LIMITING = ENABLE_RATE_LIMITING

    # Paths
    MODELS_DIR = MODELS_DIR
    TEMP_DIR = TEMP_DIR

    # Temp cleanup
    TEMP_CLEANUP_ON_STARTUP = TEMP_CLEANUP_ON_STARTUP

    @classmethod
    def get_config(cls) -> dict:
        """Get configuration as dictionary."""
        return {
            "debug": cls.DEBUG,
            "testing": cls.TESTING,
            "title": cls.TITLE,
            "version": cls.VERSION,
            "max_file_size": cls.MAX_FILE_SIZE,
            "content_types": list(CONTENT_TYPES),
            "default_model_by_content": DEFAULT_MODEL_BY_CONTENT,
            "default_threshold_by_content": DEFAULT_THRESHOLD_BY_CONTENT,
            "supported_image_types": list(SUPPORTED_IMAGE_EXTENSIONS),
            "supported_video_types": list(SUPPORTED_VIDEO_EXTENSIONS),
            "supported_audio_types": list(SUPPORTED_AUDIO_EXTENSIONS),
            "gpu_enabled": cls.USE_GPU,
            "caching_enabled": cls.ENABLE_CACHING,
            "video_support": cls.ENABLE_VIDEO_SUPPORT,
            "temp_cleanup_on_startup": cls.TEMP_CLEANUP_ON_STARTUP,
        }
