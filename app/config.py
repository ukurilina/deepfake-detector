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
    "проверка фото": {
        "file_name": "model_photo.keras",
        "content_type": "photo",
        "calibration_pivot": 0.6515938,
    },
    "альтернативная проверка фото": {
        "file_name": "model_photo_strict.keras",
        "content_type": "photo",
        "calibration_pivot": 0.027091859,
    },
    "проверка видео": {
        "file_name": "model_video.keras",
        "content_type": "video",
        "calibration_pivot": 0.52268463,
    },
    "проверка аудио": {
        "file_name": "model_audio.keras",
        "content_type": "audio",
        "calibration_pivot": 0.6681531,
    },
}

DEFAULT_MODEL_BY_CONTENT = {
    "photo": "model_photo",
    "video": "model_video",
    "audio": "model_audio",
}

CALIBRATION_PIVOT_BY_MODEL = {
    model_name: float(spec.get("calibration_pivot", 0.5))
    for model_name, spec in MODEL_REGISTRY.items()
}
# Backward-compatible view: one pivot per content type, derived from the default model.
CALIBRATION_PIVOT_BY_CONTENT = {
    content_type: float(CALIBRATION_PIVOT_BY_MODEL.get(default_model, 0.5))
    for content_type, default_model in DEFAULT_MODEL_BY_CONTENT.items()
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
ENABLE_FACE_ALIGN_FOR_VIDEO = os.getenv("ENABLE_FACE_ALIGN_FOR_VIDEO", "true").lower() == "true"
REUSE_LAST_VIDEO_FACE_TRANSFORM = os.getenv("REUSE_LAST_VIDEO_FACE_TRANSFORM", "true").lower() == "true"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Optional explicit path to ffmpeg executable (useful on Windows if ffmpeg isn't on PATH).
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "") or None

# Protection (Foolbox) output directory (kept under temp and auto cleaned).
PROTECT_DIR = os.path.join(TEMP_DIR, "protected")

# Protection defaults (tuned for minimal visible impact; can be overridden per request).
PROTECT_DEFAULT_ATTACK = os.getenv("PROTECT_DEFAULT_ATTACK", "fgsm")  # fgsm|pgd
PROTECT_DEFAULT_EPS = float(os.getenv("PROTECT_DEFAULT_EPS", "0.01"))
PROTECT_DEFAULT_STEPS = int(os.getenv("PROTECT_DEFAULT_STEPS", "10"))
PROTECT_DEFAULT_FRAME_STRIDE = int(os.getenv("PROTECT_DEFAULT_FRAME_STRIDE", "1"))
PROTECT_MAX_FRAMES = int(os.getenv("PROTECT_MAX_FRAMES", "24"))
PROTECT_TTL_SECONDS = int(os.getenv("PROTECT_TTL_SECONDS", str(60 * 20)))  # 20 min

# AntiFake (audio protection)
ANTIFAKE_DIR = os.path.join(BASE_DIR, "thirdparty", "AntiFake")
ANTIFAKE_RUN_PY = os.path.join(ANTIFAKE_DIR, "run.py")
ANTIFAKE_SPEAKER_DATABASE = os.path.join(ANTIFAKE_DIR, "speakers_database")

# AntiFake requires additional weights downloaded separately per its README.
# - Put tortoise weights under thirdparty/AntiFake/tortoise/: autoregressive.pth, diffusion_decoder.pth
# - RTVC/Coqui/Tortoise stacks may require additional checkpoints. For the bundled RTVC code path,
#   the default model folder typically contains encoder.pt and vocoder.pt.
ANTIFAKE_REQUIRED_FILES = {
    "tortoise_autoregressive": os.path.join(ANTIFAKE_DIR, "tortoise", "autoregressive.pth"),
    "tortoise_diffusion_decoder": os.path.join(ANTIFAKE_DIR, "tortoise", "diffusion_decoder.pth"),
    "rtvc_encoder": os.path.join(ANTIFAKE_DIR, "saved_models", "default", "encoder.pt"),
    "rtvc_vocoder": os.path.join(ANTIFAKE_DIR, "saved_models", "default", "vocoder.pt"),
}

# Audio protection defaults
ANTIFAKE_MAX_SECONDS = float(os.getenv("ANTIFAKE_MAX_SECONDS", "20"))
ANTIFAKE_FORCE_CPU = os.getenv("ANTIFAKE_FORCE_CPU", "1") == "1"
ANTIFAKE_TARGET_SELECTION = os.getenv("ANTIFAKE_TARGET_SELECTION", "auto")  # auto|random
ANTIFAKE_RANDOM_TARGETS = int(os.getenv("ANTIFAKE_RANDOM_TARGETS", "8"))

# AntiFake service (separate Python 3.10 microservice)
ANTIFAKE_SERVICE_URL = os.getenv("ANTIFAKE_SERVICE_URL", "") or None
ANTIFAKE_SERVICE_TIMEOUT_SECONDS = float(os.getenv("ANTIFAKE_SERVICE_TIMEOUT_SECONDS", "1800"))

# Temp cleanup configuration
TEMP_CLEANUP_ON_STARTUP = os.getenv("TEMP_CLEANUP_ON_STARTUP", "true").lower() == "true"

# Create necessary directories
for directory in [TEMP_DIR, PROTECT_DIR]:
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
    CALIBRATION_PIVOT_BY_MODEL = CALIBRATION_PIVOT_BY_MODEL
    CALIBRATION_PIVOT_BY_CONTENT = CALIBRATION_PIVOT_BY_CONTENT
    ALLOWED_MODEL_FILES = ALLOWED_MODEL_FILES
    USE_GPU = USE_GPU

    # Feature flags
    ENABLE_VIDEO_SUPPORT = os.getenv("ENABLE_VIDEO_SUPPORT", "false").lower() == "true"
    ENABLE_FACE_ALIGN_FOR_VIDEO = ENABLE_FACE_ALIGN_FOR_VIDEO
    REUSE_LAST_VIDEO_FACE_TRANSFORM = REUSE_LAST_VIDEO_FACE_TRANSFORM
    ENABLE_CACHING = ENABLE_CACHING
    ENABLE_RATE_LIMITING = ENABLE_RATE_LIMITING

    # Paths
    MODELS_DIR = MODELS_DIR
    TEMP_DIR = TEMP_DIR
    PROTECT_DIR = PROTECT_DIR

     # External tools
    FFMPEG_PATH = FFMPEG_PATH

    # Protection
    PROTECT_DEFAULT_ATTACK = PROTECT_DEFAULT_ATTACK
    PROTECT_DEFAULT_EPS = PROTECT_DEFAULT_EPS
    PROTECT_DEFAULT_STEPS = PROTECT_DEFAULT_STEPS
    PROTECT_DEFAULT_FRAME_STRIDE = PROTECT_DEFAULT_FRAME_STRIDE
    PROTECT_MAX_FRAMES = PROTECT_MAX_FRAMES
    PROTECT_TTL_SECONDS = PROTECT_TTL_SECONDS

    # AntiFake
    ANTIFAKE_DIR = ANTIFAKE_DIR
    ANTIFAKE_RUN_PY = ANTIFAKE_RUN_PY
    ANTIFAKE_SPEAKER_DATABASE = ANTIFAKE_SPEAKER_DATABASE
    ANTIFAKE_REQUIRED_FILES = ANTIFAKE_REQUIRED_FILES
    ANTIFAKE_MAX_SECONDS = ANTIFAKE_MAX_SECONDS
    ANTIFAKE_FORCE_CPU = ANTIFAKE_FORCE_CPU
    ANTIFAKE_TARGET_SELECTION = ANTIFAKE_TARGET_SELECTION
    ANTIFAKE_RANDOM_TARGETS = ANTIFAKE_RANDOM_TARGETS

    # AntiFake microservice
    ANTIFAKE_SERVICE_URL = ANTIFAKE_SERVICE_URL
    ANTIFAKE_SERVICE_TIMEOUT_SECONDS = ANTIFAKE_SERVICE_TIMEOUT_SECONDS

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
            "calibration_pivot_by_model": CALIBRATION_PIVOT_BY_MODEL,
            "calibration_pivot_by_content": CALIBRATION_PIVOT_BY_CONTENT,
            "supported_image_types": list(SUPPORTED_IMAGE_EXTENSIONS),
            "supported_video_types": list(SUPPORTED_VIDEO_EXTENSIONS),
            "supported_audio_types": list(SUPPORTED_AUDIO_EXTENSIONS),
            "gpu_enabled": cls.USE_GPU,
            "caching_enabled": cls.ENABLE_CACHING,
            "video_support": cls.ENABLE_VIDEO_SUPPORT,
            "face_align_for_video": cls.ENABLE_FACE_ALIGN_FOR_VIDEO,
            "reuse_last_video_face_transform": cls.REUSE_LAST_VIDEO_FACE_TRANSFORM,
            "temp_cleanup_on_startup": cls.TEMP_CLEANUP_ON_STARTUP,
        }
