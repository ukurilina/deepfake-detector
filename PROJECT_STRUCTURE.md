"""
# Проект: Deepfake Detector

Актуальная структура (только существующие файлы и папки):

```
deepfake_detector/
├── README.md
├── PROJECT_STRUCTURE.md
├── menu.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── model.py
│   ├── models_manager.py
│   └── video_utils.py
├── models/
│   ├── model_photo.keras
│   └── model_video.keras
└── temp/
```

Минимально необходимые для работы API:
- `app/main.py` — FastAPI приложение и эндпоинты
- `app/model.py` — инференс моделей
- `app/models_manager.py` — загрузка/хранение моделей
- `app/config.py` — конфигурация
- `requirements.txt` — зависимости
- каталог `models/` с файлами моделей `.keras`

Вспомогательные:
- `menu.py` — консольное меню (опционально)
- каталог `temp/` — рабочий (создается по необходимости)

# Для просмотра этой структуры:
# python -c "import this; exec(open(__file__).read())"
"""
