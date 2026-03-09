1# Frontend MVP (Vanilla JS)

This folder contains a static web frontend for the existing FastAPI backend in `app/main.py`.

## What is implemented

- Main page with 2 actions:
  - `Check file authenticity` (active)
  - `Protect my file from deepfake creation` (placeholder)
- Check page:
  - image upload from computer
  - optional model selection (`/models`)
  - threshold input
  - request to backend `POST /detect` as `multipart/form-data` (`file` field)
  - result rendering (percent, verdict, label, confidence, model, threshold)
  - loading and error states
- Placeholder pages/features marked as `Coming soon`:
  - video upload
  - URL analysis
  - protection mode

## Run locally (Windows-friendly)

From project root (`deepfake_detector`), start backend:

```bash
python -m uvicorn app.main:app --reload
```

In another terminal, start frontend static server:

```bash
cd frontend
python -m http.server 3000
```

Open in browser:

- `http://127.0.0.1:3000`

## API URL configuration

Edit `frontend/assets/js/config.js`:

- `API_BASE_URL`: backend base URL (default `http://127.0.0.1:8000`)
- `MAX_FILE_SIZE_MB`: client-side file limit
- `SUPPORTED_EXTENSIONS`: allowed image extensions

## Planned (not implemented in backend)

- Video deepfake detection endpoint
- URL-based file analysis endpoint
- File protection/transformation endpoint

These remain disabled intentionally in the UI.

