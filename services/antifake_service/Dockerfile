FROM python:3.10-slim

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (ffmpeg is optional here; we use soundfile/librosa primarily)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/antifake_service/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /srv/requirements.txt

# Bring AntiFake code (kept in repo under thirdparty)
COPY thirdparty/AntiFake /srv/thirdparty/AntiFake

# Service code
COPY services/antifake_service/app /srv/app

EXPOSE 8010

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]

