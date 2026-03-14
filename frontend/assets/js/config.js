const API_PORT = 8000;

function resolveApiBaseUrl() {
  const queryApi = new URLSearchParams(window.location.search).get("api");
  if (queryApi) {
    return queryApi.replace(/\/$/, "");
  }

  const protocol = window.location.protocol;
  const hostname = window.location.hostname;

  if (protocol === "http:" || protocol === "https:") {
    return `${protocol}//${hostname}:${API_PORT}`;
  }

  return `http://127.0.0.1:${API_PORT}`;
}

export const APP_CONFIG = {
  API_BASE_URL: resolveApiBaseUrl(),
  MAX_FILE_SIZE_MB: 20,
  SUPPORTED_MEDIA: {
    photo: {
      extensions: [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"],
      mimePrefix: "image/",
      inputAccept: "image/*",
      submitText: "Проверить изображение",
    },
    video: {
      extensions: [".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv"],
      mimePrefix: "video/",
      inputAccept: "video/*",
      submitText: "Проверить видео",
    },
    audio: {
      extensions: [".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"],
      mimePrefix: "audio/",
      inputAccept: "audio/*",
      submitText: "Проверить аудио",
    },
  },
  CONTENT_TYPES: ["photo", "video", "audio"],
  DEFAULT_THRESHOLD: 50,
};
