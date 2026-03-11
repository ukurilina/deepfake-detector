import { APP_CONFIG } from "./config.js";
import { analyzeFileByUrl, detectImage, fetchModels } from "./api.js";
import { buildVerdict, safePercent, setStatus, setText, toggleHidden } from "./ui.js";

function validateFile(file) {
  if (!file) {
    return "Выберите файл изображения.";
  }

  const lowerName = (file.name || "").toLowerCase();
  const hasAllowedExtension = APP_CONFIG.SUPPORTED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
  const hasAllowedMime = (file.type || "").startsWith(APP_CONFIG.SUPPORTED_MIME_PREFIX);

  if (!hasAllowedExtension || !hasAllowedMime) {
    return `Неподдерживаемый тип файла. Разрешено: ${APP_CONFIG.SUPPORTED_EXTENSIONS.join(", ")}`;
  }

  const maxBytes = APP_CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024;
  if (file.size > maxBytes) {
    return `Файл слишком большой. Максимальный размер: ${APP_CONFIG.MAX_FILE_SIZE_MB} МБ.`;
  }

  return "";
}

function validateUrl(urlText) {
  if (!urlText || !urlText.trim()) {
    return "Введите URL.";
  }

  try {
    const parsed = new URL(urlText);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return "Поддерживаются только ссылки http/https.";
    }
  } catch (_error) {
    return "Введите корректный URL.";
  }

  return "";
}

function normalizeResult(payload) {
  const percentFromPercent = safePercent(payload?.percent);
  const percentFromProbability = safePercent(Number(payload?.probability) * 100);
  const percent = percentFromPercent ?? percentFromProbability;

  return {
    percent,
    label: payload?.label || "",
    confidencePercent: safePercent(Number(payload?.confidence) * 100),
    modelUsed: payload?.model_used || "Н/Д",
    thresholdPercent: safePercent(Number(payload?.threshold) * 100),
  };
}

function renderResult(result) {
  toggleHidden("result", false);
  setText("resultPercent", result.percent === null ? "Н/Д" : `${result.percent.toFixed(2)}%`);
  setText("resultVerdict", buildVerdict(result.percent, result.label));
  setText("resultLabel", result.label || "Н/Д");
  setText(
    "resultConfidence",
    Number.isFinite(result.confidencePercent) ? `${result.confidencePercent.toFixed(2)}%` : "Н/Д"
  );
  setText("resultModel", result.modelUsed);
  setText(
    "resultThreshold",
    Number.isFinite(result.thresholdPercent) ? `${result.thresholdPercent.toFixed(2)}%` : "Н/Д"
  );
}

async function runDetection(requestFactory) {
  setStatus("");
  toggleHidden("result", true);

  try {
    const payload = await requestFactory();
    const result = normalizeResult(payload);

    if (result.percent === null) {
      throw new Error("Ответ сервера не содержит полей вероятности.");
    }

    renderResult(result);
    setStatus("Проверка успешно завершена.", "success");
  } catch (error) {
    setStatus(error.message || "Непредвиденная ошибка.", "error");
  }
}

async function loadModels() {
  const modelSelect = document.getElementById("modelSelect");
  const urlModelSelect = document.getElementById("urlModelSelect");

  if (!modelSelect && !urlModelSelect) {
    return;
  }

  const selectors = [modelSelect, urlModelSelect].filter(Boolean);

  try {
    const models = await fetchModels();
    if (!models.length) {
      for (const select of selectors) {
        select.innerHTML = '<option value="">Нет доступных моделей</option>';
        select.disabled = true;
      }
      return;
    }

    for (const select of selectors) {
      select.innerHTML = '<option value="">Модель по умолчанию</option>';
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        select.appendChild(option);
      }
    }
  } catch (error) {
    setStatus(`Не удалось загрузить модели: ${error.message}`, "error");
  }
}

function setMode(mode) {
  const isFileMode = mode === "file";
  toggleHidden("fileSection", !isFileMode);
  toggleHidden("urlSection", isFileMode);
  toggleHidden("result", true);
  setStatus("");

  const fileBtn = document.getElementById("modeFileButton");
  const urlBtn = document.getElementById("modeUrlButton");
  if (fileBtn) {
    fileBtn.classList.toggle("active", isFileMode);
    fileBtn.setAttribute("aria-selected", String(isFileMode));
  }
  if (urlBtn) {
    urlBtn.classList.toggle("active", !isFileMode);
    urlBtn.setAttribute("aria-selected", String(!isFileMode));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("detectForm");
  const fileInput = document.getElementById("fileInput");
  const submitButton = document.getElementById("submitButton");
  const resetButton = document.getElementById("resetButton");
  const thresholdInput = document.getElementById("thresholdInput");
  const modelSelect = document.getElementById("modelSelect");
  const urlModelSelect = document.getElementById("urlModelSelect");

  const modeFileButton = document.getElementById("modeFileButton");
  const modeUrlButton = document.getElementById("modeUrlButton");

  const urlForm = document.getElementById("urlForm");
  const urlInput = document.getElementById("urlInput");
  const urlSubmitButton = document.getElementById("urlSubmitButton");
  const thresholdInputUrl = document.getElementById("thresholdInputUrl");
  const urlResetButton = document.getElementById("urlResetButton");

  loadModels();
  setMode("file");

  modeFileButton?.addEventListener("click", () => setMode("file"));
  modeUrlButton?.addEventListener("click", () => setMode("url"));

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = fileInput?.files?.[0];
    const validationError = validateFile(file);
    if (validationError) {
      setStatus(validationError, "error");
      return;
    }

    const thresholdPercent = Number(thresholdInput?.value ?? APP_CONFIG.DEFAULT_THRESHOLD);
    if (!Number.isFinite(thresholdPercent) || thresholdPercent < 0 || thresholdPercent > 100) {
      setStatus("Порог срабатывания должен быть в диапазоне от 0% до 100%.", "error");
      return;
    }

    const threshold = thresholdPercent / 100;

    submitButton.disabled = true;
    submitButton.textContent = "Анализ...";
    setStatus("Отправка изображения в модель...");

    await runDetection(() =>
      detectImage({
        file,
        model: modelSelect?.value || "",
        threshold,
      })
    );

    submitButton.disabled = false;
    submitButton.textContent = "Проверить изображение";
  });

  urlForm?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const urlValue = (urlInput?.value || "").trim();
    const urlError = validateUrl(urlValue);
    if (urlError) {
      setStatus(urlError, "error");
      return;
    }

    const thresholdPercent = Number(thresholdInputUrl?.value ?? APP_CONFIG.DEFAULT_THRESHOLD);
    if (!Number.isFinite(thresholdPercent) || thresholdPercent < 0 || thresholdPercent > 100) {
      setStatus("Порог срабатывания должен быть в диапазоне от 0% до 100%.", "error");
      return;
    }

    const threshold = thresholdPercent / 100;

    urlSubmitButton.disabled = true;
    urlSubmitButton.textContent = "Анализ...";
    setStatus("Скачивание файла по URL и анализ...");

    await runDetection(() =>
      analyzeFileByUrl({
        url: urlValue,
        model: urlModelSelect?.value || "",
        threshold,
      })
    );

    urlSubmitButton.disabled = false;
    urlSubmitButton.textContent = "Проверить по URL";
  });

  resetButton?.addEventListener("click", () => {
    form?.reset();
    setStatus("");
    toggleHidden("result", true);
  });

  urlResetButton?.addEventListener("click", () => {
    urlForm?.reset();
    setStatus("");
    toggleHidden("result", true);
  });
});
