import { APP_CONFIG } from "./config.js";
import { analyzeFileByUrl, detectFile, fetchModelsByContentType } from "./api.js";
import {
  buildVerdict,
  drawImageRegions,
  findSuspiciousRegions,
  resetHeatmapView,
  safePercent,
  setStatus,
  setText,
  toggleHidden,
  validateHeatmap2D,
} from "./ui.js";

const heatmapState = {
  frameHeatmaps: [],
  currentFrameIndex: 0,
  sourceImageUrl: "",
  sourceImageIsObjectUrl: false,
};

function clearSourceImageUrl() {
  if (heatmapState.sourceImageUrl && heatmapState.sourceImageIsObjectUrl) {
    URL.revokeObjectURL(heatmapState.sourceImageUrl);
  }
  heatmapState.sourceImageUrl = "";
  heatmapState.sourceImageIsObjectUrl = false;
}

function setSourceImageFromFile(file, contentType) {
  clearSourceImageUrl();
  if (contentType !== "photo" || !file) {
    return;
  }
  heatmapState.sourceImageUrl = URL.createObjectURL(file);
  heatmapState.sourceImageIsObjectUrl = true;
}

function setSourceImageFromUrl(url, contentType) {
  clearSourceImageUrl();
  if (contentType !== "photo" || !url) {
    return;
  }
  heatmapState.sourceImageUrl = url;
  heatmapState.sourceImageIsObjectUrl = false;
}

function getMediaConfig(contentType) {
  return APP_CONFIG.SUPPORTED_MEDIA[contentType] || APP_CONFIG.SUPPORTED_MEDIA.photo;
}

function getDefaultThresholdPercent(contentType) {
  void contentType;
  return APP_CONFIG.DEFAULT_THRESHOLD;
}

function setThresholdDefault(input, contentType) {
  if (!input) {
    return;
  }
  const nextValue = getDefaultThresholdPercent(contentType);
  input.value = String(Number(nextValue.toFixed(6)));
}

function parseThresholdPercent(inputValue, contentType) {
  const raw = String(inputValue ?? "").trim();
  const effective = raw === "" ? getDefaultThresholdPercent(contentType) : Number(raw);
  return effective;
}

function validateFile(file, contentType) {
  if (!file) {
    return "Выберите файл.";
  }

  const mediaConfig = getMediaConfig(contentType);

  const lowerName = (file.name || "").toLowerCase();
  const hasAllowedExtension = mediaConfig.extensions.some((ext) => lowerName.endsWith(ext));
  const hasAllowedMime = (file.type || "").startsWith(mediaConfig.mimePrefix);

  if (!hasAllowedExtension || !hasAllowedMime) {
    return `Неподдерживаемый тип файла. Разрешено: ${mediaConfig.extensions.join(", ")}`;
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
    modelUsed: payload?.model_used || "Н/Д",
    thresholdPercent: safePercent(Number(payload?.threshold) * 100),
    heatmap: validateHeatmap2D(payload?.heatmap) ? payload.heatmap : null,
    frameHeatmaps: Array.isArray(payload?.frame_heatmaps)
      ? payload.frame_heatmaps.filter((frame) => validateHeatmap2D(frame))
      : [],
    sourceImageDataUrl: typeof payload?.source_image_data_url === "string" ? payload.source_image_data_url : "",
  };
}

function getHeatmapSizeLabel(heatmap) {
  if (!validateHeatmap2D(heatmap)) {
    return "Н/Д";
  }
  return `${heatmap[0].length}x${heatmap.length}`;
}

async function renderHeatmap(result) {
  // В интерфейсе больше не показываем саму карту внимания (heatmap).
  // Оставляем только исходное изображение с прямоугольниками по «горячим» областям.

  const heatmap = result?.heatmap;
  const hasSourceImage = Boolean(heatmapState.sourceImageUrl);
  if (validateHeatmap2D(heatmap) && hasSourceImage) {
    const regions = findSuspiciousRegions(heatmap);
    if (regions.length) {
      const heatmapWidth = heatmap[0]?.length || 0;
      const heatmapHeight = heatmap.length;
      const rendered = await drawImageRegions(
        "regionsCanvas",
        heatmapState.sourceImageUrl,
        regions,
        heatmapWidth,
        heatmapHeight
      );
      if (rendered) {
        toggleHidden("heatmapSection", false);
        toggleHidden("regionsSection", false);
        setText("regionsMeta", `Найдено спорных областей: ${regions.length}`);
        return;
      }
    }
  }

  // Для видео (frameHeatmaps) или если heatmap/картинка недоступны — просто скрываем блок.
  resetHeatmapView();
}

async function renderResult(result) {
  toggleHidden("result", false);
  setText("resultPercent", result.percent === null ? "Н/Д" : `${result.percent.toFixed(2)}%`);
  setText("resultVerdict", buildVerdict(result.percent, result.label));
  setText("resultLabel", result.label || "Н/Д");
  setText("resultModel", result.modelUsed);
  setText(
    "resultThreshold",
    Number.isFinite(result.thresholdPercent) ? `${result.thresholdPercent.toFixed(2)}%` : "Н/Д"
  );
  await renderHeatmap(result);
}

function isModelAllowedForContent(selectedModel, allowedModels) {
  if (!selectedModel) {
    return true;
  }
  return Array.isArray(allowedModels) && allowedModels.includes(selectedModel);
}

async function runDetection(requestFactory) {
  setStatus("");
  toggleHidden("result", true);

  try {
    const payload = await requestFactory();
    const result = normalizeResult(payload);

    if (result.sourceImageDataUrl) {
      setSourceImageFromUrl(result.sourceImageDataUrl, "photo");
    }

    if (result.percent === null) {
      setStatus("Ответ сервера не содержит полей вероятности.", "error");
      return;
    }

    await renderResult(result);
    setStatus("Проверка успешно завершена.", "success");
  } catch (error) {
    setStatus(error.message || "Непредвиденная ошибка.", "error");
  }
}

async function populateModelSelect(select, contentType) {
  if (!select) {
    return;
  }

  try {
    const { models, defaultModel } = await fetchModelsByContentType(contentType);
    select.dataset.allowedModels = JSON.stringify(models);
    if (!models.length) {
      select.innerHTML = '<option value="">Нет доступных моделей</option>';
      select.disabled = true;
      return;
    }

    select.disabled = false;
    select.innerHTML = '';
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      if (defaultModel && model === defaultModel) {
        option.textContent = `${model} (по умолчанию)`;
      }
      select.appendChild(option);
    }
  } catch (error) {
    select.dataset.allowedModels = JSON.stringify([]);
    select.innerHTML = '<option value="">Ошибка загрузки моделей</option>';
    select.disabled = true;
    setStatus(`Не удалось загрузить модели: ${error.message}`, "error");
  }
}

function getAllowedModelsFromSelect(select) {
  if (!select) {
    return [];
  }

  try {
    const raw = select.dataset.allowedModels || "[]";
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_error) {
    return [];
  }
}

function syncFileUiByContentType(contentType, refs) {
  const mediaConfig = getMediaConfig(contentType);
  const isComingSoon = Boolean(mediaConfig.comingSoon);

  if (contentType === "video") {
    refs.fileInputLabel.textContent = "Видеофайл";
  } else if (contentType === "audio") {
    refs.fileInputLabel.textContent = "Аудиофайл";
  } else {
    refs.fileInputLabel.textContent = "Файл изображения";
  }
  refs.fileInput.setAttribute("accept", mediaConfig.inputAccept);
  refs.submitButton.textContent = mediaConfig.submitText;
  refs.submitButton.disabled = isComingSoon;

  if (isComingSoon) {
    setStatus("Проверка аудио будет добавлена позже.", "error");
  } else {
    setStatus("");
  }
}

function setMode(mode) {
  const isFileMode = mode === "file";
  toggleHidden("fileSection", !isFileMode);
  toggleHidden("urlSection", isFileMode);
  toggleHidden("result", true);
  resetHeatmapView();
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
  const fileContentTypeSelect = document.getElementById("fileContentTypeSelect");
  const fileInputLabel = document.getElementById("fileInputLabel");
  const urlModelSelect = document.getElementById("urlModelSelect");
  const urlContentTypeSelect = document.getElementById("urlContentTypeSelect");

  const modeFileButton = document.getElementById("modeFileButton");
  const modeAudioButton = document.getElementById("modeAudioButton");
  const modeUrlButton = document.getElementById("modeUrlButton");

  const urlForm = document.getElementById("urlForm");
  const urlInput = document.getElementById("urlInput");
  const urlSubmitButton = document.getElementById("urlSubmitButton");
  const thresholdInputUrl = document.getElementById("thresholdInputUrl");
  const urlResetButton = document.getElementById("urlResetButton");

  const refs = {
    fileInput,
    submitButton,
    fileInputLabel,
  };

  const initialFileContentType = fileContentTypeSelect?.value || "photo";
  const initialUrlContentType = urlContentTypeSelect?.value || "photo";
  setThresholdDefault(thresholdInput, initialFileContentType);
  setThresholdDefault(thresholdInputUrl, initialUrlContentType);
  syncFileUiByContentType(initialFileContentType, refs);
  populateModelSelect(modelSelect, initialFileContentType);
  populateModelSelect(urlModelSelect, initialUrlContentType);
  setMode("file");

  modeFileButton?.addEventListener("click", () => setMode("file"));
  modeAudioButton?.addEventListener("click", async () => {
    setMode("file");
    if (fileContentTypeSelect) {
      fileContentTypeSelect.value = "audio";
      syncFileUiByContentType("audio", refs);
      await populateModelSelect(modelSelect, "audio");
    }
  });
  modeUrlButton?.addEventListener("click", () => setMode("url"));


  fileContentTypeSelect?.addEventListener("change", async () => {
    const contentType = fileContentTypeSelect.value;
    setThresholdDefault(thresholdInput, contentType);
    syncFileUiByContentType(contentType, refs);
    await populateModelSelect(modelSelect, contentType);
  });

  urlContentTypeSelect?.addEventListener("change", async () => {
    const contentType = urlContentTypeSelect.value;
    setThresholdDefault(thresholdInputUrl, contentType);
    await populateModelSelect(urlModelSelect, contentType);
  });

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const contentType = fileContentTypeSelect?.value || "photo";
    const mediaConfig = getMediaConfig(contentType);
    if (mediaConfig.comingSoon) {
      setStatus("Проверка аудио будет добавлена позже.", "error");
      return;
    }

    const file = fileInput?.files?.[0];
    const validationError = validateFile(file, contentType);
    if (validationError) {
      setStatus(validationError, "error");
      return;
    }

    const thresholdPercent = parseThresholdPercent(thresholdInput?.value, contentType);
    if (!Number.isFinite(thresholdPercent) || thresholdPercent < 0 || thresholdPercent > 100) {
      setStatus("Порог срабатывания должен быть в диапазоне от 0% до 100%.", "error");
      return;
    }

    const threshold = thresholdPercent / 100;
    const selectedModel = modelSelect?.value || "";
    const allowedModels = getAllowedModelsFromSelect(modelSelect);
    if (!isModelAllowedForContent(selectedModel, allowedModels)) {
      setStatus("Выбранная модель не соответствует типу контента. Выберите модель из списка.", "error");
      return;
    }

    submitButton.disabled = true;
    submitButton.textContent = "Анализ...";
    if (contentType === "video") {
      setStatus("Отправка видео в модель...");
    } else if (contentType === "audio") {
      setStatus("Отправка аудио в модель...");
    } else {
      setStatus("Отправка изображения в модель...");
    }
    setSourceImageFromFile(file, contentType);

    await runDetection(() =>
      detectFile({
        file,
        model: selectedModel,
        threshold,
        contentType,
      })
    );

    submitButton.disabled = false;
    submitButton.textContent = mediaConfig.submitText;
  });

  urlForm?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const urlValue = (urlInput?.value || "").trim();
    const urlError = validateUrl(urlValue);
    if (urlError) {
      setStatus(urlError, "error");
      return;
    }

    const contentType = urlContentTypeSelect?.value || "photo";
    const mediaConfig = getMediaConfig(contentType);
    if (mediaConfig.comingSoon) {
      setStatus("Проверка аудио по URL будет добавлена позже.", "error");
      return;
    }

    const thresholdPercent = parseThresholdPercent(thresholdInputUrl?.value, contentType);
    if (!Number.isFinite(thresholdPercent) || thresholdPercent < 0 || thresholdPercent > 100) {
      setStatus("Порог срабатывания должен быть в диапазоне от 0% до 100%.", "error");
      return;
    }

    const threshold = thresholdPercent / 100;
    const selectedModel = urlModelSelect?.value || "";
    const allowedModels = getAllowedModelsFromSelect(urlModelSelect);
    if (!isModelAllowedForContent(selectedModel, allowedModels)) {
      setStatus("Выбранная модель не соответствует типу контента. Выберите модель из списка.", "error");
      return;
    }

    urlSubmitButton.disabled = true;
    urlSubmitButton.textContent = "Анализ...";
    setStatus("Скачивание файла по URL и анализ...");
    setSourceImageFromUrl(urlValue, contentType);

    await runDetection(() =>
      analyzeFileByUrl({
        url: urlValue,
        model: selectedModel,
        threshold,
        contentType,
      })
    );

    urlSubmitButton.disabled = false;
    urlSubmitButton.textContent = "Проверить по URL";
  });

  resetButton?.addEventListener("click", () => {
    clearSourceImageUrl();
    form?.reset();
    const contentType = fileContentTypeSelect?.value || "photo";
    setThresholdDefault(thresholdInput, contentType);
    syncFileUiByContentType(contentType, refs);
    populateModelSelect(modelSelect, contentType);
    setStatus("");
    toggleHidden("result", true);
    resetHeatmapView();
  });

  urlResetButton?.addEventListener("click", () => {
    clearSourceImageUrl();
    urlForm?.reset();
    const contentType = urlContentTypeSelect?.value || "photo";
    setThresholdDefault(thresholdInputUrl, contentType);
    populateModelSelect(urlModelSelect, contentType);
    setStatus("");
    toggleHidden("result", true);
    resetHeatmapView();
  });

  window.addEventListener("beforeunload", () => {
    clearSourceImageUrl();
  });
});
