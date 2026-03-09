import { APP_CONFIG } from "./config.js";
import { detectImage, fetchModels } from "./api.js";
import { buildVerdict, safePercent, setStatus, setText, toggleHidden } from "./ui.js";

function validateFile(file) {
  if (!file) {
    return "Please choose an image file.";
  }

  const lowerName = (file.name || "").toLowerCase();
  const hasAllowedExtension = APP_CONFIG.SUPPORTED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
  const hasAllowedMime = (file.type || "").startsWith(APP_CONFIG.SUPPORTED_MIME_PREFIX);

  if (!hasAllowedExtension || !hasAllowedMime) {
    return `Unsupported file type. Allowed: ${APP_CONFIG.SUPPORTED_EXTENSIONS.join(", ")}`;
  }

  const maxBytes = APP_CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024;
  if (file.size > maxBytes) {
    return `File is too large. Max size is ${APP_CONFIG.MAX_FILE_SIZE_MB} MB.`;
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
    confidence: payload?.confidence,
    modelUsed: payload?.model_used || "N/A",
    threshold: payload?.threshold,
    raw: payload,
  };
}

function renderResult(result) {
  toggleHidden("result", false);
  setText("resultPercent", result.percent === null ? "N/A" : `${result.percent.toFixed(2)}%`);
  setText("resultVerdict", buildVerdict(result.percent, result.label));
  setText("resultLabel", result.label || "N/A");
  setText("resultConfidence", Number.isFinite(result.confidence) ? result.confidence.toFixed(4) : "N/A");
  setText("resultModel", result.modelUsed);
  setText("resultThreshold", Number.isFinite(result.threshold) ? String(result.threshold) : "N/A");
  setText("rawJson", JSON.stringify(result.raw, null, 2));
}

async function loadModels() {
  const modelSelect = document.getElementById("modelSelect");
  if (!modelSelect) {
    return;
  }

  try {
    const models = await fetchModels();
    if (!models.length) {
      modelSelect.innerHTML = '<option value="">No models available</option>';
      modelSelect.disabled = true;
      return;
    }

    modelSelect.innerHTML = '<option value="">Default model</option>';
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      modelSelect.appendChild(option);
    }
  } catch (error) {
    setStatus(`Could not load models: ${error.message}`, "error");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("detectForm");
  const fileInput = document.getElementById("fileInput");
  const submitButton = document.getElementById("submitButton");
  const resetButton = document.getElementById("resetButton");
  const thresholdInput = document.getElementById("thresholdInput");
  const modelSelect = document.getElementById("modelSelect");

  loadModels();

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus("");
    toggleHidden("result", true);

    const file = fileInput?.files?.[0];
    const validationError = validateFile(file);
    if (validationError) {
      setStatus(validationError, "error");
      return;
    }

    const threshold = Number(thresholdInput?.value ?? APP_CONFIG.DEFAULT_THRESHOLD);
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
      setStatus("Threshold must be between 0 and 1.", "error");
      return;
    }

    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";
    setStatus("Sending image to model...");

    try {
      const payload = await detectImage({
        file,
        model: modelSelect?.value || "",
        threshold,
      });

      const result = normalizeResult(payload);
      if (result.percent === null) {
        throw new Error("Server response does not contain probability fields.");
      }

      renderResult(result);
      setStatus("Detection completed successfully.", "success");
    } catch (error) {
      setStatus(error.message || "Unexpected error.", "error");
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = "Check Image";
    }
  });

  resetButton?.addEventListener("click", () => {
    form?.reset();
    setStatus("");
    toggleHidden("result", true);
  });
});

