import { protectFile } from "./api.js";
import { setStatus } from "./ui.js";
import { APP_CONFIG } from "./config.js";

function toggleVideoOptions(contentType) {
  const videoOptions = document.getElementById("videoOptions");
  if (!videoOptions) return;
  videoOptions.style.display = contentType === "video" ? "grid" : "none";

  const stepsRow = document.getElementById("protectSteps")?.closest?.(".form-row");
  const attackRow = document.getElementById("protectAttack")?.closest?.(".form-row");
  if (stepsRow) stepsRow.style.display = contentType === "audio" ? "none" : "grid";
  if (attackRow) attackRow.style.display = contentType === "audio" ? "none" : "grid";
}

function toggleStepsVisibility({ contentType, attack }) {
  const stepsRow = document.getElementById("protectSteps")?.closest?.(".form-row");
  if (!stepsRow) return;

  if (contentType === "audio") {
    stepsRow.style.display = "none";
    return;
  }

  stepsRow.style.display = attack === "pgd" ? "grid" : "none";
}

function applyAcceptFilter(input, contentType) {
  if (!input) return;
  const config = APP_CONFIG.SUPPORTED_MEDIA?.[contentType];
  if (!config) {
    input.removeAttribute("accept");
    return;
  }
  // Prefer explicit extensions (more reliable than MIME on Windows).
  input.accept = (config.extensions || []).join(",");
}

function validateProtectFile(file, contentType) {
  if (!file) {
    return "Выберите файл.";
  }
  const config = APP_CONFIG.SUPPORTED_MEDIA?.[contentType];
  if (!config) {
    return "Неподдерживаемый тип контента.";
  }
  const lowerName = (file.name || "").toLowerCase();
  const okExt = (config.extensions || []).some((ext) => lowerName.endsWith(ext));
  if (!okExt) {
    return `Неподдерживаемое расширение. Разрешено: ${(config.extensions || []).join(", ")}`;
  }
  const maxBytes = (APP_CONFIG.MAX_FILE_SIZE_MB || 20) * 1024 * 1024;
  if (file.size > maxBytes) {
    return `Файл слишком большой. Максимальный размер: ${APP_CONFIG.MAX_FILE_SIZE_MB || 20} МБ.`;
  }
  return "";
}

document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("protectInput");
  const button = document.getElementById("protectButton");
  const download = document.getElementById("protectDownload");
  const contentTypeSelect = document.getElementById("protectContentType");
  const attackSelect = document.getElementById("protectAttack");
  const epsInput = document.getElementById("protectEps");
  const stepsInput = document.getElementById("protectSteps");
  const maxFramesInput = document.getElementById("protectMaxFrames");

  toggleVideoOptions(contentTypeSelect?.value || "photo");
  applyAcceptFilter(fileInput, contentTypeSelect?.value || "photo");
  toggleStepsVisibility({
    contentType: contentTypeSelect?.value || "photo",
    attack: attackSelect?.value || "fgsm",
  });

  contentTypeSelect?.addEventListener("change", () => {
    toggleVideoOptions(contentTypeSelect.value);
    applyAcceptFilter(fileInput, contentTypeSelect.value);
    toggleStepsVisibility({ contentType: contentTypeSelect.value, attack: attackSelect?.value || "fgsm" });
  });

  attackSelect?.addEventListener("change", () => {
    toggleStepsVisibility({ contentType: contentTypeSelect?.value || "photo", attack: attackSelect.value });
  });

  let lastDownloadUrl = "";
  let lastContentType = "photo";

  async function downloadProtectedFile() {
    if (!lastDownloadUrl) {
      setStatus("Нет файла для скачивания.", "error");
      return;
    }
    try {
      setStatus("Скачивание...", "loading");
      download.disabled = true;
      const response = await fetch(lastDownloadUrl);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download =
        lastContentType === "video"
          ? "protected.mp4"
          : lastContentType === "audio"
            ? "protected.wav"
            : "protected.png";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setStatus("Файл скачан.", "success");
    } catch (e) {
      setStatus(e?.message || "Не удалось скачать файл.", "error");
    } finally {
      download.disabled = false;
    }
  }

  download?.addEventListener("click", (event) => {
    event.preventDefault();
    downloadProtectedFile();
  });

  button?.addEventListener("click", async (event) => {
    event.preventDefault();
    download.style.display = "none";
    lastDownloadUrl = "";
    lastContentType = "photo";

    const file = fileInput?.files?.[0];
    const contentType = contentTypeSelect?.value || "photo";

    const validationError = validateProtectFile(file, contentType);
    if (validationError) {
      setStatus(validationError, "error");
      return;
    }
    const attack = attackSelect?.value || "fgsm";
    const eps = parseFloat(epsInput?.value || "0.01");
    const steps = parseInt(stepsInput?.value || "10", 10);
    const maxFrames = parseInt(maxFramesInput?.value || "24", 10);

    try {
      setStatus("Обработка...", "loading");
      button.disabled = true;

      const result = await protectFile({
        file,
        contentType,
        attack,
        eps,
        steps,
        maxFrames,
      });

      if (!result?.download_url) {
        setStatus("Backend не вернул ссылку на скачивание.", "error");
        return;
      }

      const url = `${APP_CONFIG.API_BASE_URL}${result.download_url}`;
      lastDownloadUrl = url;
      lastContentType = contentType;
      download.style.display = "inline-flex";
      setStatus("Готово. Скачайте защищённый файл.", "success");
    } catch (e) {
      setStatus(e?.message || "Не удалось защитить файл.", "error");
    } finally {
      button.disabled = false;
    }
  });
});
