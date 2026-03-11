export function setText(id, text) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = text;
  }
}

export function toggleHidden(id, shouldHide) {
  const el = document.getElementById(id);
  if (!el) {
    return;
  }
  el.classList.toggle("hidden", shouldHide);
}

export function setStatus(message, type = "") {
  const el = document.getElementById("status");
  if (!el) {
    return;
  }

  el.textContent = message || "";
  el.classList.remove("error", "success");
  if (type) {
    el.classList.add(type);
  }
}

export function safePercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  return Math.max(0, Math.min(100, numeric));
}

export function buildVerdict(percent, label) {
  if (label === "deepfake") {
    return "Вероятно дипфейк";
  }
  if (label === "real") {
    return "Вероятно оригинал";
  }
  if (percent === null) {
    return "Неизвестно";
  }
  return percent >= 50 ? "Вероятно дипфейк" : "Вероятно оригинал";
}
