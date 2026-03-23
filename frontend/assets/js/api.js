import { APP_CONFIG } from "./config.js";

async function parseResponse(response) {
  const text = await response.text();
  let payload;

  try {
    payload = text ? JSON.parse(text) : {};
  } catch (_error) {
    payload = {};
  }

  if (!response.ok) {
    const detail = payload.detail || payload.error || `HTTP ${response.status}`;
    throw new Error(detail);
  }

  return payload;
}

async function fetchJson(endpoint, options = {}) {
  try {
    const response = await fetch(endpoint, options);
    return await parseResponse(response);
  } catch (error) {
    const message = String(error?.message || "");
    if (message.toLowerCase().includes("failed to fetch")) {
      throw new Error(
        `Нет соединения с backend (${APP_CONFIG.API_BASE_URL}). Проверьте, что API запущен и доступен.`
      );
    }
    throw error;
  }
}

export async function fetchModels() {
  const payload = await fetchJson(`${APP_CONFIG.API_BASE_URL}/models`);
  return Array.isArray(payload.models) ? payload.models : [];
}

export async function fetchModelsByContentType(contentType) {
  const params = new URLSearchParams();
  if (contentType) {
    params.set("content_type", contentType);
  }
  const query = params.toString();
  const endpoint = `${APP_CONFIG.API_BASE_URL}/models${query ? `?${query}` : ""}`;
  const payload = await fetchJson(endpoint);
  return {
    models: Array.isArray(payload.models) ? payload.models : [],
    defaultModel: payload.default_model || "",
  };
}

export async function detectFile({ file, model, threshold, contentType }) {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  if (model) {
    params.set("model", model);
  }
  if (typeof threshold === "number") {
    params.set("threshold", String(threshold));
  }
  if (contentType) {
    params.set("content_type", contentType);
  }

  const query = params.toString();
  const endpoint = `${APP_CONFIG.API_BASE_URL}/detect${query ? `?${query}` : ""}`;

  return fetchJson(endpoint, {
    method: "POST",
    body: formData,
  });
}

export async function analyzeFileByUrl({ url, model, threshold, contentType }) {
  return fetchJson(`${APP_CONFIG.API_BASE_URL}/predict/url`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      url,
      content_type: contentType,
      model: model || null,
      threshold,
    }),
  });
}

export async function protectFile({ file, contentType, model, attack, eps, steps, maxFrames }) {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  if (contentType) {
    params.set("content_type", contentType);
  }
  if (model) {
    params.set("model", model);
  }
  if (attack) {
    params.set("attack", attack);
  }
  if (typeof eps === "number" && !Number.isNaN(eps)) {
    params.set("eps", String(eps));
  }
  if (typeof steps === "number" && !Number.isNaN(steps)) {
    params.set("steps", String(steps));
  }
  if (contentType === "video" && typeof maxFrames === "number" && !Number.isNaN(maxFrames)) {
    params.set("max_frames", String(maxFrames));
  }

  const query = params.toString();
  const endpoint = `${APP_CONFIG.API_BASE_URL}/protect${query ? `?${query}` : ""}`;
  return fetchJson(endpoint, {
    method: "POST",
    body: formData,
  });
}

