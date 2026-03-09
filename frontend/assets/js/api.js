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

export async function fetchModels() {
  const response = await fetch(`${APP_CONFIG.API_BASE_URL}/models`);
  const payload = await parseResponse(response);
  return Array.isArray(payload.models) ? payload.models : [];
}

export async function detectImage({ file, model, threshold }) {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  if (model) {
    params.set("model", model);
  }
  if (typeof threshold === "number") {
    params.set("threshold", String(threshold));
  }

  const query = params.toString();
  const endpoint = `${APP_CONFIG.API_BASE_URL}/detect${query ? `?${query}` : ""}`;

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  return parseResponse(response);
}

