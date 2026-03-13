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

export function validateHeatmap2D(heatmap) {
  if (!Array.isArray(heatmap) || !heatmap.length || !Array.isArray(heatmap[0]) || !heatmap[0].length) {
    return false;
  }

  const width = heatmap[0].length;
  for (const row of heatmap) {
    if (!Array.isArray(row) || row.length !== width) {
      return false;
    }
    for (const value of row) {
      if (!Number.isFinite(Number(value))) {
        return false;
      }
    }
  }

  return true;
}

function toColor(value) {
  const t = Math.max(0, Math.min(255, Number(value))) / 255;

  // Simple heat palette: blue -> cyan -> yellow -> red.
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 * t - 0.5)));
  const g = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(2 * t - 1.0))));
  const b = Math.round(255 * Math.min(1, Math.max(0, 1.2 - 1.5 * t)));

  return [r, g, b];
}

export function drawHeatmapToCanvas(canvasId, heatmap) {
  if (!validateHeatmap2D(heatmap)) {
    return false;
  }

  const canvas = document.getElementById(canvasId);
  if (!(canvas instanceof HTMLCanvasElement)) {
    return false;
  }

  const height = heatmap.length;
  const width = heatmap[0].length;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return false;
  }

  const imageData = ctx.createImageData(width, height);
  const buffer = imageData.data;

  let offset = 0;
  for (let y = 0; y < height; y += 1) {
    const row = heatmap[y];
    for (let x = 0; x < width; x += 1) {
      const [r, g, b] = toColor(row[x]);
      buffer[offset] = r;
      buffer[offset + 1] = g;
      buffer[offset + 2] = b;
      buffer[offset + 3] = 255;
      offset += 4;
    }
  }

  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(imageData, 0, 0);
  return true;
}

export function resetHeatmapView() {
  toggleHidden("heatmapSection", true);
  toggleHidden("regionsSection", true);
  setText("heatmapMeta", "");
  setText("heatmapFrameIndicator", "");
  setText("regionsMeta", "");
}

export function findSuspiciousRegions(heatmap) {
  if (!validateHeatmap2D(heatmap)) {
    return [];
  }

  const height = heatmap.length;
  const width = heatmap[0].length;
  const values = [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      values.push(Math.max(0, Math.min(255, Number(heatmap[y][x]))));
    }
  }

  values.sort((a, b) => a - b);
  const percentileIndex = Math.floor(values.length * 0.84);
  const threshold = values[Math.min(values.length - 1, Math.max(0, percentileIndex))];

  const visited = new Uint8Array(width * height);
  const regions = [];
  const minPixels = Math.max(10, Math.floor((width * height) * 0.0012));
  const directions = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ];

  const indexOf = (x, y) => y * width + x;
  const maxRegionW = Math.max(24, Math.floor(width * 0.26));
  const maxRegionH = Math.max(24, Math.floor(height * 0.26));
  const maxRegionArea = Math.max(220, Math.floor(width * height * 0.09));

  function compactRegion(region) {
    const centerX = region.x + Math.floor(region.width / 2);
    const centerY = region.y + Math.floor(region.height / 2);
    let targetW = Math.min(region.width, maxRegionW);
    let targetH = Math.min(region.height, maxRegionH);

    const area = targetW * targetH;
    if (area > maxRegionArea) {
      const ratio = Math.sqrt(maxRegionArea / area);
      targetW = Math.max(10, Math.floor(targetW * ratio));
      targetH = Math.max(10, Math.floor(targetH * ratio));
    }

    const x = Math.max(0, Math.min(width - targetW, centerX - Math.floor(targetW / 2)));
    const y = Math.max(0, Math.min(height - targetH, centerY - Math.floor(targetH / 2)));

    return {
      ...region,
      x,
      y,
      width: targetW,
      height: targetH,
    };
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (Number(heatmap[y][x]) < threshold) {
        continue;
      }

      const startIndex = indexOf(x, y);
      if (visited[startIndex]) {
        continue;
      }

      const queue = [[x, y]];
      visited[startIndex] = 1;

      let qIndex = 0;
      let minX = x;
      let minY = y;
      let maxX = x;
      let maxY = y;
      let pixels = 0;
      let sumScore = 0;

      while (qIndex < queue.length) {
        const [cx, cy] = queue[qIndex];
        qIndex += 1;

        const value = Number(heatmap[cy][cx]);
        pixels += 1;
        sumScore += value;
        minX = Math.min(minX, cx);
        minY = Math.min(minY, cy);
        maxX = Math.max(maxX, cx);
        maxY = Math.max(maxY, cy);

        for (const [dx, dy] of directions) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            continue;
          }

          const nIndex = indexOf(nx, ny);
          if (visited[nIndex] || Number(heatmap[ny][nx]) < threshold) {
            continue;
          }

          visited[nIndex] = 1;
          queue.push([nx, ny]);
        }
      }

      if (pixels < minPixels) {
        continue;
      }

      regions.push({
        x: minX,
        y: minY,
        width: maxX - minX + 1,
        height: maxY - minY + 1,
        score: sumScore / pixels,
        pixels,
      });
    }
  }

  regions.sort((a, b) => b.score - a.score);
  if (regions.length) {
    return regions.slice(0, 5).map(compactRegion);
  }

  // Fallback: берём небольшие окна вокруг самых горячих точек, если связные области не найдены.
  const hotspots = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const score = Number(heatmap[y][x]);
      if (score >= threshold) {
        hotspots.push({ x, y, score });
      }
    }
  }

  hotspots.sort((a, b) => b.score - a.score);
  const winW = Math.max(12, Math.floor(width * 0.14));
  const winH = Math.max(12, Math.floor(height * 0.14));
  const fallbackRegions = [];

  for (const spot of hotspots) {
    const x = Math.max(0, Math.min(width - winW, spot.x - Math.floor(winW / 2)));
    const y = Math.max(0, Math.min(height - winH, spot.y - Math.floor(winH / 2)));

    const overlaps = fallbackRegions.some((region) => {
      const ix = Math.max(region.x, x);
      const iy = Math.max(region.y, y);
      const ax = Math.min(region.x + region.width, x + winW);
      const ay = Math.min(region.y + region.height, y + winH);
      return ax > ix && ay > iy;
    });

    if (!overlaps) {
      fallbackRegions.push({
        x,
        y,
        width: winW,
        height: winH,
        score: spot.score,
      });
    }

    if (fallbackRegions.length >= 3) {
      break;
    }
  }

  return fallbackRegions.map(compactRegion);
}

export async function drawImageRegions(canvasId, imageUrl, regions, heatmapWidth, heatmapHeight) {
  const canvas = document.getElementById(canvasId);
  if (!(canvas instanceof HTMLCanvasElement) || !imageUrl) {
    return false;
  }

  const image = new Image();
  image.decoding = "async";
  image.referrerPolicy = "no-referrer";
  image.crossOrigin = "anonymous";

  const loaded = await new Promise((resolve) => {
    image.onload = () => resolve(true);
    image.onerror = () => resolve(false);
    image.src = imageUrl;
  });

  if (!loaded) {
    return false;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return false;
  }

  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  const sourceW = Number.isFinite(Number(heatmapWidth)) && Number(heatmapWidth) > 0 ? Number(heatmapWidth) : 256;
  const sourceH = Number.isFinite(Number(heatmapHeight)) && Number(heatmapHeight) > 0 ? Number(heatmapHeight) : 256;
  const sx = canvas.width / sourceW;
  const sy = canvas.height / sourceH;

  ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 180));
  ctx.strokeStyle = "#ef4444";
  ctx.fillStyle = "rgba(239, 68, 68, 0.18)";

  for (let i = 0; i < regions.length; i += 1) {
    const region = regions[i];
    const x = Math.max(0, Math.round(region.x * sx));
    const y = Math.max(0, Math.round(region.y * sy));
    const w = Math.max(1, Math.round(region.width * sx));
    const h = Math.max(1, Math.round(region.height * sy));

    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }

  return true;
}

