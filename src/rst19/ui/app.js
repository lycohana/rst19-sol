const state = {
  frames: [],
  selected: null,
  result: null,
  preview: null,
};

const $ = (selector) => document.querySelector(selector);

function formatNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString("en-US") : "—";
}

function formatDecimal(value, digits = 2) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("visible"), 2600);
}

function addLog(code, message, muted = false) {
  const list = $("#log-list");
  const item = document.createElement("div");
  item.className = `log-item${muted ? " muted" : ""}`;
  item.innerHTML = `<span class="log-time">${code}</span><span></span>`;
  item.lastElementChild.textContent = message;
  list.prepend(item);
  while (list.children.length > 5) list.lastElementChild.remove();
}

function renderFrames() {
  const list = $("#frame-list");
  if (!list) return;
  list.replaceChildren();
  state.frames.forEach((frame, index) => {
    const button = document.createElement("button");
    button.className = `frame-item${frame.filename === state.selected ? " active" : ""}`;
    button.type = "button";
    button.dataset.filename = frame.filename;
    button.innerHTML = `<span class="frame-number">${String(index + 1).padStart(2, "0")}</span><span class="frame-name"></span><span class="frame-check">○</span>`;
    button.querySelector(".frame-name").textContent = frame.filename.replace("_9901.fits", "");
    button.addEventListener("click", () => selectFrame(frame.filename));
    list.appendChild(button);
  });
  $("#frame-count").textContent = state.frames.length;
  $("#frame-index").textContent = state.selected ? `FRAME ${state.frames.findIndex((frame) => frame.filename === state.selected) + 1} / ${state.frames.length}` : "FRAME — / —";
}

async function requestJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `请求失败：${response.status}`);
  return payload;
}

function setLoading(loading) {
  $("#stage-loading").classList.toggle("visible", loading);
  $("#run-button").disabled = loading;
  $("#run-button span:nth-child(2)").textContent = loading ? "正在分析…" : "运行当前帧";
}

function drawOverlay() {
  const image = $("#preview-image");
  const canvas = $("#detection-canvas");
  const result = state.result;
  if (!result || !image.naturalWidth) return;
  const rect = image.getBoundingClientRect();
  const stageRect = $("#image-stage").getBoundingClientRect();
  const scale = Math.min(rect.width / image.naturalWidth, rect.height / image.naturalHeight);
  const renderedWidth = image.naturalWidth * scale;
  const renderedHeight = image.naturalHeight * scale;
  const offsetX = rect.left - stageRect.left + (rect.width - renderedWidth) / 2;
  const offsetY = rect.top - stageRect.top + (rect.height - renderedHeight) / 2;
  canvas.width = stageRect.width * window.devicePixelRatio;
  canvas.height = stageRect.height * window.devicePixelRatio;
  canvas.style.width = `${stageRect.width}px`;
  canvas.style.height = `${stageRect.height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  context.clearRect(0, 0, stageRect.width, stageRect.height);
  const [height, width] = result.detection.image_shape;
  result.detection.sources.forEach((source) => {
    const x = offsetX + source.x / width * renderedWidth;
    const y = offsetY + source.y / height * renderedHeight;
    const radius = source.snr > 100 ? 3.1 : 2.1;
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fillStyle = source.snr > 100 ? "rgba(238, 177, 73, .92)" : "rgba(238, 177, 73, .66)";
    context.fill();
  });
}

function renderSources() {
  const tbody = $("#source-table");
  const sources = [...(state.result?.detection?.sources || [])].sort((a, b) => b.snr - a.snr).slice(0, 12);
  tbody.replaceChildren();
  if (!sources.length) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="6" class="empty-cell">当前参数没有返回候选源</td>`;
    tbody.appendChild(row);
    return;
  }
  sources.forEach((source) => {
    const row = document.createElement("tr");
    const flags = source.flags.length ? source.flags.join(" · ") : "—";
    row.innerHTML = `<td>${String(source.detection_id).padStart(4, "0")}</td><td>${formatDecimal(source.x, 1)} / ${formatDecimal(source.y, 1)}</td><td>${formatNumber(Math.round(source.peak))}</td><td>${formatDecimal(source.snr, 1)}</td><td>${formatDecimal(source.fwhm, 1)}</td><td>${flags}</td>`;
    tbody.appendChild(row);
  });
}

function renderResult() {
  const detection = state.result.detection;
  const truncated = detection.truncated;
  $("#candidate-count").textContent = formatNumber(detection.candidate_count);
  $("#candidate-foot").textContent = truncated ? "全候选峰 · 已限制绘制" : "全候选峰 · 全部返回";
  $("#returned-count").textContent = formatNumber(detection.returned_count);
  $("#returned-foot").textContent = truncated ? `绘制前 ${formatNumber(detection.returned_count)} 个` : "无绘制上限截断";
  $("#background-value").textContent = formatDecimal(detection.background, 2);
  $("#noise-value").textContent = formatDecimal(detection.noise, 2);
  $("#threshold-foot").textContent = `threshold ${formatDecimal(detection.threshold, 2)}`;
  $("#table-count").textContent = `${formatNumber(detection.returned_count)} returned`;
  $("#selected-frame-title").textContent = state.selected.replace("_9901.fits", "");
  renderFrames();
  renderSources();
  addLog("DONE", `${state.selected} · ${formatNumber(detection.candidate_count)} 候选峰`);
}

async function runAnalysis() {
  if (!state.selected) {
    showToast("请先从左侧选择一个 FITS 帧");
    return;
  }
  const params = new URLSearchParams({
    filename: state.selected,
    threshold_sigma: $("#threshold-sigma").value,
    min_distance: $("#min-distance").value,
    max_sources: $("#max-sources").value,
  });
  setLoading(true);
  addLog("RUN", `正在分析 ${state.selected}`);
  try {
    const [result, preview] = await Promise.all([
      requestJson(`/api/analyze?${params}`),
      requestJson(`/api/preview?filename=${encodeURIComponent(state.selected)}&max_side=960`),
    ]);
    state.result = result;
    state.preview = preview.data_url;
    const image = $("#preview-image");
    image.onload = () => {
      image.style.opacity = "1";
      $("#stage-placeholder").style.display = "none";
      drawOverlay();
    };
    image.src = state.preview;
    renderResult();
    showToast("当前帧分析完成");
  } catch (error) {
    addLog("ERR", error.message);
    showToast(error.message);
  } finally {
    setLoading(false);
  }
}

async function selectFrame(filename) {
  state.selected = filename;
  state.result = null;
  $("#selected-frame-title").textContent = filename.replace("_9901.fits", "");
  $("#stage-placeholder").style.display = "flex";
  $("#preview-image").style.opacity = "0";
  $("#preview-image").removeAttribute("src");
  $("#detection-canvas").getContext("2d")?.clearRect(0, 0, 2000, 2000);
  renderFrames();
  await runAnalysis();
}

async function loadFrames() {
  try {
    const payload = await requestJson("/api/frames");
    state.frames = payload.frames;
    if (!state.frames.length) throw new Error("数据目录中没有 FITS 文件");
    state.selected = state.frames[0].filename;
    renderFrames();
    await runAnalysis();
  } catch (error) {
    addLog("ERR", error.message);
    showToast(error.message);
  }
}

$("#analysis-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runAnalysis();
});

$("#threshold-sigma").addEventListener("input", (event) => {
  $("#threshold-output").value = `${Number(event.target.value).toFixed(1)} σ`;
});

$("#current-date").textContent = new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date());
window.addEventListener("resize", () => window.requestAnimationFrame(drawOverlay));
loadFrames();
