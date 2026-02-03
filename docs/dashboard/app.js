const canvases = {
  network: document.getElementById("networkCanvas"),
  raster: document.getElementById("rasterCanvas"),
  heatmapInput: document.getElementById("heatmapInput"),
  heatmapOutput: document.getElementById("heatmapOutput"),
  rate: document.getElementById("rateCanvas"),
  weight: document.getElementById("weightCanvas"),
  state: document.getElementById("stateCanvas"),
};

const metricNodes = {
  fps: document.getElementById("fps"),
  activeEdges: document.getElementById("activeEdges"),
  spikeRate: document.getElementById("spikeRate"),
  list: document.getElementById("metricList"),
  status: document.getElementById("dataStatus"),
  statusDot: document.getElementById("dataStatusDot"),
};

const theme = {
  background: "#0f1529",
  grid: "rgba(255,255,255,0.06)",
  text: "#e8eefc",
  muted: "#98a4be",
  input: "#60a5fa",
  hidden: "#6ee7b7",
  output: "#f97316",
  excit: "#38bdf8",
  inhib: "#fb7185",
  accent: "#5ddcff",
  heat: ["#13172b", "#334155", "#7c3aed", "#22d3ee", "#facc15"],
};

const dataConfig = (() => {
  const params = new URLSearchParams(window.location.search);
  return {
    neuronCsv: params.get("neuron") || "data/neuron.csv",
    synapseCsv: params.get("synapse") || "data/synapse.csv",
    topologyJson: params.get("topology") || "data/topology.json",
    refreshMs: Number(params.get("refresh") || 1200),
  };
})();

const dataState = {
  neuronRows: null,
  synapseRows: null,
  topology: null,
  live: false,
  lastUpdated: null,
};

let network = buildNetwork();
let raster = buildRaster(60, 140);
let lastFrame = performance.now();
let fpsSmooth = 48;

function resizeAll() {
  Object.values(canvases).forEach((canvas) => setupCanvas(canvas));
}

function setupCanvas(canvas) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function buildNetwork() {
  const layerSizes = [9, 12, 8];
  const layerX = [0.15, 0.52, 0.88];
  const nodes = [];
  const edges = [];

  layerSizes.forEach((count, layerIdx) => {
    for (let i = 0; i < count; i += 1) {
      nodes.push({
        layer: layerIdx,
        index: i,
        x: layerX[layerIdx],
        y: (i + 1) / (count + 1),
      });
    }
  });

  const layerOffsets = [0, layerSizes[0], layerSizes[0] + layerSizes[1]];

  for (let i = 0; i < layerSizes[0]; i += 1) {
    for (let j = 0; j < layerSizes[1]; j += 1) {
      edges.push({
        from: layerOffsets[0] + i,
        to: layerOffsets[1] + j,
        weight: randWeight(),
      });
    }
  }
  for (let i = 0; i < layerSizes[1]; i += 1) {
    for (let j = 0; j < layerSizes[2]; j += 1) {
      edges.push({
        from: layerOffsets[1] + i,
        to: layerOffsets[2] + j,
        weight: randWeight(),
      });
    }
  }

  return { nodes, edges, layerSizes };
}

function normalizeTopology(topology) {
  if (!topology || !Array.isArray(topology.nodes) || !Array.isArray(topology.edges)) {
    return null;
  }
  const nodes = topology.nodes.map((node, idx) => ({
    index: idx,
    x: clamp01(node.x ?? Math.random()),
    y: clamp01(node.y ?? Math.random()),
    layer: Number(node.layer ?? 1),
  }));
  const edges = topology.edges.map((edge) => ({
    from: edge.from ?? 0,
    to: edge.to ?? 0,
    weight: Number(edge.weight ?? 0),
    receptor: edge.receptor || "ampa",
  }));
  return { nodes, edges, layerSizes: [] };
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value)));
}

function buildRaster(rows, cols) {
  const data = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() < 0.05 ? 1 : 0))
  );
  return { rows, cols, data };
}

function randWeight() {
  return Math.random() * 2 - 1;
}

function updateMetrics() {
  metricNodes.fps.textContent = fpsSmooth.toFixed(0);
  metricNodes.activeEdges.textContent = network.edges.length.toString();

  const spikeRateValue = latestValue(dataState.neuronRows, "spike_rate_hz");
  metricNodes.spikeRate.textContent = spikeRateValue
    ? `${spikeRateValue.toFixed(1)} Hz`
    : `${(Math.random() * 60 + 20).toFixed(1)} Hz`;

  const metrics = [
    ["Current Accuracy", latestValue(dataState.neuronRows, "accuracy") || "0.0%"],
    [
      "Weight Mean",
      latestValue(dataState.synapseRows, "weights_mean")
        ? `${latestValue(dataState.synapseRows, "weights_mean").toFixed(3)}`
        : `${(Math.random() * 0.6 + 0.2).toFixed(3)} ? ${(Math.random() * 0.3).toFixed(3)}`,
    ],
    [
      "Avg Spike Rate",
      latestValue(dataState.neuronRows, "spike_rate_hz")
        ? `${latestValue(dataState.neuronRows, "spike_rate_hz").toFixed(1)} Hz`
        : `${(Math.random() * 120 + 60).toFixed(1)} Hz`,
    ],
    [
      "Active Neurons",
      latestValue(dataState.neuronRows, "active_neurons") || `${Math.floor(Math.random() * 12) + 28}/45`,
    ],
    [
      "Progress",
      latestValue(dataState.neuronRows, "progress") || `${Math.floor(Math.random() * 40) + 20}/32000`,
    ],
  ];

  metricNodes.list.innerHTML = metrics
    .map(([label, value]) => `<li>${label}: <span>${value}</span></li>`)
    .join("");

  if (dataState.live) {
    metricNodes.status.textContent = "Live data";
    metricNodes.statusDot.parentElement.classList.add("live");
  } else {
    metricNodes.status.textContent = "Demo data";
    metricNodes.statusDot.parentElement.classList.remove("live");
  }
}

function updateRaster() {
  if (dataState.neuronRows) {
    const nextRaster = buildRasterFromRows(dataState.neuronRows, 120);
    if (nextRaster) {
      raster = nextRaster;
      return;
    }
  }
  for (let r = 0; r < raster.rows; r += 1) {
    const row = raster.data[r];
    row.shift();
    row.push(Math.random() < 0.08 + r / (raster.rows * 50) ? 1 : 0);
  }
}

function buildRasterFromRows(rows, maxCols) {
  const lastRow = rows[rows.length - 1];
  if (!lastRow) return null;
  const spikeKeys = Object.keys(lastRow).filter((key) => key.startsWith("spike_i"));
  if (spikeKeys.length === 0) return null;
  const keys = spikeKeys.sort((a, b) => Number(a.slice(7)) - Number(b.slice(7)));
  const cols = Math.min(maxCols, rows.length);
  const slice = rows.slice(-cols);
  const data = keys.map((key) =>
    slice.map((row) => (Number(row[key] || 0) > 0 ? 1 : 0))
  );
  return { rows: keys.length, cols, data };
}

function drawNetwork() {
  const ctx = setupCanvas(canvases.network);
  if (!ctx) return;
  const { width, height } = canvases.network.getBoundingClientRect();

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  ctx.lineWidth = 1;
  network.edges.forEach((edge) => {
    const from = network.nodes[edge.from];
    const to = network.nodes[edge.to];
    if (!from || !to) return;
    const weight = edge.weight || 0;
    const color = weight >= 0 ? theme.excit : theme.inhib;
    ctx.strokeStyle = color;
    ctx.globalAlpha = Math.min(0.6, Math.abs(weight)) + 0.1;
    ctx.beginPath();
    ctx.moveTo(from.x * width, from.y * height);
    ctx.lineTo(to.x * width, to.y * height);
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  network.nodes.forEach((node) => {
    const radius = node.layer === 1 ? 4.5 : 4;
    const fill = node.layer === 0 ? theme.input : node.layer === 1 ? theme.hidden : theme.output;
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(node.x * width, node.y * height, radius, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawRaster() {
  const ctx = setupCanvas(canvases.raster);
  if (!ctx) return;
  const { width, height } = canvases.raster.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const cellW = width / raster.cols;
  const cellH = height / raster.rows;

  for (let r = 0; r < raster.rows; r += 1) {
    for (let c = 0; c < raster.cols; c += 1) {
      if (raster.data[r][c] === 1) {
        ctx.fillStyle = r < 20 ? theme.input : r < 45 ? theme.hidden : theme.output;
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
      }
    }
  }
}

function drawHeatmap(canvas, rows, cols, values) {
  const ctx = setupCanvas(canvas);
  if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const cellW = width / cols;
  const cellH = height / rows;
  let idx = 0;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const value = values ? values[idx % values.length] : Math.random();
      const color = heatColor(value);
      ctx.fillStyle = color;
      ctx.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);
      idx += 1;
    }
  }
}

function heatColor(value) {
  const palette = theme.heat;
  const idx = Math.min(palette.length - 1, Math.floor(value * palette.length));
  return palette[idx];
}

function drawBars(canvas, count, color, values) {
  const ctx = setupCanvas(canvas);
  if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const gap = 3;
  const barWidth = (width - gap * (count - 1)) / count;
  for (let i = 0; i < count; i += 1) {
    const value = values ? values[i % values.length] : Math.random();
    const barHeight = value * (height - 12) + 6;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.6 + value * 0.4;
    ctx.fillRect(i * (barWidth + gap), height - barHeight, barWidth, barHeight);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = theme.accent;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.moveTo(0, height * 0.45);
  ctx.lineTo(width, height * 0.45);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawHistogram(canvas, values) {
  const ctx = setupCanvas(canvas);
  if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const bins = 18;
  const raw = values && values.length ? values : Array.from({ length: bins }, () => Math.random());
  const bucket = Array.from({ length: bins }, () => 0);
  raw.forEach((value) => {
    const idx = Math.min(bins - 1, Math.floor(((value + 1) / 2) * bins));
    bucket[idx] += 1;
  });

  const maxVal = Math.max(...bucket);
  const barWidth = width / bins;

  bucket.forEach((value, i) => {
    const barHeight = maxVal ? (value / maxVal) * (height - 8) : 0;
    ctx.fillStyle = theme.accent;
    ctx.globalAlpha = 0.3 + (value / (maxVal || 1)) * 0.6;
    ctx.fillRect(i * barWidth + 2, height - barHeight, barWidth - 4, barHeight);
  });
  ctx.globalAlpha = 1;
}

function drawStateSpace() {
  const ctx = setupCanvas(canvases.state);
  if (!ctx) return;
  const { width, height } = canvases.state.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = theme.grid;
  for (let i = 1; i < 4; i += 1) {
    ctx.beginPath();
    ctx.moveTo((width / 4) * i, 0);
    ctx.lineTo((width / 4) * i, height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, (height / 4) * i);
    ctx.lineTo(width, (height / 4) * i);
    ctx.stroke();
  }

  for (let i = 0; i < 80; i += 1) {
    const x = Math.random() * width;
    const y = Math.random() * height;
    ctx.fillStyle = i % 3 === 0 ? theme.output : i % 2 === 0 ? theme.hidden : theme.input;
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

async function refreshData() {
  const [neuronRows, synapseRows, topology] = await Promise.all([
    loadCsv(dataConfig.neuronCsv),
    loadCsv(dataConfig.synapseCsv),
    loadJson(dataConfig.topologyJson),
  ]);

  dataState.neuronRows = neuronRows;
  dataState.synapseRows = synapseRows;
  dataState.topology = topology;
  dataState.live = Boolean(neuronRows || synapseRows || topology);
  dataState.lastUpdated = new Date();

  if (topology) {
    const normalized = normalizeTopology(topology);
    if (normalized) {
      network = normalized;
    }
  }
}

async function loadCsv(path) {
  try {
    const response = await fetch(`${path}?ts=${Date.now()}`);
    if (!response.ok) return null;
    const text = await response.text();
    return parseCsv(text);
  } catch (error) {
    return null;
  }
}

async function loadJson(path) {
  try {
    const response = await fetch(`${path}?ts=${Date.now()}`);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    return null;
  }
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return null;
  const headers = splitCsvLine(lines[0]);
  const rows = lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    const row = {};
    headers.forEach((header, idx) => {
      row[header] = values[idx] ?? "";
    });
    return row;
  });
  return rows;
}

function splitCsvLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current);
  return result.map((value) => value.trim());
}

function latestValue(rows, key) {
  if (!rows || rows.length === 0) return null;
  const value = rows[rows.length - 1][key];
  if (value === undefined || value === null || value === "") return null;
  const asNumber = Number(value);
  return Number.isNaN(asNumber) ? value : asNumber;
}

function extractSampleValues(rows, prefix, max) {
  if (!rows || rows.length === 0) return null;
  const row = rows[rows.length - 1];
  const keys = Object.keys(row).filter((key) => key.startsWith(prefix));
  if (keys.length === 0) return null;
  keys.sort((a, b) => Number(a.slice(prefix.length)) - Number(b.slice(prefix.length)));
  const values = keys.map((key) => Number(row[key] || 0));
  return max ? values.slice(0, max) : values;
}

function tick(now) {
  const delta = now - lastFrame;
  lastFrame = now;
  const fps = 1000 / Math.max(delta, 1);
  fpsSmooth = fpsSmooth * 0.9 + fps * 0.1;

  updateRaster();
  updateMetrics();
  drawNetwork();
  drawRaster();

  const weightSamples = extractSampleValues(dataState.synapseRows, "weights_i", 120);
  drawHeatmap(canvases.heatmapInput, 16, 18, weightSamples);
  drawHeatmap(canvases.heatmapOutput, 10, 12, weightSamples);

  const rateSamples = extractSampleValues(dataState.neuronRows, "spike_i", 32);
  drawBars(canvases.rate, 32, theme.danger, rateSamples);
  drawHistogram(canvases.weight, weightSamples);
  drawStateSpace();

  requestAnimationFrame(tick);
}

window.addEventListener("resize", resizeAll);
resizeAll();
refreshData();
setInterval(refreshData, dataConfig.refreshMs);
requestAnimationFrame(tick);
