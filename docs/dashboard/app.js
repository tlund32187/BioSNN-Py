const canvases = {
  network: document.getElementById("networkCanvas"),
  raster: document.getElementById("rasterCanvas"),
  accuracy: document.getElementById("accuracyCanvas"),
  heatmapInput: document.getElementById("heatmapInput"),
  heatmapOutput: document.getElementById("heatmapOutput"),
  rate: document.getElementById("rateCanvas"),
  weight: document.getElementById("weightCanvas"),
  state: document.getElementById("stateCanvas"),
};

const networkControls = {
  viewSelect: document.getElementById("networkViewSelect"),
  tooltip: document.getElementById("networkTooltip"),
};

const uiControls = {
  rasterStride: document.getElementById("rasterStride"),
  rasterWindow: document.getElementById("rasterWindow"),
  rasterMaxPoints: document.getElementById("rasterMaxPoints"),
  weightProjection: document.getElementById("weightProjectionSelect"),
  weightStep: document.getElementById("weightStepSelect"),
  weightClampMin: document.getElementById("weightClampMin"),
  weightClampMax: document.getElementById("weightClampMax"),
  weightMaxDim: document.getElementById("weightMaxDim"),
  smoothWindow: document.getElementById("smoothWindow"),
};

const metricNodes = {
  fps: document.getElementById("fps"),
  activeEdges: document.getElementById("activeEdges"),
  spikeRate: document.getElementById("spikeRate"),
  list: document.getElementById("metricList"),
  status: document.getElementById("dataStatus"),
  statusDot: document.getElementById("dataStatusDot"),
  trainAccLatest: document.getElementById("trainAccLatest"),
  evalAccLatest: document.getElementById("evalAccLatest"),
  lossLatest: document.getElementById("lossLatest"),
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
  modelGlif: "#60a5fa",
  modelAdex: "#f59e0b",
  modelUnknown: "#94a3b8",
  heat: ["#13172b", "#334155", "#7c3aed", "#22d3ee", "#facc15"],
};

const dataConfig = (() => {
  const params = new URLSearchParams(window.location.search);
  return {
    neuronCsv: params.get("neuron") || "data/neuron.csv",
    synapseCsv: params.get("synapse") || "data/synapse.csv",
    spikesCsv: params.get("spikes") || "data/spikes.csv",
    metricsCsv: params.get("metrics") || "data/metrics.csv",
    weightsCsv: params.get("weights") || "data/weights.csv",
    topologyJson: params.get("topology") || "data/topology.json",
    refreshMs: Number(params.get("refresh") || 1200),
  };
})();

const dataState = {
  neuronRows: null,
  synapseRows: null,
  spikesRows: null,
  metricsRows: null,
  weightsRows: null,
  weightsIndex: null,
  topology: null,
  topologyMode: "neurons",
  live: false,
  lastUpdated: null,
};

let network = buildNetwork();
let networkNeuron = network;
let networkPopulation = null;
let networkView = "neurons";
let userViewSelection = false;
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



function normalizePopulationTopology(topology) {
  if (!topology || !Array.isArray(topology.nodes) || !Array.isArray(topology.edges)) {
    return null;
  }
  const nodes = topology.nodes.map((node, idx) => ({
    id: node.id ?? node.label ?? `pop_${idx}`,
    label: node.label ?? node.id ?? `Pop ${idx + 1}`,
    layer: Number(node.layer ?? 0),
    nNeurons: Number(node.n_neurons ?? node.n ?? 0),
    model: String(node.model ?? "unknown").toLowerCase(),
    x: clamp01(node.x ?? Math.random()),
    y: clamp01(node.y ?? Math.random()),
  }));
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const edges = topology.edges.map((edge, idx) => ({
    id: edge.id ?? `${edge.from ?? "?"}->${edge.to ?? "?"}-${idx}`,
    from: edge.from,
    to: edge.to,
    nSynapses: Number(edge.n_synapses ?? 0),
    meanWeight: Number(edge.mean_weight ?? 0),
    stdWeight: Number(edge.std_weight ?? 0),
    meanDelay: edge.mean_delay_steps !== undefined ? Number(edge.mean_delay_steps) : null,
    receptorCounts: edge.receptor_counts ?? null,
    targetCounts: edge.target_counts ?? null,
  }));
  return { mode: "population", nodes, edges, nodeMap };
}

function resolveNetworkView() {
  const defaultView = dataState.topologyMode === "populations" ? "populations" : "neurons";
  if (!userViewSelection) {
    networkView = defaultView;
    if (networkControls.viewSelect) {
      networkControls.viewSelect.value = networkView;
    }
  }
  if (networkView === "populations" && networkPopulation) {
    network = networkPopulation;
    return;
  }
  if (networkView === "neurons" && networkNeuron) {
    network = networkNeuron;
    return;
  }
  network = buildNetwork();
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

  const spikeRateValue = latestValue(dataState.metricsRows, "spike_fraction_total");
  metricNodes.spikeRate.textContent = spikeRateValue
    ? `${(Number(spikeRateValue) * 100).toFixed(1)}%`
    : `${(Math.random() * 60 + 20).toFixed(1)} Hz`;

  const metrics = [
    ["Current Accuracy", latestValue(dataState.metricsRows, "train_accuracy") || "0.0%"],
    [
      "Weight Mean",
      latestValue(dataState.synapseRows, "weights_mean")
        ? `${latestValue(dataState.synapseRows, "weights_mean").toFixed(3)}`
        : `${(Math.random() * 0.6 + 0.2).toFixed(3)} +/- ${(Math.random() * 0.3).toFixed(3)}`,
    ],
    [
      "Avg Spike Rate",
      latestValue(dataState.metricsRows, "spike_fraction_total")
        ? `${(Number(latestValue(dataState.metricsRows, "spike_fraction_total")) * 100).toFixed(1)}%`
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
  if (networkView === "populations" && networkPopulation) {
    drawPopulationNetwork();
    return;
  }
  drawNeuronNetwork();
}

function drawNeuronNetwork() {
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


function drawPopulationNetwork() {
  const ctx = setupCanvas(canvases.network);
  if (!ctx || !networkPopulation) return;
  const { width, height } = canvases.network.getBoundingClientRect();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const nodes = networkPopulation.nodes;
  const nodeMap = networkPopulation.nodeMap || new Map(nodes.map((node) => [node.id, node]));
  const edges = networkPopulation.edges;

  const screenNodes = [];
  const screenEdges = [];

  const radiusFor = (n) => {
    const base = 10;
    const scale = 2.4;
    const r = base + scale * Math.sqrt(Math.max(n, 0));
    return Math.min(28, Math.max(8, r));
  };

  edges.forEach((edge) => {
    const from = nodeMap.get(edge.from);
    const to = nodeMap.get(edge.to);
    if (!from || !to) return;
    const x1 = from.x * width;
    const y1 = from.y * height;
    const x2 = to.x * width;
    const y2 = to.y * height;
    const thickness = Math.min(6, 0.6 + 1.4 * Math.log1p(edge.nSynapses || 0));
    const receptorCounts = edge.receptorCounts || {};
    const gabaCount = receptorCounts.gaba ?? receptorCounts.GABA ?? 0;
    const color = gabaCount > 0 ? theme.inhib : theme.excit;

    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = thickness;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    screenEdges.push({
      edge,
      x1,
      y1,
      x2,
      y2,
      thickness,
    });
  });
  ctx.globalAlpha = 1;

  nodes.forEach((node) => {
    const x = node.x * width;
    const y = node.y * height;
    const radius = radiusFor(node.nNeurons);
    let fill = theme.modelUnknown;
    if (node.model.includes("glif")) {
      fill = theme.modelGlif;
    } else if (node.model.includes("adex")) {
      fill = theme.modelAdex;
    }
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = theme.text;
    ctx.font = "11px 'Space Grotesk'";
    ctx.textAlign = "center";
    ctx.fillText(node.label, x, y - radius - 6);

    screenNodes.push({ node, x, y, radius });
  });

  networkPopulation._screen = { nodes: screenNodes, edges: screenEdges };
}

function drawRaster() {
  if (dataState.spikesRows && dataState.spikesRows.length) {
    drawSpikeRasterFromEvents(dataState.spikesRows);
    return;
  }
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
  const [neuronRows, synapseRows, spikesRows, metricsRows, weightsRows, topology] = await Promise.all([
    loadCsv(dataConfig.neuronCsv),
    loadCsv(dataConfig.synapseCsv),
    loadCsv(dataConfig.spikesCsv),
    loadCsv(dataConfig.metricsCsv),
    loadCsv(dataConfig.weightsCsv),
    loadJson(dataConfig.topologyJson),
  ]);

  dataState.neuronRows = neuronRows;
  dataState.synapseRows = synapseRows;
  dataState.spikesRows = spikesRows;
  dataState.metricsRows = metricsRows;
  dataState.weightsRows = weightsRows;
  dataState.weightsIndex = weightsRows ? buildWeightsIndex(weightsRows) : null;
  dataState.topology = topology;
  dataState.live = Boolean(neuronRows || synapseRows || spikesRows || metricsRows || weightsRows || topology);
  dataState.lastUpdated = new Date();

  if (topology) {
    if (topology.mode === "population") {
      networkPopulation = normalizePopulationTopology(topology);
      dataState.topologyMode = "populations";
    } else {
      networkNeuron = normalizeTopology(topology);
      dataState.topologyMode = "neurons";
    }
    resolveNetworkView();
  }

  refreshWeightSelectors();
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


function showNetworkTooltip(html, x, y) {
  if (!networkControls.tooltip) return;
  networkControls.tooltip.innerHTML = html;
  networkControls.tooltip.style.left = `${x + 12}px`;
  networkControls.tooltip.style.top = `${y + 12}px`;
  networkControls.tooltip.classList.add("visible");
}

function hideNetworkTooltip() {
  if (!networkControls.tooltip) return;
  networkControls.tooltip.classList.remove("visible");
}

function onNetworkHover(event) {
  if (networkView !== "populations" || !networkPopulation || !networkPopulation._screen) {
    hideNetworkTooltip();
    return;
  }
  const rect = canvases.network.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;
  const screen = networkPopulation._screen;

  for (const item of screen.nodes) {
    const dx = mx - item.x;
    const dy = my - item.y;
    if (Math.sqrt(dx * dx + dy * dy) <= item.radius + 4) {
      const node = item.node;
      showNetworkTooltip(
        `<strong>${node.label}</strong><br/>` +
          `Neurons: ${node.nNeurons}<br/>` +
          `Model: ${node.model}<br/>` +
          `Layer: ${node.layer}`,
        mx,
        my
      );
      return;
    }
  }

  for (const item of screen.edges) {
    const dist = pointLineDistance(mx, my, item.x1, item.y1, item.x2, item.y2);
    if (dist <= Math.max(6, item.thickness + 3)) {
      const edge = item.edge;
      const delayText = edge.meanDelay === null ? "n/a" : edge.meanDelay.toFixed(2);
      showNetworkTooltip(
        `<strong>${edge.from} -> ${edge.to}</strong><br/>` +
          `Synapses: ${edge.nSynapses}<br/>` +
          `Weight: ${edge.meanWeight.toFixed(3)} +/- ${edge.stdWeight.toFixed(3)}<br/>` +
          `Delay steps: ${delayText}`,
        mx,
        my
      );
      return;
    }
  }

  hideNetworkTooltip();
}

function pointLineDistance(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (dx === 0 && dy === 0) return Math.hypot(px - x1, py - y1);
  const t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
  const clamped = Math.max(0, Math.min(1, t));
  const lx = x1 + clamped * dx;
  const ly = y1 + clamped * dy;
  return Math.hypot(px - lx, py - ly);
}

function buildWeightsIndex(rows) {
  const index = {};
  rows.forEach((row) => {
    const proj = row.proj || "projection";
    const step = Number(row.step || 0);
    const pre = Number(row.pre || 0);
    const post = Number(row.post || 0);
    const w = Number(row.w || 0);
    if (!index[proj]) {
      index[proj] = { steps: [], byStep: {} };
    }
    if (!index[proj].byStep[step]) {
      index[proj].byStep[step] = [];
      index[proj].steps.push(step);
    }
    index[proj].byStep[step].push({ pre, post, w });
  });
  Object.values(index).forEach((entry) => {
    entry.steps.sort((a, b) => a - b);
  });
  return index;
}

function refreshWeightSelectors() {
  if (!uiControls.weightProjection || !dataState.weightsIndex) return;
  const projections = Object.keys(dataState.weightsIndex);
  if (projections.length === 0) return;

  const currentProj = uiControls.weightProjection.value;
  if (!projections.includes(currentProj)) {
    uiControls.weightProjection.innerHTML = projections
      .map((proj) => `<option value="${proj}">${proj}</option>`)
      .join("");
    uiControls.weightProjection.value = projections[0];
  }

  const steps = dataState.weightsIndex[uiControls.weightProjection.value]?.steps || [];
  if (uiControls.weightStep && steps.length) {
    const currentStep = Number(uiControls.weightStep.value || steps[steps.length - 1]);
    uiControls.weightStep.innerHTML = steps
      .map((step) => `<option value="${step}">${step}</option>`)
      .join("");
    uiControls.weightStep.value = steps.includes(currentStep) ? currentStep : steps[steps.length - 1];
  }
}

function getRasterSettings() {
  const stride = Number(uiControls.rasterStride?.value || 1);
  const windowSteps = Number(uiControls.rasterWindow?.value || 200);
  const maxPoints = Number(uiControls.rasterMaxPoints?.value || 200000);
  return {
    stride: Math.max(1, stride),
    windowSteps: Math.max(10, windowSteps),
    maxPoints: Math.max(1000, maxPoints),
  };
}

function getPopulationInfo() {
  if (dataState.topology && dataState.topology.mode === "population") {
    const nodes = dataState.topology.nodes || [];
    const ordered = [...nodes].sort((a, b) => (a.layer ?? 0) - (b.layer ?? 0));
    const sizes = {};
    ordered.forEach((node) => {
      sizes[node.id || node.label] = node.n_neurons || node.n || 0;
    });
    return { order: ordered.map((node) => node.id || node.label), sizes };
  }
  return { order: ["pop0"], sizes: { pop0: 0 } };
}

function drawSpikeRasterFromEvents(rows) {
  const ctx = setupCanvas(canvases.raster);
  if (!ctx) return;
  const { width, height } = canvases.raster.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  if (!rows || rows.length === 0) {
    ctx.fillStyle = theme.muted;
    ctx.fillText("Raster unavailable", 12, 20);
    return;
  }

  const settings = getRasterSettings();
  const lastRow = rows[rows.length - 1];
  const maxStep = Number(lastRow.step || 0);
  const minStep = Math.max(0, maxStep - settings.windowSteps + 1);

  let filtered = rows.filter((row) => Number(row.step) >= minStep);
  if (settings.stride > 1) {
    filtered = filtered.filter((row) => Number(row.step) % settings.stride === 0);
  }

  if (filtered.length > settings.maxPoints) {
    const stride = Math.ceil(filtered.length / settings.maxPoints);
    filtered = filtered.filter((_, idx) => idx % stride === 0);
  }

  const popInfo = getPopulationInfo();
  const gap = 2;
  const offsets = {};
  let totalRows = 0;
  popInfo.order.forEach((pop) => {
    offsets[pop] = totalRows;
    totalRows += (popInfo.sizes[pop] || 0) + gap;
  });
  totalRows = Math.max(totalRows, 1);

  filtered.forEach((row) => {
    const pop = row.pop || "pop0";
    const offset = offsets[pop] ?? 0;
    const neuron = Number(row.neuron || 0);
    const step = Number(row.step || 0);
    const x = ((step - minStep) / settings.windowSteps) * width;
    const y = ((offset + neuron + 0.5) / totalRows) * height;
    const color = pop.toLowerCase().includes("input")
      ? theme.input
      : pop.toLowerCase().includes("hidden")
        ? theme.hidden
        : pop.toLowerCase().includes("output")
          ? theme.output
          : theme.accent;
    ctx.fillStyle = color;
    ctx.fillRect(x, y, 2, 2);
  });
}

function drawAccuracyChart() {
  const ctx = setupCanvas(canvases.accuracy);
  if (!ctx) return;
  const { width, height } = canvases.accuracy.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const rows = dataState.metricsRows;
  if (!rows || rows.length === 0) {
    ctx.fillStyle = theme.muted;
    ctx.fillText("Accuracy unavailable", 12, 20);
    return;
  }

  const smooth = Math.max(1, Number(uiControls.smoothWindow?.value || 1));
  const train = rows.map((row) => Number(row.train_accuracy || row.trainAcc || ""));
  const evalAcc = rows.map((row) => Number(row.eval_accuracy || row.evalAcc || ""));
  const fallback = rows.map((row) => Number(row.spike_fraction_total || 0));

  const trainSeries = train.some((v) => !Number.isNaN(v)) ? smoothSeries(train, smooth) : fallback;
  const evalSeries = evalAcc.some((v) => !Number.isNaN(v)) ? smoothSeries(evalAcc, smooth) : null;

  drawLineSeries(ctx, trainSeries, theme.accent, width, height);
  if (evalSeries) {
    drawLineSeries(ctx, evalSeries, theme.output, width, height);
  }

  const last = rows[rows.length - 1];
  if (metricNodes.trainAccLatest) {
    metricNodes.trainAccLatest.textContent = last.train_accuracy || "--";
  }
  if (metricNodes.evalAccLatest) {
    metricNodes.evalAccLatest.textContent = last.eval_accuracy || "--";
  }
  if (metricNodes.lossLatest) {
    metricNodes.lossLatest.textContent = last.loss || "--";
  }
}

function drawLineSeries(ctx, values, color, width, height) {
  const filtered = values.filter((v) => !Number.isNaN(v));
  if (filtered.length === 0) return;
  const minVal = Math.min(...filtered);
  const maxVal = Math.max(...filtered);
  const span = maxVal - minVal || 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((value, idx) => {
    if (Number.isNaN(value)) return;
    const x = (idx / Math.max(values.length - 1, 1)) * width;
    const y = height - ((value - minVal) / span) * height;
    if (idx === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function smoothSeries(values, window) {
  return values.map((value, idx) => {
    const start = Math.max(0, idx - window + 1);
    const slice = values.slice(start, idx + 1).filter((v) => !Number.isNaN(v));
    if (slice.length === 0) return value;
    return slice.reduce((sum, v) => sum + v, 0) / slice.length;
  });
}

function buildWeightMatrix(edges, nPre, nPost, maxDim) {
  if (!edges || edges.length === 0) {
    return { matrix: [[0]], nPre: 1, nPost: 1 };
  }
  let preDim = nPre || 0;
  let postDim = nPost || 0;
  edges.forEach((edge) => {
    preDim = Math.max(preDim, edge.pre + 1);
    postDim = Math.max(postDim, edge.post + 1);
  });
  const targetDim = Number(maxDim || 64);
  const preBins = Math.min(preDim || 1, targetDim);
  const postBins = Math.min(postDim || 1, targetDim);

  const matrix = Array.from({ length: preBins }, () => Array.from({ length: postBins }, () => 0));
  const counts = Array.from({ length: preBins }, () => Array.from({ length: postBins }, () => 0));

  edges.forEach((edge) => {
    const i = Math.floor((edge.pre / preDim) * preBins);
    const j = Math.floor((edge.post / postDim) * postBins);
    const ii = Math.min(preBins - 1, Math.max(0, i));
    const jj = Math.min(postBins - 1, Math.max(0, j));
    matrix[ii][jj] += edge.w;
    counts[ii][jj] += 1;
  });

  for (let i = 0; i < preBins; i += 1) {
    for (let j = 0; j < postBins; j += 1) {
      if (counts[i][j] > 0) {
        matrix[i][j] /= counts[i][j];
      }
    }
  }

  return { matrix, nPre: preDim, nPost: postDim };
}

function drawHeatmapMatrix(canvas, matrix, clampMin, clampMax) {
  const ctx = setupCanvas(canvas);
  if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellW = width / cols;
  const cellH = height / rows;

  const minVal = clampMin ?? -1;
  const maxVal = clampMax ?? 1;
  const span = maxVal - minVal || 1;

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const value = matrix[r][c];
      const norm = (value - minVal) / span;
      const clamped = Math.max(0, Math.min(1, norm));
      const color = value >= 0 ? theme.excit : theme.inhib;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.2 + clamped * 0.8;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }
  ctx.globalAlpha = 1;
}

function renderWeightHeatmaps() {
  if (!dataState.weightsIndex) {
    drawHeatmap(canvases.heatmapInput, 16, 18, null);
    drawHeatmap(canvases.heatmapOutput, 10, 12, null);
    return;
  }
  const projections = Object.keys(dataState.weightsIndex);
  if (projections.length === 0) return;

  const selectedProj = uiControls.weightProjection?.value || projections[0];
  const steps = dataState.weightsIndex[selectedProj]?.steps || [];
  const selectedStep = uiControls.weightStep?.value
    ? Number(uiControls.weightStep.value)
    : steps[steps.length - 1];

  const edges = dataState.weightsIndex[selectedProj]?.byStep[selectedStep] || [];
  const clampMin = uiControls.weightClampMin ? Number(uiControls.weightClampMin.value) : -1;
  const clampMax = uiControls.weightClampMax ? Number(uiControls.weightClampMax.value) : 1;
  const maxDim = uiControls.weightMaxDim ? Number(uiControls.weightMaxDim.value) : 64;

  const { matrix } = buildWeightMatrix(edges, 0, 0, maxDim);
  drawHeatmapMatrix(canvases.heatmapInput, matrix, clampMin, clampMax);

  if (projections.length > 1) {
    const secondary = projections.find((proj) => proj !== selectedProj) || projections[0];
    const sEdges = dataState.weightsIndex[secondary]?.byStep[selectedStep] || [];
    const sMatrix = buildWeightMatrix(sEdges, 0, 0, maxDim).matrix;
    drawHeatmapMatrix(canvases.heatmapOutput, sMatrix, clampMin, clampMax);
  } else {
    drawHeatmapMatrix(canvases.heatmapOutput, matrix, clampMin, clampMax);
  }
}

function getSelectedWeightsEdges() {
  if (!dataState.weightsIndex) return null;
  const projections = Object.keys(dataState.weightsIndex);
  if (projections.length === 0) return null;
  const selectedProj = uiControls.weightProjection?.value || projections[0];
  const steps = dataState.weightsIndex[selectedProj]?.steps || [];
  const selectedStep = uiControls.weightStep?.value
    ? Number(uiControls.weightStep.value)
    : steps[steps.length - 1];
  return dataState.weightsIndex[selectedProj]?.byStep[selectedStep] || null;
}

function extractWeightSamples() {
  const edges = getSelectedWeightsEdges();
  if (!edges) {
    return extractSampleValues(dataState.synapseRows, "weights_i", 120);
  }
  return edges.map((edge) => edge.w);
}

function computeSpikeRates() {
  if (!dataState.spikesRows || dataState.spikesRows.length === 0) return null;
  const lastRow = dataState.spikesRows[dataState.spikesRows.length - 1];
  const maxStep = Number(lastRow.step || 0);
  const windowSteps = Math.max(50, Number(uiControls.rasterWindow?.value || 200));
  const minStep = Math.max(0, maxStep - windowSteps + 1);
  const popInfo = getPopulationInfo();
  const counts = {};
  dataState.spikesRows.forEach((row) => {
    const step = Number(row.step || 0);
    if (step < minStep) return;
    const pop = row.pop || "pop0";
    counts[pop] = (counts[pop] || 0) + 1;
  });

  const rates = popInfo.order.map((pop) => {
    const n = popInfo.sizes[pop] || 1;
    const count = counts[pop] || 0;
    return n ? count / (windowSteps * n) : 0;
  });

  return { rates, labels: popInfo.order };
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

  renderWeightHeatmaps();
  drawAccuracyChart();

  const rates = computeSpikeRates();
  if (rates) {
    drawBars(canvases.rate, rates.rates.length, theme.danger, rates.rates);
  } else {
    const rateSamples = extractSampleValues(dataState.neuronRows, "spike_i", 32);
    drawBars(canvases.rate, 32, theme.danger, rateSamples);
  }

  const weightSamples = extractWeightSamples();
  drawHistogram(canvases.weight, weightSamples);
  drawStateSpace();

  requestAnimationFrame(tick);
}

window.addEventListener("resize", resizeAll);
if (networkControls.viewSelect) {
  networkControls.viewSelect.addEventListener("change", (event) => {
    userViewSelection = true;
    networkView = event.target.value;
    resolveNetworkView();
  });
}
if (canvases.network) {
  canvases.network.addEventListener("mousemove", onNetworkHover);
  canvases.network.addEventListener("mouseleave", hideNetworkTooltip);
}
resizeAll();
if (uiControls.weightProjection) {
  uiControls.weightProjection.addEventListener("change", () => {
    refreshWeightSelectors();
  });
}
if (uiControls.weightStep) {
  uiControls.weightStep.addEventListener("change", () => {});
}
refreshData();
setInterval(refreshData, dataConfig.refreshMs);
requestAnimationFrame(tick);
