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
  neuronMaxPerPop: document.getElementById("neuronMaxPerPop"),
  neuronMaxPerPopValue: document.getElementById("neuronMaxPerPopValue"),
  neuronViewMode: document.getElementById("neuronViewMode"),
  neuronSampleMode: document.getElementById("neuronSampleMode"),
  neuronLayoutMode: document.getElementById("neuronLayoutMode"),
  neuronSampleInfo: document.getElementById("neuronSampleInfo"),
  neuronClampBadge: document.getElementById("neuronClampBadge"),
  neuronControls: document.getElementById("neuronControls"),
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
  dataLink: document.getElementById("dataLink"),
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
    totalSteps: Number(params.get("total_steps") || params.get("steps") || 0),
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
  totalSteps: 0,
  popIndex: null,
  neuronView: {
    maxPerPop: 64,
    mode: "auto",
    sampleMode: "evenlySpaced",
    layoutMode: "layered",
    shownByPop: {},
    clampWarning: false,
  },
  neuronViewVersion: 0,
  neuronRatesCache: null,
  live: false,
  lastUpdated: null,
};

const NEURON_SHOW_ALL_CAP = 512;
const NEURON_EDGE_SAMPLE_CAP = 600;

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

function buildPopulationIndexMap(popTopology) {
  if (!popTopology || !Array.isArray(popTopology.nodes)) {
    return null;
  }
  const pops = [];
  let offset = 0;
  popTopology.nodes.forEach((node) => {
    const n = Number(node.nNeurons ?? 0);
    const name = node.id ?? node.label ?? `pop_${pops.length}`;
    const pop = {
      name,
      n,
      offsetStart: offset,
      offsetEnd: offset + n,
      layer: Number(node.layer ?? 0),
      x: clamp01(node.x ?? Math.random()),
      y: clamp01(node.y ?? Math.random()),
    };
    pops.push(pop);
    offset += n;
  });
  const byName = new Map(pops.map((pop) => [pop.name, pop]));
  const popFromGlobalIndex = (globalIdx) => {
    for (const pop of pops) {
      if (globalIdx >= pop.offsetStart && globalIdx < pop.offsetEnd) {
        return { popName: pop.name, localIdx: globalIdx - pop.offsetStart };
      }
    }
    return null;
  };
  const globalIndex = (popName, localIdx) => {
    const pop = byName.get(popName);
    if (!pop) return null;
    if (localIdx < 0 || localIdx >= pop.n) return null;
    return pop.offsetStart + localIdx;
  };
  return { pops, byName, total: offset, popFromGlobalIndex, globalIndex };
}

function sampleIndices(nTotal, maxShow, mode) {
  if (nTotal <= 0) return [];
  if (mode === "all" || nTotal <= maxShow) {
    return Array.from({ length: nTotal }, (_, idx) => idx);
  }
  const target = Math.max(1, Math.min(maxShow, nTotal));
  if (mode === "firstK") {
    return Array.from({ length: target }, (_, idx) => idx);
  }
  if (mode !== "evenlySpaced") {
    throw new Error(`Unknown sampling mode: ${mode}`);
  }
  if (target === 1) return [0];
  const last = nTotal - 1;
  const step = last / (target - 1);
  const indices = [];
  for (let i = 0; i < target; i += 1) {
    indices.push(Math.floor(i * step));
  }
  indices[indices.length - 1] = last;
  return indices;
}

function buildNeuronViewState(popIndex, current) {
  if (!popIndex) return current;
  const maxPerPop = Number(current?.maxPerPop ?? 64);
  const mode = current?.mode ?? "auto";
  const sampleMode = current?.sampleMode ?? "evenlySpaced";
  const layoutMode = current?.layoutMode ?? "layered";
  const shownByPop = {};
  let clampWarning = false;

  popIndex.pops.forEach((pop) => {
    const nTotal = pop.n;
    let indices = [];
    if (mode === "all") {
      const target = Math.min(nTotal, NEURON_SHOW_ALL_CAP);
      if (nTotal > NEURON_SHOW_ALL_CAP) {
        clampWarning = true;
      }
      indices = sampleIndices(nTotal, target, "all");
    } else if (mode === "sample") {
      indices = sampleIndices(nTotal, maxPerPop, sampleMode);
    } else {
      const useAll = nTotal <= maxPerPop;
      indices = sampleIndices(nTotal, maxPerPop, useAll ? "all" : sampleMode);
    }
    shownByPop[pop.name] = { nTotal, indices, nShown: indices.length };
  });

  return {
    maxPerPop,
    mode,
    sampleMode,
    layoutMode,
    shownByPop,
    clampWarning,
  };
}

function bumpNeuronViewVersion() {
  dataState.neuronViewVersion += 1;
  dataState.neuronRatesCache = null;
}

function getSpikeWindowSteps() {
  return Math.max(10, Number(uiControls.rasterWindow?.value || 200));
}

function getDtSeconds() {
  const last = dataState.metricsRows?.[dataState.metricsRows.length - 1];
  if (last) {
    const candidate = Number(last.dt ?? last.delta_t ?? last.timestep ?? last.time_step ?? 0);
    if (candidate > 0) return candidate;
  }
  return 1e-3;
}

function computeNeuronRatesForVisible() {
  if (!dataState.spikesRows || dataState.spikesRows.length === 0) return null;
  if (!dataState.neuronView?.shownByPop) return null;
  const windowSteps = getSpikeWindowSteps();
  const key = `${dataState.spikesRows.length}|${windowSteps}|${dataState.neuronViewVersion}`;
  if (dataState.neuronRatesCache?.key === key) {
    return dataState.neuronRatesCache.ratesByPop;
  }

  const lastRow = dataState.spikesRows[dataState.spikesRows.length - 1];
  const maxStep = Number(lastRow.step || 0);
  const minStep = Math.max(0, maxStep - windowSteps + 1);
  const shownByPop = dataState.neuronView.shownByPop;
  const shownSets = {};
  Object.keys(shownByPop).forEach((pop) => {
    shownSets[pop] = new Set(shownByPop[pop].indices || []);
  });

  const counts = new Map();
  dataState.spikesRows.forEach((row) => {
    const step = Number(row.step || 0);
    if (step < minStep) return;
    const pop = row.pop || "pop0";
    const neuron = Number(row.neuron || 0);
    const set = shownSets[pop];
    if (!set || !set.has(neuron)) return;
    const keyPop = `${pop}:${neuron}`;
    counts.set(keyPop, (counts.get(keyPop) || 0) + 1);
  });

  const dt = getDtSeconds();
  const windowSeconds = windowSteps * dt || 1;
  const ratesByPop = new Map();
  Object.entries(shownByPop).forEach(([pop, info]) => {
    const rateMap = new Map();
    (info.indices || []).forEach((idx) => {
      const count = counts.get(`${pop}:${idx}`) || 0;
      rateMap.set(idx, count / windowSeconds);
    });
    ratesByPop.set(pop, rateMap);
  });

  dataState.neuronRatesCache = { key, ratesByPop };
  return ratesByPop;
}

function formatIndicesForInfo(indices, maxItems) {
  if (!indices || indices.length === 0) {
    return { text: "", fullText: "" };
  }
  const fullText = indices.join(", ");
  if (indices.length <= 5) {
    return { text: fullText, fullText };
  }
  const head = indices.slice(0, 4).join(", ");
  const tail = indices[indices.length - 1];
  return { text: `${head}, …, ${tail}`, fullText };
}

function syncNeuronViewControls() {
  if (!uiControls.neuronMaxPerPop || !uiControls.neuronViewMode || !uiControls.neuronSampleMode) {
    return;
  }
  uiControls.neuronMaxPerPop.value = String(dataState.neuronView.maxPerPop ?? 64);
  if (uiControls.neuronMaxPerPopValue) {
    uiControls.neuronMaxPerPopValue.textContent = uiControls.neuronMaxPerPop.value;
  }
  uiControls.neuronViewMode.value = dataState.neuronView.mode ?? "auto";
  uiControls.neuronSampleMode.value = dataState.neuronView.sampleMode ?? "evenlySpaced";
  if (uiControls.neuronLayoutMode) {
    uiControls.neuronLayoutMode.value = dataState.neuronView.layoutMode ?? "layered";
  }
}

function updateNeuronControlsEnabled() {
  const enabled = Boolean(dataState.popIndex) && networkView === "neurons";
  [
    uiControls.neuronMaxPerPop,
    uiControls.neuronViewMode,
    uiControls.neuronSampleMode,
    uiControls.neuronLayoutMode,
  ].forEach((control) => {
    if (control) control.disabled = !enabled;
  });
  if (uiControls.neuronControls) {
    uiControls.neuronControls.classList.toggle("hidden", !enabled);
  }
}

function readNeuronViewSettings() {
  const maxPerPop = Math.max(8, Number(uiControls.neuronMaxPerPop?.value || 64));
  const mode = uiControls.neuronViewMode?.value || "auto";
  const sampleMode = uiControls.neuronSampleMode?.value || "evenlySpaced";
  const layoutMode = uiControls.neuronLayoutMode?.value || "layered";
  return { maxPerPop, mode, sampleMode, layoutMode };
}

function applyNeuronViewSettings() {
  if (!dataState.popIndex || !networkPopulation) {
    updateNeuronViewInfo();
    return;
  }
  const settings = readNeuronViewSettings();
  dataState.neuronView = buildNeuronViewState(dataState.popIndex, {
    ...dataState.neuronView,
    ...settings,
  });
  bumpNeuronViewVersion();
  if (Array.isArray(dataState.topology?.neuron_nodes) && Array.isArray(dataState.topology?.neuron_edges)) {
    networkNeuron = buildNeuronTopologyFromPayload(
      dataState.topology,
      dataState.popIndex,
      dataState.neuronView
    );
  } else {
    networkNeuron = buildNeuronTopologyFromPopulations(
      networkPopulation,
      dataState.popIndex,
      dataState.neuronView
    );
  }
  resolveNetworkView();
  updateNeuronViewInfo();
}

function updateNeuronViewInfo() {
  const infoEl = uiControls.neuronSampleInfo;
  if (!infoEl) return;
  if (networkView !== "neurons") {
    infoEl.textContent = "";
    infoEl.title = "";
    infoEl.dataset.copyText = "";
    infoEl.classList.add("hidden");
    if (uiControls.neuronClampBadge) {
      uiControls.neuronClampBadge.classList.add("hidden");
    }
    return;
  }
  if (!dataState.popIndex || !dataState.neuronView) {
    infoEl.textContent = "Neuron sampling unavailable (no population topology).";
    infoEl.title = "";
    infoEl.dataset.copyText = "";
    infoEl.classList.remove("hidden");
    if (uiControls.neuronClampBadge) {
      uiControls.neuronClampBadge.classList.add("hidden");
    }
    return;
  }
  const ordered = [...dataState.popIndex.pops].sort((a, b) => a.layer - b.layer);
  const lines = [];
  const titles = [];
  const copyLines = [];
  ordered.forEach((pop) => {
    const shown = dataState.neuronView.shownByPop?.[pop.name];
    if (!shown) return;
    const formatted = formatIndicesForInfo(shown.indices, 32);
    const isSampled = shown.nShown < shown.nTotal;
    const line = isSampled
      ? `${pop.name}: showing ${shown.nShown}/${shown.nTotal} (sample: ${formatted.text})`
      : `${pop.name}: showing ${shown.nShown}/${shown.nTotal}`;
    lines.push(line);
    titles.push(`${pop.name}: ${formatted.fullText}`);
    copyLines.push(
      `${pop.name}: ${shown.nShown}/${shown.nTotal} [${formatted.fullText}]`
    );
  });
  infoEl.textContent = lines.join("\n");
  infoEl.title = titles.join("\n");
  infoEl.dataset.copyText = copyLines.join("\n");
  infoEl.classList.remove("hidden");
  if (uiControls.neuronClampBadge) {
    uiControls.neuronClampBadge.classList.toggle("hidden", !dataState.neuronView.clampWarning);
  }
}

function jitterForIndex(idx, scale) {
  const seed = (idx * 9301 + 49297) % 233280;
  const unit = seed / 233280 - 0.5;
  return unit * scale;
}

function hashString(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function buildNeuronTopologyFromPopulations(popTopology, popIndex, neuronView) {
  if (!popTopology || !popIndex) {
    return null;
  }
  const nodes = [];
  const nodeIndexByGlobal = new Map();
  popIndex.pops.forEach((pop) => {
    const shown = neuronView?.shownByPop?.[pop.name];
    const indices = shown?.indices ?? [];
    const count = indices.length;
    const spread = Math.min(0.6, 0.8 / Math.max(count, 1));
    indices.forEach((localIdx, idx) => {
      const offsetY = count <= 1 ? 0 : (idx / (count - 1) - 0.5) * spread;
      const x = clamp01(pop.x + jitterForIndex(localIdx, 0.05));
      const y = clamp01(pop.y + offsetY);
      const globalIdx = pop.offsetStart + localIdx;
      nodes.push({
        index: globalIdx,
        pop: pop.name,
        localIdx,
        globalIdx,
        x,
        y,
        layer: pop.layer,
      });
      nodeIndexByGlobal.set(globalIdx, nodes.length - 1);
    });
  });
  applyNeuronLayout(nodes, popIndex, neuronView?.layoutMode);
  const edges = [];
  if (Array.isArray(popTopology.edges) && popTopology.edges.length) {
    popTopology.edges.forEach((edge) => {
      const fromPop = popIndex.byName.get(edge.from);
      const toPop = popIndex.byName.get(edge.to);
      if (!fromPop || !toPop) return;
      const shownPre = neuronView?.shownByPop?.[fromPop.name]?.indices ?? [];
      const shownPost = neuronView?.shownByPop?.[toPop.name]?.indices ?? [];
      if (!shownPre.length || !shownPost.length) return;

      const nSyn = Number(edge.nSynapses || 0);
      const p =
        nSyn > 0 && fromPop.n > 0 && toPop.n > 0
          ? Math.min(1, nSyn / (fromPop.n * toPop.n))
          : 0;
      const target = Math.max(1, Math.round(p * shownPre.length * shownPost.length));
      const sampleCount = Math.min(target, NEURON_EDGE_SAMPLE_CAP);
      if (sampleCount <= 0) return;

      const baseSeed = hashString(`${edge.from}->${edge.to}`);
      const rand = makeRng(baseSeed);
      const receptorCounts = edge.receptorCounts || {};
      const gabaCount = receptorCounts.gaba ?? receptorCounts.GABA ?? 0;
      const meanWeight = Number(edge.meanWeight || 0);
      const baseWeight =
        meanWeight !== 0
          ? meanWeight
          : gabaCount > 0
            ? -0.4
            : 0.4;

      const seen = new Set();
      let guard = 0;
      const startLen = edges.length;
      while (edges.length - startLen < sampleCount && guard < sampleCount * 4) {
        guard += 1;
        const preLocal = shownPre[Math.floor(rand() * shownPre.length)];
        const postLocal = shownPost[Math.floor(rand() * shownPost.length)];
        const globalPre = fromPop.offsetStart + preLocal;
        const globalPost = toPop.offsetStart + postLocal;
        const fromIdx = nodeIndexByGlobal.get(globalPre);
        const toIdx = nodeIndexByGlobal.get(globalPost);
        if (fromIdx === undefined || toIdx === undefined) continue;
        const key = `${fromIdx}-${toIdx}`;
        if (seen.has(key)) continue;
        seen.add(key);
        edges.push({
          from: fromIdx,
          to: toIdx,
          weight: baseWeight + (rand() - 0.5) * 0.2,
        });
      }
    });
  }
  return { nodes, edges, layerSizes: [] };
}

function buildNeuronTopologyFromPayload(neuronPayload, popIndex, neuronView) {
  if (!neuronPayload || !Array.isArray(neuronPayload.neuron_nodes) || !popIndex) {
    return null;
  }
  const nodes = [];
  const nodeIndexByGlobal = new Map();
  const allowed = new Set();
  if (neuronView?.shownByPop) {
    Object.entries(neuronView.shownByPop).forEach(([popName, info]) => {
      const pop = popIndex.byName.get(popName);
      if (!pop) return;
      (info.indices || []).forEach((localIdx) => {
        allowed.add(pop.offsetStart + localIdx);
      });
    });
  }

  neuronPayload.neuron_nodes.forEach((node) => {
    const globalIdx = Number(node.index ?? node.global_idx ?? node.id ?? 0);
    if (allowed.size > 0 && !allowed.has(globalIdx)) return;
    const popInfo = popIndex.popFromGlobalIndex(globalIdx);
    const popName = node.pop ?? popInfo?.popName ?? "pop";
    const localIdx = node.local_idx ?? node.localIdx ?? popInfo?.localIdx ?? 0;
    nodes.push({
      index: globalIdx,
      pop: popName,
      localIdx,
      globalIdx,
      x: clamp01(node.x ?? Math.random()),
      y: clamp01(node.y ?? Math.random()),
      layer: Number(node.layer ?? 0),
    });
    nodeIndexByGlobal.set(globalIdx, nodes.length - 1);
  });
  applyNeuronLayout(nodes, popIndex, neuronView?.layoutMode);

  const edges = [];
  if (Array.isArray(neuronPayload.neuron_edges)) {
    neuronPayload.neuron_edges.forEach((edge) => {
      const fromGlobal = Number(edge.from ?? edge.source ?? 0);
      const toGlobal = Number(edge.to ?? edge.target ?? 0);
      const fromIdx = nodeIndexByGlobal.get(fromGlobal);
      const toIdx = nodeIndexByGlobal.get(toGlobal);
      if (fromIdx === undefined || toIdx === undefined) return;
      edges.push({
        from: fromIdx,
        to: toIdx,
        weight: Number(edge.weight ?? 0),
        receptor: edge.receptor || "ampa",
      });
    });
  }
  return { nodes, edges, layerSizes: [] };
}

function applyNeuronLayout(nodes, popIndex, layoutMode) {
  if (!nodes || nodes.length === 0) return;
  const marginX = 0.08;
  const marginY = 0.08;
  if (layoutMode === "spatial") {
    normalizeNodePositions(nodes, marginX, marginY);
    return;
  }
  if (popIndex) {
    const groups = new Map();
    nodes.forEach((node) => {
      if (!node.pop) return;
      if (!groups.has(node.pop)) {
        groups.set(node.pop, []);
      }
      groups.get(node.pop).push(node);
    });
    if (groups.size > 0) {
      const orderedPops = [...popIndex.pops].sort((a, b) => a.layer - b.layer);
      const activePops = orderedPops.filter((pop) => groups.has(pop.name));
      const count = activePops.length || groups.size;
      const step = count > 1 ? (1 - 2 * marginX) / (count - 1) : 0;
      activePops.forEach((pop, idx) => {
        const group = groups.get(pop.name) || [];
        group.sort((a, b) => (a.localIdx ?? 0) - (b.localIdx ?? 0));
        const x = marginX + step * idx;
        const n = group.length;
        group.forEach((node, i) => {
          const y = marginY + ((i + 1) / (n + 1)) * (1 - 2 * marginY);
          node.x = clamp01(x + jitterForIndex(node.localIdx ?? i, 0.015));
          node.y = clamp01(y);
        });
      });
      return;
    }
  }
  normalizeNodePositions(nodes, marginX, marginY);
}

function normalizeNodePositions(nodes, marginX, marginY) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  nodes.forEach((node) => {
    minX = Math.min(minX, node.x ?? 0);
    maxX = Math.max(maxX, node.x ?? 0);
    minY = Math.min(minY, node.y ?? 0);
    maxY = Math.max(maxY, node.y ?? 0);
  });
  const rangeX = Math.max(maxX - minX, 1e-6);
  const rangeY = Math.max(maxY - minY, 1e-6);
  nodes.forEach((node) => {
    const nx = (node.x - minX) / rangeX;
    const ny = (node.y - minY) / rangeY;
    node.x = marginX + nx * (1 - 2 * marginX);
    node.y = marginY + ny * (1 - 2 * marginY);
  });
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

function getTotalSteps() {
  if (Number.isFinite(dataConfig.totalSteps) && dataConfig.totalSteps > 0) {
    return dataConfig.totalSteps;
  }
  const meta = dataState.topology?.meta ?? dataState.topology?.metadata ?? null;
  const candidate =
    meta?.total_steps ?? meta?.steps ?? dataState.topology?.total_steps ?? dataState.topology?.steps;
  const asNumber = Number(candidate || 0);
  return Number.isFinite(asNumber) && asNumber > 0 ? asNumber : 0;
}

function isLearningDisabled(rows) {
  if (!rows || rows.length === 0) return true;
  const last = rows[rows.length - 1];
  const train = Number(last.train_accuracy ?? last.trainAcc);
  const evalAcc = Number(last.eval_accuracy ?? last.evalAcc);
  const loss = Number(last.loss ?? last.train_loss ?? last.eval_loss);
  const allMissing =
    (Number.isNaN(train) || !Number.isFinite(train)) &&
    (Number.isNaN(evalAcc) || !Number.isFinite(evalAcc)) &&
    (Number.isNaN(loss) || !Number.isFinite(loss));
  return allMissing;
}

function updateMetrics() {
  metricNodes.fps.textContent = fpsSmooth.toFixed(0);
  metricNodes.activeEdges.textContent = network.edges.length.toString();

  const spikeRateValue = latestValue(dataState.metricsRows, "spike_fraction_total");
  metricNodes.spikeRate.textContent = spikeRateValue
    ? `${(Number(spikeRateValue) * 100).toFixed(1)}%`
    : `${(Math.random() * 60 + 20).toFixed(1)} Hz`;

  const learningDisabled = isLearningDisabled(dataState.metricsRows);
  const stepValue =
    latestValue(dataState.metricsRows, "step") ??
    latestValue(dataState.metricsRows, "time_step") ??
    latestValue(dataState.metricsRows, "timestep");
  const totalSteps = getTotalSteps();
  const stepNumber = stepValue !== null ? Number(stepValue) : null;
  const progressLabel =
    stepNumber !== null && Number.isFinite(stepNumber)
      ? totalSteps > 0
        ? `Step: ${stepNumber} / ${totalSteps} (${((stepNumber / totalSteps) * 100).toFixed(1)}%)`
        : `Step: ${stepNumber}`
      : "Step: --";

  const metrics = [
    [
      "Current Accuracy",
      learningDisabled
        ? "Learning disabled"
        : latestValue(dataState.metricsRows, "train_accuracy") || "0.0%",
    ],
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
      learningDisabled ? progressLabel : progressLabel,
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

  const screenNodes = [];
  network.nodes.forEach((node) => {
    const radius = node.layer === 1 ? 4.5 : 4;
    const fill = node.layer === 0 ? theme.input : node.layer === 1 ? theme.hidden : theme.output;
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(node.x * width, node.y * height, radius, 0, Math.PI * 2);
    ctx.fill();
    screenNodes.push({
      node,
      x: node.x * width,
      y: node.y * height,
      radius,
    });
  });

  if (networkNeuron) {
    networkNeuron._screen = { nodes: screenNodes };
  }
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
  const [neuronRes, synapseRes, spikesRes, metricsRes, weightsRes, topologyRes] = await Promise.all([
    loadCsv(dataConfig.neuronCsv),
    loadCsv(dataConfig.synapseCsv),
    loadCsv(dataConfig.spikesCsv),
    loadCsv(dataConfig.metricsCsv),
    loadCsv(dataConfig.weightsCsv),
    loadJson(dataConfig.topologyJson),
  ]);

  const neuronRows = neuronRes.data;
  const synapseRows = synapseRes.data;
  const spikesRows = spikesRes.data;
  const metricsRows = metricsRes.data;
  const weightsRows = weightsRes.data;
  const topology = topologyRes.data;

  dataState.neuronRows = neuronRows;
  dataState.synapseRows = synapseRows;
  dataState.spikesRows = spikesRows;
  dataState.metricsRows = metricsRows;
  dataState.weightsRows = weightsRows;
  dataState.weightsIndex = weightsRows ? buildWeightsIndex(weightsRows) : null;
  dataState.topology = topology;
  dataState.totalSteps = getTotalSteps();
  dataState.live = Boolean(neuronRows || synapseRows || spikesRows || metricsRows || weightsRows || topology);
  dataState.lastUpdated = new Date();

  updateDataStatus([
    neuronRes,
    synapseRes,
    spikesRes,
    metricsRes,
    weightsRes,
    topologyRes,
  ]);
  updateDataLink();

  if (topology) {
    if (topology.mode === "population") {
      networkPopulation = normalizePopulationTopology(topology);
      dataState.popIndex = buildPopulationIndexMap(networkPopulation);
      dataState.neuronView = buildNeuronViewState(dataState.popIndex, dataState.neuronView);
      bumpNeuronViewVersion();
      if (Array.isArray(topology.neuron_nodes) && Array.isArray(topology.neuron_edges)) {
        networkNeuron = buildNeuronTopologyFromPayload(
          topology,
          dataState.popIndex,
          dataState.neuronView
        );
      } else {
        networkNeuron = buildNeuronTopologyFromPopulations(
          networkPopulation,
          dataState.popIndex,
          dataState.neuronView
        );
      }
      dataState.topologyMode = "populations";
      syncNeuronViewControls();
      updateNeuronControlsEnabled();
      updateNeuronViewInfo();
    } else {
      networkNeuron = normalizeTopology(topology);
      dataState.topologyMode = "neurons";
      dataState.popIndex = null;
      updateNeuronControlsEnabled();
      updateNeuronViewInfo();
    }
    resolveNetworkView();
  }

  refreshWeightSelectors();
}

async function loadCsv(path) {
  try {
    const response = await fetch(`${path}?ts=${Date.now()}`);
    if (!response.ok) {
      return { data: null, error: `${response.status} ${response.statusText}`, url: path };
    }
    const text = await response.text();
    return { data: parseCsv(text), error: null, url: path };
  } catch (error) {
    return { data: null, error: String(error), url: path };
  }
}

async function loadJson(path) {
  try {
    const response = await fetch(`${path}?ts=${Date.now()}`);
    if (!response.ok) {
      return { data: null, error: `${response.status} ${response.statusText}`, url: path };
    }
    return { data: await response.json(), error: null, url: path };
  } catch (error) {
    return { data: null, error: String(error), url: path };
  }
}

function updateDataStatus(results) {
  if (!metricNodes.status || !metricNodes.statusDot) return;
  const failures = results.filter((res) => !res.data && res.error);
  if (failures.length > 0) {
    metricNodes.status.textContent = `Missing data (${failures.length})`;
    metricNodes.status.title = failures
      .map((res) => `${res.url}: ${res.error}`)
      .join("\n");
    metricNodes.statusDot.parentElement?.classList.remove("live");
    return;
  }

  if (dataState.live) {
    metricNodes.status.textContent = "Live data";
    metricNodes.status.title = "";
    metricNodes.statusDot.parentElement?.classList.add("live");
  } else {
    metricNodes.status.textContent = "Demo data";
    metricNodes.status.title = "";
    metricNodes.statusDot.parentElement?.classList.remove("live");
  }
}

function resolveRunFolderUrl() {
  const candidates = [
    dataConfig.topologyJson,
    dataConfig.neuronCsv,
    dataConfig.synapseCsv,
    dataConfig.spikesCsv,
    dataConfig.metricsCsv,
    dataConfig.weightsCsv,
  ];
  for (const candidate of candidates) {
    if (!candidate) continue;
    const url = new URL(candidate, window.location.href);
    const parts = url.pathname.split("/");
    parts.pop();
    const basePath = parts.join("/") + "/";
    return `${url.origin}${basePath}`;
  }
  return null;
}

function updateDataLink() {
  const link = metricNodes.dataLink;
  if (!link) return;
  const runUrl = resolveRunFolderUrl();
  if (!runUrl) {
    link.style.display = "none";
    return;
  }
  link.href = runUrl;
  link.style.display = "inline-flex";
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
  if (networkView === "neurons" && networkNeuron?._screen) {
    const rect = canvases.network.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const screen = networkNeuron._screen;
    const ratesByPop = computeNeuronRatesForVisible();

    for (const item of screen.nodes) {
      const dx = mx - item.x;
      const dy = my - item.y;
      if (Math.sqrt(dx * dx + dy * dy) <= item.radius + 3) {
        const node = item.node;
        const popName = node.pop || "pop";
        const localIdx = node.localIdx ?? node.index ?? 0;
        const rate =
          ratesByPop?.get(popName)?.get(localIdx) ??
          ratesByPop?.get(popName)?.get(Number(localIdx));
        const rateText = rate !== undefined ? `<br/>Rate: ${rate.toFixed(2)} Hz` : "";
        showNetworkTooltip(
          `<strong>${popName}[${localIdx}]</strong>${rateText}`,
          mx,
          my
        );
        return;
      }
    }
    hideNetworkTooltip();
    return;
  }
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
  if (dataState.popIndex) {
    const ordered = [...dataState.popIndex.pops].sort((a, b) => a.layer - b.layer);
    const sizes = {};
    ordered.forEach((pop) => {
      sizes[pop.name] = pop.n;
    });
    return { order: ordered.map((pop) => pop.name), sizes };
  }
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
  if (isLearningDisabled(rows)) {
    ctx.fillStyle = theme.muted;
    ctx.fillText("Learning disabled", 12, 20);
    if (metricNodes.trainAccLatest) {
      metricNodes.trainAccLatest.textContent = "--";
    }
    if (metricNodes.evalAccLatest) {
      metricNodes.evalAccLatest.textContent = "--";
    }
    if (metricNodes.lossLatest) {
      metricNodes.lossLatest.textContent = "--";
    }
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
    updateNeuronControlsEnabled();
    updateNeuronViewInfo();
  });
}
const onNeuronViewChange = () => {
  if (uiControls.neuronMaxPerPop && uiControls.neuronMaxPerPopValue) {
    uiControls.neuronMaxPerPopValue.textContent = uiControls.neuronMaxPerPop.value;
  }
  applyNeuronViewSettings();
};
if (uiControls.neuronMaxPerPop) {
  uiControls.neuronMaxPerPop.addEventListener("input", onNeuronViewChange);
  uiControls.neuronMaxPerPop.addEventListener("change", onNeuronViewChange);
}
if (uiControls.neuronViewMode) {
  uiControls.neuronViewMode.addEventListener("change", onNeuronViewChange);
}
if (uiControls.neuronSampleMode) {
  uiControls.neuronSampleMode.addEventListener("change", onNeuronViewChange);
}
if (uiControls.neuronLayoutMode) {
  uiControls.neuronLayoutMode.addEventListener("change", onNeuronViewChange);
}
if (canvases.network) {
  canvases.network.addEventListener("mousemove", onNetworkHover);
  canvases.network.addEventListener("mouseleave", hideNetworkTooltip);
}
resizeAll();
syncNeuronViewControls();
updateNeuronControlsEnabled();
updateNeuronViewInfo();
if (uiControls.neuronSampleInfo) {
  uiControls.neuronSampleInfo.addEventListener("click", async () => {
    const text = uiControls.neuronSampleInfo.dataset.copyText;
    if (!text || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // ignore clipboard errors
    }
  });
}
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
