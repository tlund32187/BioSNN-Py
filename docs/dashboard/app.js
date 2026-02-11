const canvases = {
  network: document.getElementById("networkCanvas"),
  raster: document.getElementById("rasterCanvas"),
  accuracy: document.getElementById("accuracyCanvas"),
  heatmapInput: document.getElementById("heatmapInput"),
  heatmapOutput: document.getElementById("heatmapOutput"),
  rate: document.getElementById("rateCanvas"),
  weight: document.getElementById("weightCanvas"),
  state: document.getElementById("stateCanvas"),
  modulator: document.getElementById("modulatorCanvas"),
  receptor: document.getElementById("receptorCanvas"),
  vision: document.getElementById("visionCanvas"),
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
  edgeOpacityByDistance: document.getElementById("edgeOpacityByDistance"),
  showDelayTooltip: document.getElementById("showDelayTooltip"),
  neuronSampleInfo: document.getElementById("neuronSampleInfo"),
  neuronClampBadge: document.getElementById("neuronClampBadge"),
  neuronControls: document.getElementById("neuronControls"),
  legendLocality: document.getElementById("legendLocality"),
  legendDelay: document.getElementById("legendDelay"),
  weightProjection: document.getElementById("weightProjectionSelect"),
  weightProjection2: document.getElementById("weightProjectionSelect2"),
  weightProjectionLabel2: document.getElementById("weightProjectionLabel2"),
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

const runNodes = {
  demoSelect: document.getElementById("runDemoSelect"),
  stepsInput: document.getElementById("runStepsInput"),
  deviceSelect: document.getElementById("runDeviceSelect"),
  fusedLayoutSelect: document.getElementById("runFusedLayoutSelect"),
  ringStrategySelect: document.getElementById("runRingStrategySelect"),
  learningToggle: document.getElementById("runLearningToggle"),
  monitorsToggle: document.getElementById("runMonitorsToggle"),
  modulatorToggle: document.getElementById("runModulatorToggle"),
  advancedSection: document.getElementById("runAdvancedSection"),
  logicBackendWrap: document.getElementById("runLogicBackendWrap"),
  logicBackendSelect: document.getElementById("runLogicBackendSelect"),
  explorationEnabledToggle: document.getElementById("runExplorationEnabledToggle"),
  epsilonStartInput: document.getElementById("runEpsilonStartInput"),
  epsilonEndInput: document.getElementById("runEpsilonEndInput"),
  epsilonDecayTrialsInput: document.getElementById("runEpsilonDecayTrialsInput"),
  tieBreakSelect: document.getElementById("runTieBreakSelect"),
  rewardDeliveryStepsInput: document.getElementById("runRewardDeliveryStepsInput"),
  advancedSynapseEnabledToggle: document.getElementById("runAdvancedSynapseEnabledToggle"),
  advancedSynapseConductanceToggle: document.getElementById("runAdvancedSynapseConductanceToggle"),
  advancedSynapseNmdaBlockToggle: document.getElementById("runAdvancedSynapseNmdaBlockToggle"),
  advancedSynapseStpToggle: document.getElementById("runAdvancedSynapseStpToggle"),
  receptorModeSelect: document.getElementById("runReceptorModeSelect"),
  modulatorFieldTypeSelect: document.getElementById("runModulatorFieldTypeSelect"),
  modulatorKindsSelect: document.getElementById("runModulatorKindsSelect"),
  wrapperEnabledToggle: document.getElementById("runWrapperEnabledToggle"),
  wrapperAchGainInput: document.getElementById("runWrapperAchGainInput"),
  wrapperNeGainInput: document.getElementById("runWrapperNeGainInput"),
  wrapperHtDecayInput: document.getElementById("runWrapperHtDecayInput"),
  excitabilityEnabledToggle: document.getElementById("runExcitabilityEnabledToggle"),
  excitabilityAchGainInput: document.getElementById("runExcitabilityAchGainInput"),
  excitabilityNeGainInput: document.getElementById("runExcitabilityNeGainInput"),
  excitabilityHtGainInput: document.getElementById("runExcitabilityHtGainInput"),
  homeostasisEnabledToggle: document.getElementById("runHomeostasisEnabledToggle"),
  homeostasisAlphaInput: document.getElementById("runHomeostasisAlphaInput"),
  homeostasisEtaInput: document.getElementById("runHomeostasisEtaInput"),
  homeostasisTargetInput: document.getElementById("runHomeostasisTargetInput"),
  homeostasisClampMinInput: document.getElementById("runHomeostasisClampMinInput"),
  homeostasisClampMaxInput: document.getElementById("runHomeostasisClampMaxInput"),
  homeostasisScopeSelect: document.getElementById("runHomeostasisScopeSelect"),
  pruningEnabledToggle: document.getElementById("runPruningEnabledToggle"),
  neurogenesisEnabledToggle: document.getElementById("runNeurogenesisEnabledToggle"),
  startButton: document.getElementById("runStartButton"),
  stopButton: document.getElementById("runStopButton"),
  stateText: document.getElementById("runStateText"),
  featureList: document.getElementById("runFeatureList"),
};

const taskNodes = {
  title: document.getElementById("taskTitle"),
  chip: document.getElementById("taskChip"),
  summary: document.getElementById("taskSummary"),
  table: document.getElementById("taskTruthTable"),
  takeaway: document.getElementById("taskTakeaway"),
};

const auxNodes = {
  modulatorCard: document.getElementById("modulatorCard"),
  modulatorFieldSelect: document.getElementById("modulatorFieldSelect"),
  modulatorKindSelect: document.getElementById("modulatorKindSelect"),
  receptorCard: document.getElementById("receptorCard"),
  receptorProjectionSelect: document.getElementById("receptorProjectionSelect"),
  receptorMetricSelect: document.getElementById("receptorMetricSelect"),
  visionCard: document.getElementById("visionCard"),
};

const theme = {
  background: "#0f1529",
  grid: "rgba(255,255,255,0.06)",
  text: "#e8eefc",
  muted: "#98a4be",
  input: "#60a5fa",
  hidden: "#6ee7b7",
  output: "#f97316",
  inputRelay: "#7da0d4",
  relay: "#93c5fd",
  excit: "#38bdf8",
  inhib: "#fb7185",
  accent: "#5ddcff",
  modelGlif: "#60a5fa",
  modelAdex: "#f59e0b",
  modelUnknown: "#94a3b8",
  heat: ["#13172b", "#334155", "#7c3aed", "#22d3ee", "#facc15"],
};

function inferRoleFromPopName(name) {
  const lowered = String(name || "").toLowerCase();
  if (lowered.startsWith("inputrelay") || lowered.includes("relay")) {
    return "input_relay";
  }
  if (lowered.startsWith("input")) {
    return "input";
  }
  if (lowered.startsWith("hidden")) {
    return "hidden";
  }
  if (lowered.startsWith("output")) {
    return "output";
  }
  return "unknown";
}

const dataConfig = (() => {
  const params = new URLSearchParams(window.location.search);
  const readParam = (key, fallback) => {
    const raw = params.get(key);
    if (raw === null || raw === undefined) return fallback;
    const value = String(raw).trim();
    if (!value || value.toLowerCase() === "none" || value.toLowerCase() === "disabled") {
      return null;
    }
    return value;
  };
  const runParam = readParam("run", null);
  const normalizeRunPath = (value) => {
    if (!value) return null;
    const path = String(value).trim();
    if (!path) return null;
    return path.endsWith("/") ? path.slice(0, -1) : path;
  };
  const runPath = normalizeRunPath(runParam);
  const apiParamRaw = params.get("api");
  const apiParam = apiParamRaw ? String(apiParamRaw).trim().toLowerCase() : "";
  const apiDisabled = apiParam === "0" || apiParam === "false" || apiParam === "off" || apiParam === "no";
  const apiTimeoutRaw = Number(params.get("api_timeout_ms") || 8000);
  const apiTimeoutMs = Number.isFinite(apiTimeoutRaw)
    ? Math.min(60000, Math.max(1000, apiTimeoutRaw))
    : 8000;
  const pathFromRun = (filename) => (runPath ? `${runPath}/${filename}` : null);
  const topologyJson = readParam("topology", pathFromRun("topology.json") || "data/topology.json");
  const neuronCsv = readParam("neuron", pathFromRun("neuron.csv") || "data/neuron.csv");
  const synapseCsv = readParam("synapse", pathFromRun("synapse.csv") || "data/synapse.csv");
  const spikesCsv = readParam("spikes", pathFromRun("spikes.csv") || "data/spikes.csv");
  const metricsCsv = readParam("metrics", pathFromRun("metrics.csv") || "data/metrics.csv");
  const weightsCsv = readParam("weights", pathFromRun("weights.csv") || "data/weights.csv");
  const trialsCsv = readParam("trials", pathFromRun("trials.csv"));
  const evalCsv = readParam("eval", pathFromRun("eval.csv"));
  const confusionCsv = readParam("confusion", pathFromRun("confusion.csv"));
  const modgridJson = readParam("modgrid", pathFromRun("modgrid.json"));
  const receptorsCsv = readParam("receptors", pathFromRun("receptors.csv"));
  const visionJson = readParam("vision", pathFromRun("vision.json"));
  return {
    runPath,
    neuronCsv,
    synapseCsv,
    spikesCsv,
    metricsCsv,
    weightsCsv,
    trialsCsv,
    evalCsv,
    confusionCsv,
    modgridJson,
    receptorsCsv,
    visionJson,
    topologyJson,
    runConfigJson: pathFromRun("run_config.json"),
    runFeaturesJson: pathFromRun("run_features.json"),
    runStatusJson: pathFromRun("run_status.json"),
    useApi: !apiDisabled,
    apiTimeoutMs,
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
  trialsRows: null,
  evalRows: null,
  confusionRows: null,
  modgridData: null,
  receptorsRows: null,
  visionData: null,
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
  runConfig: null,
  runFeatures: null,
  runStatus: null,
};

const runState = {
  demos: [],
  status: null,
  apiAvailable: false,
  apiBootstrapInFlight: false,
  activeRunPath: dataConfig.runPath || null,
  lastAppliedRunId: null,
};

const FALLBACK_DEMO_DEFINITIONS = [
  {
    id: "network",
    name: "Network",
    defaults: { demo_id: "network", steps: 500, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: false }, modulators: { enabled: false, kinds: [] } },
  },
  {
    id: "vision",
    name: "Vision",
    defaults: { demo_id: "vision", steps: 500, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: false }, modulators: { enabled: false, kinds: [] } },
  },
  {
    id: "pruning_sparse",
    name: "Pruning Sparse",
    defaults: { demo_id: "pruning_sparse", steps: 5000, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: false }, modulators: { enabled: false, kinds: [] } },
  },
  {
    id: "neurogenesis_sparse",
    name: "Neurogenesis Sparse",
    defaults: { demo_id: "neurogenesis_sparse", steps: 5000, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: false }, modulators: { enabled: false, kinds: [] } },
  },
  {
    id: "propagation_impulse",
    name: "Propagation Impulse",
    defaults: { demo_id: "propagation_impulse", steps: 120, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: false }, modulators: { enabled: false, kinds: [] } },
  },
  {
    id: "delay_impulse",
    name: "Delay Impulse",
    defaults: { demo_id: "delay_impulse", steps: 120, delay_steps: 3, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: false }, modulators: { enabled: false, kinds: [] } },
  },
  {
    id: "learning_gate",
    name: "Learning Gate",
    defaults: { demo_id: "learning_gate", steps: 200, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: true, rule: "three_factor_hebbian", lr: 0.1 }, modulators: { enabled: false, kinds: [] } },
  },
  {
    id: "dopamine_plasticity",
    name: "Dopamine Plasticity",
    defaults: { demo_id: "dopamine_plasticity", steps: 220, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: true, rule: "three_factor_hebbian", lr: 0.1 }, modulators: { enabled: true, kinds: ["dopamine"], pulse_step: 50, amount: 1.0 } },
  },
  {
    id: "logic_curriculum",
    name: "Logic Curriculum",
    defaults: { demo_id: "logic_curriculum", steps: 2500, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: true, rule: "rstdp", lr: 0.1 }, modulators: { enabled: false, kinds: [] }, logic_curriculum_gates: "or,and,nor,nand,xor,xnor", logic_curriculum_replay_ratio: 0.35, logic: { reward_delivery_steps: 2, reward_delivery_clamp_input: true, exploration: { enabled: true, mode: "epsilon_greedy", epsilon_start: 0.2, epsilon_end: 0.01, epsilon_decay_trials: 3000, tie_break: "random_among_max", seed: 123 } } },
  },
  {
    id: "logic_and",
    name: "Logic AND",
    defaults: { demo_id: "logic_and", steps: 5000, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: true, rule: "rstdp", lr: 0.1 }, modulators: { enabled: false, kinds: [] }, logic_gate: "and", logic_learning_mode: "rstdp", logic: { reward_delivery_steps: 2, reward_delivery_clamp_input: true, exploration: { enabled: true, mode: "epsilon_greedy", epsilon_start: 0.2, epsilon_end: 0.01, epsilon_decay_trials: 3000, tie_break: "random_among_max", seed: 123 } } },
  },
  {
    id: "logic_or",
    name: "Logic OR",
    defaults: { demo_id: "logic_or", steps: 5000, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: true, rule: "rstdp", lr: 0.1 }, modulators: { enabled: false, kinds: [] }, logic_gate: "or", logic_learning_mode: "rstdp", logic: { reward_delivery_steps: 2, reward_delivery_clamp_input: true, exploration: { enabled: true, mode: "epsilon_greedy", epsilon_start: 0.2, epsilon_end: 0.01, epsilon_decay_trials: 3000, tie_break: "random_among_max", seed: 123 } } },
  },
  {
    id: "logic_xor",
    name: "Logic XOR",
    defaults: { demo_id: "logic_xor", steps: 20000, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: true, rule: "rstdp", lr: 0.1 }, modulators: { enabled: false, kinds: [] }, logic_gate: "xor", logic_learning_mode: "rstdp", logic: { reward_delivery_steps: 2, reward_delivery_clamp_input: true, exploration: { enabled: true, mode: "epsilon_greedy", epsilon_start: 0.2, epsilon_end: 0.01, epsilon_decay_trials: 3000, tie_break: "random_among_max", seed: 123 } } },
  },
  {
    id: "logic_nand",
    name: "Logic NAND",
    defaults: { demo_id: "logic_nand", steps: 5000, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: true, rule: "rstdp", lr: 0.1 }, modulators: { enabled: false, kinds: [] }, logic_gate: "nand", logic_learning_mode: "rstdp", logic: { reward_delivery_steps: 2, reward_delivery_clamp_input: true, exploration: { enabled: true, mode: "epsilon_greedy", epsilon_start: 0.2, epsilon_end: 0.01, epsilon_decay_trials: 3000, tie_break: "random_among_max", seed: 123 } } },
  },
  {
    id: "logic_nor",
    name: "Logic NOR",
    defaults: { demo_id: "logic_nor", steps: 5000, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: true, rule: "rstdp", lr: 0.1 }, modulators: { enabled: false, kinds: [] }, logic_gate: "nor", logic_learning_mode: "rstdp", logic: { reward_delivery_steps: 2, reward_delivery_clamp_input: true, exploration: { enabled: true, mode: "epsilon_greedy", epsilon_start: 0.2, epsilon_end: 0.01, epsilon_decay_trials: 3000, tie_break: "random_among_max", seed: 123 } } },
  },
  {
    id: "logic_xnor",
    name: "Logic XNOR",
    defaults: { demo_id: "logic_xnor", steps: 5000, device: "cpu", fused_layout: "auto", ring_strategy: "dense", learning: { enabled: true, rule: "rstdp", lr: 0.1 }, modulators: { enabled: false, kinds: [] }, logic_gate: "xnor", logic_learning_mode: "rstdp", logic: { reward_delivery_steps: 2, reward_delivery_clamp_input: true, exploration: { enabled: true, mode: "epsilon_greedy", epsilon_start: 0.2, epsilon_end: 0.01, epsilon_decay_trials: 3000, tie_break: "random_among_max", seed: 123 } } },
  },
];

function deepClone(value) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return value;
  }
}

function fallbackDemos() {
  return FALLBACK_DEMO_DEFINITIONS.map((demo) => deepClone(demo));
}

function normalizeDemoDefinitions(rawDemos) {
  if (!Array.isArray(rawDemos)) return [];
  const fallbackById = new Map(FALLBACK_DEMO_DEFINITIONS.map((demo) => [String(demo.id), demo]));
  return rawDemos
    .map((demo) => {
      const id = String(demo?.id || "").trim();
      if (!id) return null;
      const fallback = fallbackById.get(id);
      const fallbackDefaults = fallback?.defaults || { demo_id: id };
      const incomingDefaults = demo?.defaults && typeof demo.defaults === "object" ? demo.defaults : {};
      return {
        id,
        name: String(demo?.name || fallback?.name || id),
        defaults: { ...deepClone(fallbackDefaults), ...deepClone(incomingDefaults), demo_id: id },
      };
    })
    .filter(Boolean);
}

function renderDemoSelectOptions(demos) {
  if (!runNodes.demoSelect) return;
  const selected = runNodes.demoSelect.value;
  while (runNodes.demoSelect.firstChild) {
    runNodes.demoSelect.removeChild(runNodes.demoSelect.firstChild);
  }
  demos.forEach((demo) => {
    const option = document.createElement("option");
    option.value = String(demo.id);
    option.textContent = String(demo.name || demo.id);
    runNodes.demoSelect.appendChild(option);
  });
  if (selected && demos.some((demo) => String(demo.id) === String(selected))) {
    runNodes.demoSelect.value = selected;
    return;
  }
  if (demos.length > 0) {
    runNodes.demoSelect.value = String(demos[0].id);
    applyRunSpecToControls(demos[0].defaults || {});
  }
}

const LOGIC_GATE_DEMO_TO_GATE = {
  logic_and: "and",
  logic_or: "or",
  logic_xor: "xor",
  logic_nand: "nand",
  logic_nor: "nor",
  logic_xnor: "xnor",
};

const LOGIC_GATES = new Set(["and", "or", "xor", "nand", "nor", "xnor"]);
const LOGIC_DEMO_IDS = new Set([
  "logic_curriculum",
  "logic_and",
  "logic_or",
  "logic_xor",
  "logic_nand",
  "logic_nor",
  "logic_xnor",
]);
const LOGIC_TRUTH_ROWS = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

function isLogicDemoId(demoId) {
  return LOGIC_DEMO_IDS.has(String(demoId || "").trim().toLowerCase());
}

function parseNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function parseInteger(value) {
  const n = parseNumber(value);
  return n === null ? null : Math.trunc(n);
}

function parseNumberOr(value, fallback) {
  const n = parseNumber(value);
  return n === null ? fallback : n;
}

function setMultiSelectValues(select, values) {
  if (!select) return;
  const selected = new Set(
    (Array.isArray(values) ? values : [])
      .map((item) => String(item || "").trim().toLowerCase())
      .filter(Boolean)
  );
  Array.from(select.options).forEach((option) => {
    option.selected = selected.has(String(option.value || "").trim().toLowerCase());
  });
}

function getMultiSelectValues(select) {
  if (!select) return [];
  return Array.from(select.selectedOptions)
    .map((option) => String(option.value || "").trim())
    .filter(Boolean);
}

function normalizeGateToken(value) {
  const token = String(value || "")
    .trim()
    .toLowerCase();
  return LOGIC_GATES.has(token) ? token : null;
}

function findLastRow(rows, predicate = null) {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  for (let idx = rows.length - 1; idx >= 0; idx -= 1) {
    const row = rows[idx];
    if (!predicate || predicate(row)) return row;
  }
  return null;
}

function resolveDemoId() {
  const fromConfig = String(dataState.runConfig?.demo_id || "").trim();
  if (fromConfig) return fromConfig;
  const fromSelect = String(runNodes.demoSelect?.value || "").trim();
  if (fromSelect) return fromSelect;
  return null;
}

function resolveLogicGateName() {
  const preferred = [
    latestValue(dataState.evalRows, "gate"),
    latestValue(dataState.trialsRows, "train_gate"),
    latestValue(dataState.trialsRows, "gate"),
    latestValue(dataState.metricsRows, "gate"),
    dataState.runFeatures?.logic_gate,
    dataState.runConfig?.logic_gate,
  ];
  for (const candidate of preferred) {
    const gate = normalizeGateToken(candidate);
    if (gate) return gate;
  }
  const demoId = resolveDemoId();
  if (demoId && LOGIC_GATE_DEMO_TO_GATE[demoId]) {
    return LOGIC_GATE_DEMO_TO_GATE[demoId];
  }
  return null;
}

function isLogicCurriculumRun() {
  const demoId = resolveDemoId();
  if (demoId === "logic_curriculum") return true;
  const configured = String(dataState.runConfig?.logic_curriculum_gates || "").trim();
  return configured.length > 0;
}

function resolveDisplayedLogicGate() {
  if (!isLogicCurriculumRun()) {
    return resolveLogicGateName();
  }
  const lastEvalRow = findLastRow(dataState.evalRows, (row) => normalizeGateToken(row?.gate));
  const evalGate = normalizeGateToken(lastEvalRow?.gate);
  if (evalGate) return evalGate;
  const lastTrialGate = findLastRow(
    dataState.trialsRows,
    (row) => normalizeGateToken(row?.train_gate || row?.gate)
  );
  return (
    normalizeGateToken(lastTrialGate?.train_gate) ||
    normalizeGateToken(lastTrialGate?.gate) ||
    resolveLogicGateName()
  );
}

function gateTargetBit(gate, x0, x1) {
  const a = Number(x0) > 0 ? 1 : 0;
  const b = Number(x1) > 0 ? 1 : 0;
  switch (gate) {
    case "and":
      return a & b;
    case "or":
      return a | b;
    case "xor":
      return a ^ b;
    case "nand":
      return 1 - (a & b);
    case "nor":
      return 1 - (a | b);
    case "xnor":
      return 1 - (a ^ b);
    default:
      return null;
  }
}

function logicPassCriterionText(gate) {
  if (gate === "xor") {
    return "Pass criterion: 1.0 for 500 evals or >=0.99 for 2000 evals.";
  }
  if (gate === "and" || gate === "or") {
    return "Pass criterion: 1.0 for 200 evals or >=0.99 for 500 evals.";
  }
  return "Use eval accuracy and confusion to confirm stable truth-table behavior.";
}

function renderTaskPanel() {
  if (!taskNodes.summary || !taskNodes.table || !taskNodes.takeaway || !taskNodes.title || !taskNodes.chip) {
    return;
  }
  const curriculum = isLogicCurriculumRun();
  const gate = resolveDisplayedLogicGate();
  if (!gate) {
    taskNodes.title.textContent = "Demo Task";
    taskNodes.chip.textContent = "Run summary";
    taskNodes.summary.textContent = "This run does not expose a logic-gate truth table.";
    taskNodes.table.innerHTML = '<p class="task-empty">Truth-table rows appear for logic demos.</p>';
    const train = latestValue(dataState.metricsRows, "train_accuracy");
    const evalAcc = latestValue(dataState.metricsRows, "eval_accuracy");
    taskNodes.takeaway.textContent = `Train=${train ?? "--"} | Eval=${evalAcc ?? "--"}`;
    return;
  }

  const latestEvalRow = findLastRow(
    dataState.evalRows,
    (row) => normalizeGateToken(row?.gate) === gate
  );
  const latestTrialRow = findLastRow(
    dataState.trialsRows,
    (row) => normalizeGateToken(row?.train_gate || row?.gate) === gate
  );
  const currentPhase = parseInteger(latestEvalRow?.phase) ?? parseInteger(latestTrialRow?.phase);
  const lastTrainGate =
    normalizeGateToken(latestTrialRow?.train_gate) || normalizeGateToken(latestTrialRow?.gate);

  taskNodes.title.textContent = `Logic Gate: ${gate.toUpperCase()}`;
  taskNodes.chip.textContent = curriculum ? "Curriculum phase" : "Truth table";
  if (curriculum && currentPhase !== null) {
    taskNodes.chip.textContent = `Phase ${currentPhase}`;
  }

  const latestByCase = new Map();
  const historyByCase = new Map();
  const scopedTrials = (dataState.trialsRows || []).filter((row) => {
    if (!curriculum) return true;
    const rowGate = normalizeGateToken(row?.train_gate || row?.gate);
    return rowGate === gate;
  });
  const scopedEvalRows = (dataState.evalRows || []).filter((row) => {
    if (!curriculum) return true;
    return normalizeGateToken(row?.gate) === gate;
  });
  const latestEvalPredRow = findLastRow(scopedEvalRows);
  const predKeyByCase = ["pred_00", "pred_01", "pred_10", "pred_11"];
  scopedTrials.forEach((row) => {
    let caseIdx = parseInteger(row.case_idx);
    if (caseIdx === null) {
      const x0 = parseInteger(row.x0);
      const x1 = parseInteger(row.x1);
      if (x0 !== null && x1 !== null) {
        caseIdx = x0 * 2 + x1;
      }
    }
    if (caseIdx === null || caseIdx < 0 || caseIdx > 3) return;
    latestByCase.set(caseIdx, row);
    if (!historyByCase.has(caseIdx)) {
      historyByCase.set(caseIdx, []);
    }
    historyByCase.get(caseIdx).push(row);
  });

  const bodyRows = LOGIC_TRUTH_ROWS.map(([x0, x1], idx) => {
    const row = latestByCase.get(idx) || null;
    const target = gateTargetBit(gate, x0, x1);
    const predTrial = parseInteger(row?.pred);
    const predEval = parseInteger(latestEvalPredRow?.[predKeyByCase[idx]]);
    const pred = predTrial ?? predEval;
    const out0 = parseNumber(row?.out_spikes_0);
    const out1 = parseNumber(row?.out_spikes_1);
    const total = (out0 ?? 0) + (out1 ?? 0);
    const instantConfidence =
      total > 0
        ? Math.min(1, Math.abs((out1 ?? 0) - (out0 ?? 0)) / total)
        : predTrial !== null
          ? 0
          : null;
    const history = historyByCase.get(idx) || [];
    const recentHistory = history.slice(-24);
    const recentCorrect = recentHistory
      .map((item) => parseInteger(item.correct))
      .filter((item) => item === 0 || item === 1);
    const empiricalConfidence =
      recentCorrect.length > 0
        ? recentCorrect.reduce((acc, value) => acc + Number(value), 0) / recentCorrect.length
        : null;
    const confidence = empiricalConfidence ?? instantConfidence;
    const correct = pred !== null && target !== null ? pred === target : null;
    return {
      x0,
      x1,
      target,
      pred,
      confidence,
      empiricalConfidence,
      correct,
    };
  });
  const latestEval =
    parseNumber(latestValue(scopedEvalRows, "eval_accuracy")) ??
    parseNumber(latestValue(dataState.metricsRows, "eval_accuracy"));
  const latestGlobalEval = curriculum
    ? parseNumber(latestValue(dataState.evalRows, "global_eval_accuracy")) ??
      parseNumber(latestValue(dataState.metricsRows, "global_eval_accuracy")) ??
      parseNumber(latestValue(dataState.metricsRows, "sample_accuracy_global"))
    : null;
  const latestTrain =
    parseNumber(latestValue(scopedEvalRows, curriculum ? "sample_accuracy_phase_gate" : "sample_accuracy")) ??
    parseNumber(latestValue(dataState.metricsRows, "train_accuracy")) ??
    parseNumber(latestValue(scopedTrials, "trial_acc_rolling"));
  const latestLoss = parseNumber(latestValue(dataState.metricsRows, "loss"));
  const passedFlag =
    parseInteger(latestValue(scopedEvalRows, "passed")) === 1 ||
    parseInteger(latestValue(dataState.metricsRows, "passed")) === 1;
  const solved = passedFlag || (latestEval !== null && latestEval >= 0.99);

  const coveredCases = bodyRows.filter((row) => row.pred !== null).length;
  const curriculumSuffix =
    curriculum
      ? ` | Phase gate ${gate.toUpperCase()}` +
        (lastTrainGate ? ` | Last train gate ${lastTrainGate.toUpperCase()}` : "")
      : "";
  if (curriculum) {
    taskNodes.summary.textContent =
      `Gate Eval ${latestEval !== null ? latestEval.toFixed(3) : "--"} | ` +
      `Global Eval ${latestGlobalEval !== null ? latestGlobalEval.toFixed(3) : "--"} | ` +
      `Gate Train ${latestTrain !== null ? latestTrain.toFixed(3) : "--"} | ` +
      `Loss ${latestLoss !== null ? latestLoss.toFixed(3) : "--"} | ` +
      `${solved ? "SOLVED" : "IN PROGRESS"} (${coveredCases}/4 cases observed)` +
      curriculumSuffix;
  } else {
    taskNodes.summary.textContent =
      `Eval ${latestEval !== null ? latestEval.toFixed(3) : "--"} | ` +
      `Train ${latestTrain !== null ? latestTrain.toFixed(3) : "--"} | ` +
      `Loss ${latestLoss !== null ? latestLoss.toFixed(3) : "--"} | ` +
      `${solved ? "SOLVED" : "IN PROGRESS"} (${coveredCases}/4 cases observed)` +
      curriculumSuffix;
  }

  taskNodes.table.innerHTML = `
    <table class="task-truth-table">
      <thead>
        <tr>
          <th>Input</th>
          <th>Target</th>
          <th>Pred</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        ${bodyRows
          .map((row) => {
            const predClass =
              row.pred === null
                ? "task-pred-missing"
                : row.correct
                  ? "task-pred-ok"
                  : "task-pred-bad";
            const predText = row.pred === null ? "--" : String(row.pred);
            const confText = row.confidence === null ? "--" : `${(row.confidence * 100).toFixed(1)}%`;
            return `
              <tr>
                <td>(${row.x0}, ${row.x1})</td>
                <td>${row.target ?? "--"}</td>
                <td class="${predClass}">${predText}</td>
                <td class="task-conf">${confText}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;

  taskNodes.takeaway.textContent =
    "Confidence is recent per-case correctness (fallback: WTA margin). " +
    "Train is sampled trial accuracy; Eval is full 4-case truth-table accuracy." +
    (curriculum
      ? " Gate Eval is the current phase gate; Global Eval is the average across all curriculum gates."
      : " ") +
    logicPassCriterionText(gate);
}

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
    const role = node.role ?? inferRoleFromPopName(name);
    const group = node.group ?? null;
    const pop = {
      name,
      n,
      offsetStart: offset,
      offsetEnd: offset + n,
      layer: Number(node.layer ?? 0),
      x: clamp01(node.x ?? Math.random()),
      y: clamp01(node.y ?? Math.random()),
      role,
      group,
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
    uiControls.edgeOpacityByDistance,
    uiControls.showDelayTooltip,
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

function readEdgeViewSettings() {
  return {
    edgeOpacityByDistance: Boolean(uiControls.edgeOpacityByDistance?.checked),
    showDelayTooltip: Boolean(uiControls.showDelayTooltip?.checked),
  };
}

function updateLegendNotes() {
  const settings = readEdgeViewSettings();
  if (uiControls.legendLocality) {
    uiControls.legendLocality.classList.toggle("hidden", !settings.edgeOpacityByDistance);
  }
  if (uiControls.legendDelay) {
    uiControls.legendDelay.classList.toggle("hidden", !settings.showDelayTooltip);
  }
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
        pos: null,
        layer: pop.layer,
        role: pop.role || "unknown",
        group: pop.group ?? null,
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
    const pop = popIndex.byName.get(popName);
    const pos = parseNodePos(node.pos);
    nodes.push({
      index: globalIdx,
      pop: popName,
      localIdx,
      globalIdx,
      x: clamp01(node.x ?? Math.random()),
      y: clamp01(node.y ?? Math.random()),
      pos,
      layer: Number(node.layer ?? pop?.layer ?? 0),
      role: pop?.role ?? inferRoleFromPopName(popName),
      group: pop?.group ?? null,
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
      const delaySteps =
        edge.delay_steps !== undefined
          ? Number(edge.delay_steps)
          : edge.delaySteps !== undefined
            ? Number(edge.delaySteps)
            : null;
      edges.push({
        from: fromIdx,
        to: toIdx,
        weight: Number(edge.weight ?? 0),
        receptor: edge.receptor || "ampa",
        delaySteps: Number.isFinite(delaySteps) ? delaySteps : null,
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
    normalizeNodePositions(nodes, marginX, marginY, true);
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
      const byLayer = new Map();
      activePops.forEach((pop) => {
        if (!byLayer.has(pop.layer)) {
          byLayer.set(pop.layer, []);
        }
        byLayer.get(pop.layer).push(pop);
      });
      const layers = [...byLayer.keys()].sort((a, b) => a - b);
      const layerCount = layers.length || 1;
      const layerStep = layerCount > 1 ? (1 - 2 * marginX) / (layerCount - 1) : 0;
      layers.forEach((layerVal, layerIdx) => {
        const popsInLayer = byLayer.get(layerVal) || [];
        popsInLayer.sort((a, b) => {
          const ga = a.group ?? a.name;
          const gb = b.group ?? b.name;
          if (ga < gb) return -1;
          if (ga > gb) return 1;
          return a.name.localeCompare(b.name);
        });
        const band = Math.min(0.10, layerStep * 0.65);
        const k = popsInLayer.length || 1;
        popsInLayer.forEach((pop, idx) => {
          const offset = k === 1 ? 0 : -band / 2 + (idx / (k - 1)) * band;
          const x = marginX + layerStep * layerIdx + offset;
          const group = groups.get(pop.name) || [];
          group.sort((a, b) => (a.localIdx ?? 0) - (b.localIdx ?? 0));
          const n = group.length;
          group.forEach((node, i) => {
            const y = marginY + ((i + 1) / (n + 1)) * (1 - 2 * marginY);
            node.x = clamp01(x + jitterForIndex(node.localIdx ?? i, 0.015));
            node.y = clamp01(y);
          });
        });
      });
      return;
    }
  }
  normalizeNodePositions(nodes, marginX, marginY, false);
}

function normalizeNodePositions(nodes, marginX, marginY, usePos) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  nodes.forEach((node) => {
    const pos = usePos ? parseNodePos(node.pos) : null;
    const rawX = pos ? pos[0] : node.x ?? 0;
    const rawY = pos ? pos[1] : node.y ?? 0;
    minX = Math.min(minX, rawX);
    maxX = Math.max(maxX, rawX);
    minY = Math.min(minY, rawY);
    maxY = Math.max(maxY, rawY);
  });
  const rangeX = Math.max(maxX - minX, 1e-6);
  const rangeY = Math.max(maxY - minY, 1e-6);
  nodes.forEach((node) => {
    const pos = usePos ? parseNodePos(node.pos) : null;
    const rawX = pos ? pos[0] : node.x ?? 0;
    const rawY = pos ? pos[1] : node.y ?? 0;
    const nx = (rawX - minX) / rangeX;
    const ny = (rawY - minY) / rangeY;
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
    role: node.role ?? node.meta?.role ?? inferRoleFromPopName(node.id ?? node.label ?? ""),
    group: node.group ?? node.meta?.group ?? null,
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

function parseNodePos(raw) {
  if (!raw) return null;
  if (Array.isArray(raw)) {
    if (raw.length >= 2) {
      return [
        Number(raw[0]) || 0,
        Number(raw[1]) || 0,
        Number(raw[2] ?? 0) || 0,
      ];
    }
    return null;
  }
  if (typeof raw === "object") {
    if ("x" in raw && "y" in raw) {
      return [
        Number(raw.x) || 0,
        Number(raw.y) || 0,
        Number(raw.z ?? 0) || 0,
      ];
    }
  }
  return null;
}

function buildPosByNodeId(nodes) {
  if (!nodes) return null;
  const posById = new Array(nodes.length).fill(null);
  nodes.forEach((node, idx) => {
    const pos = parseNodePos(node.pos);
    if (pos) {
      posById[idx] = pos;
    }
  });
  return posById;
}

function distanceForEdge(posById, edge) {
  if (!posById) return null;
  const a = posById[edge.from];
  const b = posById[edge.to];
  if (!a || !b) return null;
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function buildRaster(rows, cols) {
  const data = Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
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
    dataState.runConfig?.steps ??
    meta?.total_steps ??
    meta?.steps ??
    dataState.topology?.total_steps ??
    dataState.topology?.steps;
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

function getConfiguredLearningEnabled() {
  const featureEnabled = dataState.runFeatures?.learning?.enabled;
  if (typeof featureEnabled === "boolean") {
    return featureEnabled;
  }
  const configEnabled = dataState.runConfig?.learning?.enabled;
  if (typeof configEnabled === "boolean") {
    return configEnabled;
  }
  const logicMode = String(
    dataState.runConfig?.logic_learning_mode ?? dataState.runConfig?.learning_mode ?? ""
  )
    .trim()
    .toLowerCase();
  if (logicMode) {
    return logicMode !== "none";
  }
  return null;
}

function formatMaybeNumber(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "--";
  const asNumber = Number(value);
  if (!Number.isFinite(asNumber)) return String(value);
  return asNumber.toFixed(digits);
}

function getProgressLabel() {
  const totalSteps = getTotalSteps();
  const metricsStep =
    latestValue(dataState.metricsRows, "step") ??
    latestValue(dataState.metricsRows, "time_step") ??
    latestValue(dataState.metricsRows, "timestep");
  const simStepEnd = latestValue(dataState.trialsRows, "sim_step_end");
  const phase = latestValue(dataState.evalRows, "phase") ?? latestValue(dataState.trialsRows, "phase");
  const phaseTrial =
    latestValue(dataState.evalRows, "phase_trial") ?? latestValue(dataState.trialsRows, "phase_trial");
  const globalTrial = latestValue(dataState.evalRows, "trial") ?? latestValue(dataState.trialsRows, "trial");
  const isCurriculum = resolveDemoId() === "logic_curriculum";

  const phaseTrialNumber = phaseTrial !== null ? Number(phaseTrial) : null;
  if (
    isCurriculum &&
    phaseTrialNumber !== null &&
    Number.isFinite(phaseTrialNumber) &&
    totalSteps > 0
  ) {
    const pct = Math.min(100, Math.max(0, (phaseTrialNumber / totalSteps) * 100));
    const phaseText =
      phase !== null && Number.isFinite(Number(phase)) ? `Phase ${Number(phase)} ` : "";
    const globalText =
      globalTrial !== null && Number.isFinite(Number(globalTrial))
        ? ` | Global trial ${Number(globalTrial)}`
        : "";
    const simText =
      simStepEnd !== null && Number.isFinite(Number(simStepEnd))
        ? ` | Sim step ${Number(simStepEnd)}`
        : "";
    return `${phaseText}trial ${phaseTrialNumber} / ${totalSteps} (${pct.toFixed(1)}%)${globalText}${simText}`;
  }

  const fallbackStep =
    metricsStep !== null
      ? Number(metricsStep)
      : simStepEnd !== null
        ? Number(simStepEnd)
        : null;
  if (fallbackStep === null || !Number.isFinite(fallbackStep)) {
    return "Step: --";
  }
  if (totalSteps > 0) {
    return `Step: ${fallbackStep} / ${totalSteps} (${((fallbackStep / totalSteps) * 100).toFixed(1)}%)`;
  }
  return `Step: ${fallbackStep}`;
}

function updateMetrics() {
  metricNodes.fps.textContent = fpsSmooth.toFixed(0);
  metricNodes.activeEdges.textContent = network.edges.length.toString();

  const spikeRateValue = latestValue(dataState.metricsRows, "spike_fraction_total");
  metricNodes.spikeRate.textContent =
    spikeRateValue !== null ? `${(Number(spikeRateValue) * 100).toFixed(1)}%` : "--";

  const configuredLearningEnabled = getConfiguredLearningEnabled();
  const learningDisabled =
    configuredLearningEnabled === null
      ? isLearningDisabled(dataState.metricsRows) && isLearningDisabled(dataState.evalRows)
      : !configuredLearningEnabled;
  const progressLabel = getProgressLabel();
  const currentAccuracy =
    latestValue(dataState.metricsRows, "train_accuracy") ??
    latestValue(dataState.evalRows, "sample_accuracy_phase_gate") ??
    latestValue(dataState.evalRows, "sample_accuracy") ??
    latestValue(dataState.evalRows, "eval_accuracy");

  const weightMean =
    latestValue(dataState.synapseRows, "weights_mean") ??
    latestValue(dataState.trialsRows, "weights_mean");
  const avgSpike =
    latestValue(dataState.metricsRows, "spike_fraction_total") ??
    latestValue(dataState.trialsRows, "hidden_mean_spikes");
  const activeNeurons =
    latestValue(dataState.neuronRows, "active_neurons") ??
    latestValue(dataState.trialsRows, "hidden_mean_spikes");

  const metrics = [
    [
      "Current Accuracy",
      learningDisabled
        ? "Learning disabled"
        : formatMaybeNumber(currentAccuracy),
    ],
    [
      "Weight Mean",
      weightMean !== null ? `${Number(weightMean).toFixed(3)}` : "--",
    ],
    [
      "Avg Spike Rate",
      avgSpike !== null ? `${(Number(avgSpike) * 100).toFixed(1)}%` : "--",
    ],
    [
      "Active Neurons",
      activeNeurons !== null ? String(activeNeurons) : "--",
    ],
    [
      "Progress",
      progressLabel,
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

  const edgeSettings = readEdgeViewSettings();
  const useEdgeDistance =
    edgeSettings.edgeOpacityByDistance || edgeSettings.showDelayTooltip;
  const posById = useEdgeDistance ? buildPosByNodeId(network.nodes) : null;
  let edgeDistances = null;
  let dMin = Infinity;
  let dMax = -Infinity;
  if (useEdgeDistance) {
    edgeDistances = new Array(network.edges.length).fill(null);
    network.edges.forEach((edge, idx) => {
      const dist = distanceForEdge(posById, edge);
      edgeDistances[idx] = dist;
      if (edgeSettings.edgeOpacityByDistance && dist !== null) {
        dMin = Math.min(dMin, dist);
        dMax = Math.max(dMax, dist);
      }
    });
    if (!Number.isFinite(dMin) || !Number.isFinite(dMax)) {
      dMin = 0;
      dMax = 0;
    }
  }

  const screenEdges = [];
  ctx.lineWidth = 1;
  network.edges.forEach((edge, idx) => {
    const from = network.nodes[edge.from];
    const to = network.nodes[edge.to];
    if (!from || !to) return;
    const weight = edge.weight || 0;
    const color = weight >= 0 ? theme.excit : theme.inhib;
    const distance = edgeDistances ? edgeDistances[idx] : null;
    let alpha = 0.18 + Math.min(0.45, Math.sqrt(Math.abs(weight)));
    if (edgeSettings.edgeOpacityByDistance && distance !== null && dMax > dMin) {
      const t = (distance - dMin) / (dMax - dMin + 1e-9);
      const localityAlpha = 0.35 + 0.65 * Math.pow(1 - clamp01(t), 2);
      alpha *= localityAlpha;
    }
    alpha = Math.max(0.12, Math.min(1.0, alpha));
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    const x1 = from.x * width;
    const y1 = from.y * height;
    const x2 = to.x * width;
    const y2 = to.y * height;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    screenEdges.push({
      edge,
      from,
      to,
      x1,
      y1,
      x2,
      y2,
      thickness: 1,
      distance,
    });
  });
  ctx.globalAlpha = 1;

  const screenNodes = [];
  network.nodes.forEach((node) => {
    const role = node.role || "unknown";
    const radius = role === "hidden" ? 4.5 : 4;
    let fill = theme.output;
    if (role === "input") {
      fill = theme.input;
    } else if (role === "input_relay") {
      fill = theme.relay;
    } else if (role === "hidden") {
      fill = theme.hidden;
    } else if (role === "output") {
      fill = theme.output;
    } else if (node.layer === 0) {
      fill = theme.input;
    } else if (node.layer === 1) {
      fill = theme.hidden;
    }
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
    networkNeuron._screen = { nodes: screenNodes, edges: screenEdges };
  }
  drawNeuronLegend(ctx, width, height);
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
    const role = String(node.role || "").toLowerCase();
    if (role === "input") {
      fill = theme.input;
    } else if (role === "input_relay") {
      fill = theme.inputRelay;
    } else if (role === "hidden") {
      fill = theme.hidden;
    } else if (role === "output") {
      fill = theme.output;
    } else if (node.model.includes("glif")) {
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

function drawNeuronLegend(ctx, width, height) {
  const x = 12;
  const y = 12;
  const boxW = 160;
  const boxH = 96;
  ctx.save();
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = "rgba(12, 18, 34, 0.85)";
  ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
  ctx.lineWidth = 1;
  _roundedRect(ctx, x, y, boxW, boxH, 10);
  ctx.fill();
  ctx.stroke();

  const items = [
    { label: "Input", color: theme.input },
    { label: "Input relay", color: theme.relay },
    { label: "Hidden", color: theme.hidden },
    { label: "Output", color: theme.output },
  ];
  ctx.font = "10px 'Space Grotesk'";
  ctx.fillStyle = theme.text;
  let offsetY = y + 18;
  items.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(x + 12, offsetY - 4, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = theme.text;
    ctx.fillText(item.label, x + 24, offsetY);
    offsetY += 14;
  });

  ctx.strokeStyle = theme.excit;
  ctx.beginPath();
  ctx.moveTo(x + 12, y + boxH - 18);
  ctx.lineTo(x + 40, y + boxH - 18);
  ctx.stroke();
  ctx.fillStyle = theme.text;
  ctx.fillText("Excitatory", x + 46, y + boxH - 14);

  ctx.strokeStyle = theme.inhib;
  ctx.beginPath();
  ctx.moveTo(x + 12, y + boxH - 6);
  ctx.lineTo(x + 40, y + boxH - 6);
  ctx.stroke();
  ctx.fillStyle = theme.text;
  ctx.fillText("Inhibitory", x + 46, y + boxH - 2);
  ctx.restore();
}

function _roundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawRaster() {
  if (dataState.spikesRows && dataState.spikesRows.length) {
    drawSpikeRasterFromEvents(dataState.spikesRows);
    return;
  }
  if (!dataState.neuronRows || dataState.neuronRows.length === 0) {
    const ctxUnavailable = setupCanvas(canvases.raster);
    if (!ctxUnavailable) return;
    const rect = canvases.raster.getBoundingClientRect();
    ctxUnavailable.fillStyle = theme.background;
    ctxUnavailable.fillRect(0, 0, rect.width, rect.height);
    ctxUnavailable.fillStyle = theme.muted;
    ctxUnavailable.fillText("Spike events unavailable for this run.", 12, 20);
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

  if (!values || values.length === 0) {
    ctx.fillStyle = theme.muted;
    ctx.fillText("Weights unavailable for this run.", 12, 20);
    return;
  }

  const cellW = width / cols;
  const cellH = height / rows;
  let idx = 0;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const value = values[idx % values.length];
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

function drawBarsWithLabels(canvas, labels, values, color) {
  const ctx = setupCanvas(canvas);
  if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const count = Math.max(1, labels.length);
  const gap = 3;
  const barWidth = (width - gap * (count - 1)) / count;
  const maxVal = Math.max(...values, 1e-6);
  for (let i = 0; i < count; i += 1) {
    const value = values[i % values.length] || 0;
    const norm = Math.min(1, value / maxVal);
    const barHeight = norm * (height - 20) + 6;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.55 + norm * 0.45;
    ctx.fillRect(i * (barWidth + gap), height - barHeight - 12, barWidth, barHeight);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = theme.accent;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.moveTo(0, height * 0.45);
  ctx.lineTo(width, height * 0.45);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = theme.muted;
  ctx.font = "10px 'Space Grotesk'";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  const maxLabels = Math.max(1, Math.floor(width / 60));
  const labelEvery = Math.max(1, Math.ceil(labels.length / maxLabels));
  labels.forEach((label, idx) => {
    if (idx % labelEvery !== 0) return;
    const x = idx * (barWidth + gap) + barWidth / 2;
    ctx.fillText(label, x, height);
  });
}

function drawHistogram(canvas, values) {
  const ctx = setupCanvas(canvas);
  if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const bins = 18;
  if (!values || values.length === 0) {
    ctx.fillStyle = theme.muted;
    ctx.fillText("Weight distribution unavailable.", 12, 20);
    return;
  }
  const raw = values;
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

  if (!dataState.topology) {
    ctx.fillStyle = theme.muted;
    ctx.fillText("State projection unavailable for this run.", 12, 20);
    return;
  }
  const nodes = network?.nodes || [];
  if (!nodes.length) {
    ctx.fillStyle = theme.muted;
    ctx.fillText("State projection unavailable for this run.", 12, 20);
    return;
  }

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

  nodes.forEach((node) => {
    const x = Number(node.x ?? 0.5) * width;
    const y = Number(node.y ?? 0.5) * height;
    const role = inferRoleFromPopName(node.pop || "");
    if (role === "input" || role === "input_relay") {
      ctx.fillStyle = theme.input;
    } else if (role === "output") {
      ctx.fillStyle = theme.output;
    } else {
      ctx.fillStyle = theme.hidden;
    }
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.arc(x, y, 2.4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

async function refreshData() {
  const loadCsvMaybe = (path) => {
    if (!path) {
      return Promise.resolve({ data: null, error: null, url: "(disabled)", disabled: true });
    }
    return loadCsv(path);
  };
  const loadOptionalCsvMaybe = (path) => {
    if (!path) {
      return Promise.resolve({ data: null, error: null, url: "(disabled)", disabled: true });
    }
    return loadCsv(path, { optional: true });
  };
  const loadJsonMaybe = (path) => {
    if (!path) {
      return Promise.resolve({ data: null, error: null, url: "(disabled)", disabled: true });
    }
    return loadJson(path);
  };
  const loadOptionalJsonMaybe = (path) => {
    if (!path) {
      return Promise.resolve({ data: null, error: null, url: "(disabled)", disabled: true });
    }
    return loadJson(path, { optional: true });
  };
  const [
    neuronRes,
    synapseRes,
    spikesRes,
    metricsRes,
    weightsRes,
    trialsRes,
    evalRes,
    confusionRes,
    modgridRes,
    receptorsRes,
    visionRes,
    topologyRes,
    runConfigRes,
    runFeaturesRes,
    runStatusRes,
  ] = await Promise.all([
    loadCsvMaybe(dataConfig.neuronCsv),
    loadCsvMaybe(dataConfig.synapseCsv),
    loadCsvMaybe(dataConfig.spikesCsv),
    loadCsvMaybe(dataConfig.metricsCsv),
    loadCsvMaybe(dataConfig.weightsCsv),
    loadOptionalCsvMaybe(dataConfig.trialsCsv),
    loadOptionalCsvMaybe(dataConfig.evalCsv),
    loadOptionalCsvMaybe(dataConfig.confusionCsv),
    loadOptionalJsonMaybe(dataConfig.modgridJson),
    loadOptionalCsvMaybe(dataConfig.receptorsCsv),
    loadOptionalJsonMaybe(dataConfig.visionJson),
    loadOptionalJsonMaybe(dataConfig.topologyJson),
    loadOptionalJsonMaybe(dataConfig.runConfigJson),
    loadOptionalJsonMaybe(dataConfig.runFeaturesJson),
    loadOptionalJsonMaybe(dataConfig.runStatusJson),
  ]);

  const neuronRows = neuronRes.data;
  const synapseRows = synapseRes.data;
  const spikesRows = spikesRes.data;
  const metricsRows = metricsRes.data;
  const weightsRows = weightsRes.data;
  const trialsRows = trialsRes.data;
  const evalRows = evalRes.data;
  const confusionRows = confusionRes.data;
  const modgridData = modgridRes.data;
  const receptorsRows = receptorsRes.data;
  const visionData = visionRes.data;
  const topology = topologyRes.data;

  dataState.neuronRows = neuronRows;
  dataState.synapseRows = synapseRows;
  dataState.spikesRows = spikesRows;
  dataState.metricsRows = metricsRows;
  dataState.weightsRows = weightsRows;
  dataState.trialsRows = trialsRows;
  dataState.evalRows = evalRows;
  dataState.confusionRows = confusionRows;
  dataState.modgridData = modgridData;
  dataState.receptorsRows = receptorsRows;
  dataState.visionData = visionData;
  dataState.weightsIndex = weightsRows ? buildWeightsIndex(weightsRows) : null;
  dataState.topology = topology;
  dataState.runConfig = runConfigRes.data;
  dataState.runFeatures = runFeaturesRes.data;
  dataState.runStatus = runStatusRes.data || dataState.runStatus;
  dataState.totalSteps = getTotalSteps();
  dataState.live = Boolean(
    neuronRows ||
      synapseRows ||
      spikesRows ||
      metricsRows ||
      weightsRows ||
      trialsRows ||
      evalRows ||
      confusionRows ||
      modgridData ||
      receptorsRows ||
      visionData ||
      topology
  );
  dataState.lastUpdated = new Date();

  updateDataStatus([
    neuronRes,
    synapseRes,
    spikesRes,
    metricsRes,
    weightsRes,
    trialsRes,
    evalRes,
    confusionRes,
    modgridRes,
    receptorsRes,
    visionRes,
    topologyRes,
    runConfigRes,
    runFeaturesRes,
    runStatusRes,
  ]);
  updateDataLink();
  if (dataState.runConfig) {
    const runId = dataState.runConfig.run_id || dataState.runConfig.runId || null;
    if (runId && runId !== runState.lastAppliedRunId) {
      applyRunSpecToControls(dataState.runConfig);
      runState.lastAppliedRunId = runId;
    }
  }
  renderFeatureChecklist(dataState.runFeatures);
  renderTaskPanel();
  renderAuxPanels();
  if (
    !runState.apiAvailable &&
    runStatusRes.data &&
    typeof runStatusRes.data === "object"
  ) {
    setRunStateText(runStatusRes.data);
  }

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

async function loadCsv(path, { optional = false } = {}) {
  try {
    const response = await fetch(`${path}?ts=${Date.now()}`);
    if (!response.ok) {
      if (optional && response.status === 404) {
        return { data: null, error: null, url: path, disabled: true };
      }
      return { data: null, error: `${response.status} ${response.statusText}`, url: path };
    }
    const text = await response.text();
    return { data: parseCsv(text), error: null, url: path };
  } catch (error) {
    if (optional) {
      return { data: null, error: null, url: path, disabled: true };
    }
    return { data: null, error: String(error), url: path };
  }
}

async function loadJson(path, { optional = false } = {}) {
  try {
    const response = await fetch(`${path}?ts=${Date.now()}`);
    if (!response.ok) {
      if (optional && response.status === 404) {
        return { data: null, error: null, url: path, disabled: true };
      }
      return { data: null, error: `${response.status} ${response.statusText}`, url: path };
    }
    return { data: await response.json(), error: null, url: path };
  } catch (error) {
    if (optional) {
      return { data: null, error: null, url: path, disabled: true };
    }
    return { data: null, error: String(error), url: path };
  }
}

function updateDataStatus(results) {
  if (!metricNodes.status || !metricNodes.statusDot) return;
  const failures = results.filter((res) => !res.data && res.error);
  const disabled = results.filter((res) => res?.disabled);
  if (failures.length > 0) {
    const suffix = disabled.length ? `, disabled ${disabled.length}` : "";
    metricNodes.status.textContent = `Missing data (${failures.length}${suffix})`;
    const disabledLines = disabled.map((res) => `${res.url}: disabled`);
    metricNodes.status.title = failures
      .map((res) => `${res.url}: ${res.error}`)
      .concat(disabledLines)
      .join("\n");
    metricNodes.statusDot.parentElement?.classList.remove("live");
    return;
  }

  if (dataState.live) {
    metricNodes.status.textContent = disabled.length ? "Live data (some disabled)" : "Live data";
    metricNodes.status.title = disabled.length
      ? disabled.map((res) => `${res.url}: disabled`).join("\n")
      : "";
    metricNodes.statusDot.parentElement?.classList.add("live");
  } else {
    metricNodes.status.textContent = "Demo data";
    metricNodes.status.title = "";
    metricNodes.statusDot.parentElement?.classList.remove("live");
  }
}

function resolveRunFolderUrl() {
  if (runState.activeRunPath) {
    const base = new URL(runState.activeRunPath.replace(/\/+$/, "") + "/", window.location.href);
    return base.toString();
  }
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
  const edgeSettings = readEdgeViewSettings();
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
    if (edgeSettings.showDelayTooltip && Array.isArray(screen.edges)) {
      let best = null;
      let bestDist = Infinity;
      for (const item of screen.edges) {
        const dist = pointLineDistance(mx, my, item.x1, item.y1, item.x2, item.y2);
        if (dist < bestDist) {
          bestDist = dist;
          best = item;
        }
      }
      if (best && bestDist <= Math.max(6, best.thickness + 3)) {
        const fromNode = best.from;
        const toNode = best.to;
        const fromLabel = `${fromNode.pop || "pop"}[${fromNode.localIdx ?? fromNode.index ?? 0}]`;
        const toLabel = `${toNode.pop || "pop"}[${toNode.localIdx ?? toNode.index ?? 0}]`;
        let html = `<strong>${fromLabel} -> ${toLabel}</strong><br/>` +
          `Weight: ${(best.edge.weight ?? 0).toFixed(3)}`;
        if (best.distance !== null && best.distance !== undefined) {
          html += `<br/>Dist: ${best.distance.toFixed(3)}`;
        }
        if (edgeSettings.showDelayTooltip && best.edge.delaySteps !== null && best.edge.delaySteps !== undefined) {
          html += `<br/>Delay: ${best.edge.delaySteps} steps`;
        }
        showNetworkTooltip(html, mx, my);
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
      let html =
        `<strong>${edge.from} -> ${edge.to}</strong><br/>` +
        `Synapses: ${edge.nSynapses}<br/>` +
        `Weight: ${edge.meanWeight.toFixed(3)} +/- ${edge.stdWeight.toFixed(3)}`;
      if (edgeSettings.showDelayTooltip) {
        html += `<br/>Delay steps: ${delayText}`;
      }
      showNetworkTooltip(html, mx, my);
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

  if (uiControls.weightProjection2) {
    const primary = uiControls.weightProjection.value;
    const defaultSecondary =
      projections.find((proj) => proj.toLowerCase().includes("hidden->output")) ||
      projections.find((proj) => proj !== primary) ||
      projections[0];
    const currentSecondary = uiControls.weightProjection2.value;
    uiControls.weightProjection2.innerHTML = projections
      .map((proj) => `<option value="${proj}">${proj}</option>`)
      .join("");
    uiControls.weightProjection2.value =
      projections.includes(currentSecondary) && currentSecondary !== primary
        ? currentSecondary
        : defaultSecondary;
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

  const activePops = new Set();
  filtered.forEach((row) => {
    const pop = row.pop || "pop0";
    activePops.add(pop);
  });

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

  const fontSize = 10;
  ctx.font = `${fontSize}px 'Space Grotesk'`;
  ctx.fillStyle = theme.muted;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";

  const maxLabels = Math.floor(height / 28);
  const labelEvery = Math.max(1, Math.ceil(popInfo.order.length / Math.max(1, maxLabels)));

  popInfo.order.forEach((pop, idx) => {
    const start = offsets[pop] ?? 0;
    const size = popInfo.sizes[pop] || 0;
    const bandCenter = (start + size * 0.5) / totalRows * height;

    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, (start / totalRows) * height);
    ctx.lineTo(width, (start / totalRows) * height);
    ctx.stroke();

    if (idx % labelEvery === 0 && size > 0) {
      ctx.fillText(pop, 6, bandCenter);
    }
  });

  if (activePops.size > 0) {
    const nonInputActive = Array.from(activePops).some(
      (pop) => !String(pop).toLowerCase().includes("input")
    );
    if (!nonInputActive) {
      ctx.fillStyle = theme.output;
      ctx.fillText(
        "Only input spikes in this window (no hidden/output events).",
        12,
        height - 12
      );
    }
  }
}

function drawAccuracyChart() {
  const ctx = setupCanvas(canvases.accuracy);
  if (!ctx) return;
  const { width, height } = canvases.accuracy.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  const rows =
    dataState.metricsRows && dataState.metricsRows.length
      ? dataState.metricsRows
      : dataState.evalRows;
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
  const demoId = resolveDemoId();
  const isCurriculum = demoId === "logic_curriculum";
  const train = rows.map((row) =>
    Number(
      (isCurriculum ? row.global_eval_accuracy ?? row.sample_accuracy_global : null) ??
        row.train_accuracy ??
        row.trainAcc ??
        ""
    )
  );
  const evalAcc = rows.map((row) =>
    Number(
      (isCurriculum ? row.global_eval_accuracy ?? row.sample_accuracy_global : null) ??
        row.eval_accuracy ??
        row.evalAcc ??
        ""
    )
  );
  const fallback = rows.map((row) => Number(row.spike_fraction_total || 0));

  const trainSeries = train.some((v) => !Number.isNaN(v)) ? smoothSeries(train, smooth) : fallback;
  const evalSeries = evalAcc.some((v) => !Number.isNaN(v)) ? smoothSeries(evalAcc, smooth) : null;

  drawLineSeries(ctx, trainSeries, theme.accent, width, height);
  if (evalSeries) {
    drawLineSeries(ctx, evalSeries, theme.output, width, height);
  }

  const last = rows[rows.length - 1];
  if (metricNodes.trainAccLatest) {
    const trainLatest = parseNumber(
      (isCurriculum ? last.global_eval_accuracy ?? last.sample_accuracy_global : null) ??
        last.train_accuracy ??
        last.trainAcc ??
        last.sample_accuracy
    );
    metricNodes.trainAccLatest.textContent = trainLatest !== null ? trainLatest.toFixed(4) : "--";
  }
  if (metricNodes.evalAccLatest) {
    const evalLatest = parseNumber(
      (isCurriculum ? last.global_eval_accuracy ?? last.sample_accuracy_global : null) ??
        last.eval_accuracy ??
        last.evalAcc
    );
    metricNodes.evalAccLatest.textContent = evalLatest !== null ? evalLatest.toFixed(4) : "--";
  }
  if (metricNodes.lossLatest) {
    const lossLatest = parseNumber(last.loss);
    metricNodes.lossLatest.textContent = lossLatest !== null ? lossLatest.toFixed(4) : "--";
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

function renderAuxPanels() {
  drawModulatorPanel();
  drawReceptorPanel();
  drawVisionPanel();
}

function setPanelVisible(node, visible) {
  if (!node) return;
  node.classList.toggle("hidden", !visible);
}

function setSelectOptions(select, options) {
  if (!select) return null;
  const normalized = (Array.isArray(options) ? options : [])
    .map((entry) => {
      if (entry && typeof entry === "object") {
        return {
          value: String(entry.value ?? entry.label ?? ""),
          label: String(entry.label ?? entry.value ?? ""),
        };
      }
      return { value: String(entry ?? ""), label: String(entry ?? "") };
    })
    .filter((entry) => entry.value.length > 0);

  const previous = String(select.value || "");
  while (select.firstChild) {
    select.removeChild(select.firstChild);
  }
  if (normalized.length === 0) {
    return null;
  }
  normalized.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.value;
    option.textContent = entry.label;
    select.appendChild(option);
  });
  const values = normalized.map((entry) => entry.value);
  const selected = values.includes(previous) ? previous : values[0];
  select.value = selected;
  return selected;
}

function parseNumericMatrix2D(raw) {
  if (!Array.isArray(raw) || raw.length === 0) return null;
  if (!Array.isArray(raw[0])) return null;
  if (raw[0].length === 0) return null;
  if (typeof raw[0][0] !== "number") return null;
  return raw;
}

function toGridChannels(raw) {
  if (!Array.isArray(raw) || raw.length === 0) return [];
  const as2d = parseNumericMatrix2D(raw);
  if (as2d) return [as2d];
  if (Array.isArray(raw[0]) && Array.isArray(raw[0][0])) {
    return raw
      .map((entry) => parseNumericMatrix2D(entry))
      .filter((entry) => Array.isArray(entry));
  }
  return [];
}

function toVisionGrayMatrix(raw) {
  const as2d = parseNumericMatrix2D(raw);
  if (as2d) return as2d;
  const channels = toGridChannels(raw);
  if (channels.length === 0) return null;
  const first = channels[0];
  const h = first.length;
  const w = first[0]?.length || 0;
  if (h <= 0 || w <= 0) return null;
  if (channels.length === 1) return first;
  if (channels.length < 3) return first;
  const out = new Array(h);
  for (let y = 0; y < h; y += 1) {
    const row = new Array(w);
    for (let x = 0; x < w; x += 1) {
      const r = Number(channels[0][y][x] || 0);
      const g = Number(channels[1][y][x] || 0);
      const b = Number(channels[2][y][x] || 0);
      row[x] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    out[y] = row;
  }
  return out;
}

function hexToRgb(hex) {
  const clean = String(hex || "").trim().replace("#", "");
  if (clean.length !== 6) {
    return { r: 255, g: 255, b: 255 };
  }
  const value = Number.parseInt(clean, 16);
  if (!Number.isFinite(value)) {
    return { r: 255, g: 255, b: 255 };
  }
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  };
}

function paletteColor(t, palette) {
  const stops = Array.isArray(palette) && palette.length > 1 ? palette : ["#000000", "#ffffff"];
  const x = clamp01(t) * (stops.length - 1);
  const i0 = Math.floor(x);
  const i1 = Math.min(stops.length - 1, i0 + 1);
  const frac = x - i0;
  const c0 = hexToRgb(stops[i0]);
  const c1 = hexToRgb(stops[i1]);
  const r = Math.round(c0.r + (c1.r - c0.r) * frac);
  const g = Math.round(c0.g + (c1.g - c0.g) * frac);
  const b = Math.round(c0.b + (c1.b - c0.b) * frac);
  return `rgb(${r}, ${g}, ${b})`;
}

function drawMatrixHeatmap(canvas, matrix, { palette, fixedRange, emptyMessage } = {}) {
  const ctx = setupCanvas(canvas);
  if (!ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);

  if (!Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0]) || matrix[0].length === 0) {
    ctx.fillStyle = theme.muted;
    ctx.fillText(emptyMessage || "Data unavailable.", 12, 20);
    return;
  }

  const values = [];
  matrix.forEach((row) => {
    if (!Array.isArray(row)) return;
    row.forEach((value) => {
      const numeric = Number(value);
      if (Number.isFinite(numeric)) {
        values.push(numeric);
      }
    });
  });
  if (values.length === 0) {
    ctx.fillStyle = theme.muted;
    ctx.fillText(emptyMessage || "Data unavailable.", 12, 20);
    return;
  }

  let minVal = Math.min(...values);
  let maxVal = Math.max(...values);
  if (Array.isArray(fixedRange) && fixedRange.length === 2) {
    minVal = Number(fixedRange[0]);
    maxVal = Number(fixedRange[1]);
  }
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal) || Math.abs(maxVal - minVal) < 1e-12) {
    maxVal = minVal + 1;
  }

  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellW = width / cols;
  const cellH = height / rows;

  for (let y = 0; y < rows; y += 1) {
    const row = matrix[y];
    for (let x = 0; x < cols; x += 1) {
      const value = Number(row[x]);
      const t = Number.isFinite(value) ? clamp01((value - minVal) / (maxVal - minVal)) : 0;
      ctx.fillStyle = paletteColor(t, palette || theme.heat);
      ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
    }
  }
}

function drawModulatorPanel() {
  const hasPath = Boolean(dataConfig.modgridJson);
  if (!hasPath) {
    setPanelVisible(auxNodes.modulatorCard, false);
    return;
  }
  const payload = dataState.modgridData;
  const grids = payload && typeof payload === "object" ? payload.grids : null;
  const fieldNames = grids && typeof grids === "object" ? Object.keys(grids).sort() : [];
  if (fieldNames.length === 0) {
    setPanelVisible(auxNodes.modulatorCard, false);
    return;
  }

  const selectedField = setSelectOptions(
    auxNodes.modulatorFieldSelect,
    fieldNames.map((name) => ({ value: name, label: name }))
  );
  if (!selectedField) {
    setPanelVisible(auxNodes.modulatorCard, false);
    return;
  }

  const channels = toGridChannels(grids[selectedField]);
  if (channels.length === 0) {
    setPanelVisible(auxNodes.modulatorCard, false);
    return;
  }

  const kindsByField =
    payload && payload.kinds && typeof payload.kinds === "object" ? payload.kinds : {};
  const kindLabels = Array.isArray(kindsByField[selectedField]) ? kindsByField[selectedField] : [];
  const selectedKind = Number(
    setSelectOptions(
      auxNodes.modulatorKindSelect,
      channels.map((_, idx) => ({
        value: String(idx),
        label: String(kindLabels[idx] ?? `kind_${idx}`),
      }))
    ) || "0"
  );
  const matrix = channels[Math.max(0, Math.min(channels.length - 1, selectedKind))] || channels[0];

  setPanelVisible(auxNodes.modulatorCard, true);
  drawMatrixHeatmap(canvases.modulator, matrix, {
    palette: theme.heat,
    emptyMessage: "Modulator grid unavailable.",
  });
}

function drawReceptorPanel() {
  const hasPath = Boolean(dataConfig.receptorsCsv);
  if (!hasPath) {
    setPanelVisible(auxNodes.receptorCard, false);
    return;
  }

  const rows = Array.isArray(dataState.receptorsRows) ? dataState.receptorsRows : null;
  if (!rows || rows.length === 0) {
    setPanelVisible(auxNodes.receptorCard, false);
    return;
  }

  const projectionMap = new Map();
  rows.forEach((row) => {
    const proj = String(row.proj || "").trim();
    const comp = String(row.comp || "").trim();
    if (!proj) return;
    const key = comp ? `${proj}::${comp}` : proj;
    const label = comp ? `${proj}/${comp}` : proj;
    if (!projectionMap.has(key)) {
      projectionMap.set(key, label);
    }
  });
  const projectionOptions = Array.from(projectionMap.entries())
    .sort((a, b) => String(a[1]).localeCompare(String(b[1])))
    .map(([value, label]) => ({ value, label }));
  if (projectionOptions.length === 0) {
    setPanelVisible(auxNodes.receptorCard, false);
    return;
  }

  const numericMetricKeys = new Set();
  rows.forEach((row) => {
    Object.entries(row || {}).forEach(([key, value]) => {
      if (key === "step" || key === "t" || key === "proj" || key === "comp") return;
      const numeric = Number(value);
      if (Number.isFinite(numeric)) {
        numericMetricKeys.add(key);
      }
    });
  });
  const metricOptions = Array.from(numericMetricKeys)
    .sort((a, b) => String(a).localeCompare(String(b)))
    .map((key) => ({ value: key, label: key }));
  if (metricOptions.length === 0) {
    setPanelVisible(auxNodes.receptorCard, false);
    return;
  }

  const selectedProjection = setSelectOptions(auxNodes.receptorProjectionSelect, projectionOptions);
  const selectedMetric = setSelectOptions(auxNodes.receptorMetricSelect, metricOptions);
  if (!selectedProjection || !selectedMetric) {
    setPanelVisible(auxNodes.receptorCard, false);
    return;
  }

  const [selectedProj, selectedComp = ""] = String(selectedProjection).split("::");
  const series = rows
    .filter((row) => {
      const proj = String(row.proj || "").trim();
      const comp = String(row.comp || "").trim();
      return proj === selectedProj && comp === selectedComp;
    })
    .map((row) => Number(row[selectedMetric]));
  const hasSeries = series.some((value) => Number.isFinite(value));

  setPanelVisible(auxNodes.receptorCard, true);
  const ctx = setupCanvas(canvases.receptor);
  if (!ctx) return;
  const { width, height } = canvases.receptor.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);
  if (!hasSeries) {
    ctx.fillStyle = theme.muted;
    ctx.fillText("No receptor series for selected projection.", 12, 20);
    return;
  }
  const color = selectedMetric.includes("gaba") ? theme.inhib : theme.accent;
  drawLineSeries(ctx, series, color, width, height);
}

function drawVisionPanel() {
  const hasPath = Boolean(dataConfig.visionJson);
  if (!hasPath) {
    setPanelVisible(auxNodes.visionCard, false);
    return;
  }
  const payload = dataState.visionData;
  const matrix =
    payload && payload.available && payload.data !== undefined
      ? toVisionGrayMatrix(payload.data)
      : null;
  if (!matrix) {
    setPanelVisible(auxNodes.visionCard, false);
    return;
  }

  let maxValue = -Infinity;
  for (let y = 0; y < matrix.length; y += 1) {
    const row = matrix[y];
    for (let x = 0; x < row.length; x += 1) {
      const value = Number(row[x]);
      if (Number.isFinite(value)) {
        maxValue = Math.max(maxValue, value);
      }
    }
  }
  const range = maxValue > 1.0 ? [0, 255] : [0, 1];

  setPanelVisible(auxNodes.visionCard, true);
  drawMatrixHeatmap(canvases.vision, matrix, {
    palette: ["#000000", "#ffffff"],
    fixedRange: range,
    emptyMessage: "Vision frame unavailable.",
  });
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
    return { matrix: [[0]], counts: [[0]], nPre: 1, nPost: 1 };
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

  return { matrix, counts, nPre: preDim, nPost: postDim };
}

function drawHeatmapMatrix(canvas, matrix, clampMin, clampMax, counts) {
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
  const maxAbs = Math.max(Math.abs(minVal), Math.abs(maxVal), 1e-6);

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const value = matrix[r][c];
      const count = counts ? counts[r][c] : 1;
      if (count <= 0) {
        continue;
      }
      const mag = Math.min(Math.abs(value), maxAbs);
      const clamped = mag / maxAbs;
      const color = value >= 0 ? theme.excit : theme.inhib;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.15 + clamped * 0.85;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }
  ctx.globalAlpha = 1;
}

function normalizeRunPath(value) {
  if (!value) return null;
  const path = String(value).trim();
  if (!path) return null;
  return path.endsWith("/") ? path.slice(0, -1) : path;
}

function applyRunDirectory(runPath, { updateUrl = false } = {}) {
  const normalized = normalizeRunPath(runPath);
  if (!normalized) return;
  if (runState.activeRunPath === normalized) return;
  runState.activeRunPath = normalized;
  dataConfig.runPath = normalized;
  dataConfig.topologyJson = `${normalized}/topology.json`;
  dataConfig.neuronCsv = `${normalized}/neuron.csv`;
  dataConfig.synapseCsv = `${normalized}/synapse.csv`;
  dataConfig.spikesCsv = `${normalized}/spikes.csv`;
  dataConfig.metricsCsv = `${normalized}/metrics.csv`;
  dataConfig.weightsCsv = `${normalized}/weights.csv`;
  dataConfig.trialsCsv = `${normalized}/trials.csv`;
  dataConfig.evalCsv = `${normalized}/eval.csv`;
  dataConfig.confusionCsv = `${normalized}/confusion.csv`;
  dataConfig.modgridJson = `${normalized}/modgrid.json`;
  dataConfig.receptorsCsv = `${normalized}/receptors.csv`;
  dataConfig.visionJson = `${normalized}/vision.json`;
  dataConfig.runConfigJson = `${normalized}/run_config.json`;
  dataConfig.runFeaturesJson = `${normalized}/run_features.json`;
  dataConfig.runStatusJson = `${normalized}/run_status.json`;
  updateDataLink();
  if (updateUrl) {
    const url = new URL(window.location.href);
    [
      "topology",
      "neuron",
      "synapse",
      "spikes",
      "metrics",
      "weights",
      "trials",
      "eval",
      "confusion",
      "modgrid",
      "receptors",
      "vision",
    ].forEach((key) => url.searchParams.delete(key));
    url.searchParams.set("run", normalized);
    window.history.replaceState({}, "", url.toString());
  }
}

function setRunStateText(status) {
  if (!runNodes.stateText) return;
  if (!status) {
    runNodes.stateText.textContent = "State: unknown";
    return;
  }
  const runId = status.run_id ? ` (${status.run_id})` : "";
  runNodes.stateText.textContent = `State: ${status.state || "unknown"}${runId}`;
}

function updateAdvancedBiologyVisibility() {
  const demoId = String(runNodes.demoSelect?.value || "").trim().toLowerCase();
  const isLogicDemo = isLogicDemoId(demoId);
  if (runNodes.logicBackendWrap) {
    runNodes.logicBackendWrap.classList.toggle("hidden", !isLogicDemo);
  }
  if (runNodes.advancedSection && isLogicDemo) {
    runNodes.advancedSection.open = true;
  }
}

function setRunControlsEnabled(enabled) {
  const selectionControls = [
    runNodes.demoSelect,
    runNodes.stepsInput,
    runNodes.deviceSelect,
    runNodes.fusedLayoutSelect,
    runNodes.ringStrategySelect,
    runNodes.learningToggle,
    runNodes.monitorsToggle,
    runNodes.modulatorToggle,
    runNodes.logicBackendSelect,
    runNodes.explorationEnabledToggle,
    runNodes.epsilonStartInput,
    runNodes.epsilonEndInput,
    runNodes.epsilonDecayTrialsInput,
    runNodes.tieBreakSelect,
    runNodes.rewardDeliveryStepsInput,
    runNodes.advancedSynapseEnabledToggle,
    runNodes.advancedSynapseConductanceToggle,
    runNodes.advancedSynapseNmdaBlockToggle,
    runNodes.advancedSynapseStpToggle,
    runNodes.receptorModeSelect,
    runNodes.modulatorFieldTypeSelect,
    runNodes.modulatorKindsSelect,
    runNodes.wrapperEnabledToggle,
    runNodes.wrapperAchGainInput,
    runNodes.wrapperNeGainInput,
    runNodes.wrapperHtDecayInput,
    runNodes.excitabilityEnabledToggle,
    runNodes.excitabilityAchGainInput,
    runNodes.excitabilityNeGainInput,
    runNodes.excitabilityHtGainInput,
    runNodes.homeostasisEnabledToggle,
    runNodes.homeostasisAlphaInput,
    runNodes.homeostasisEtaInput,
    runNodes.homeostasisTargetInput,
    runNodes.homeostasisClampMinInput,
    runNodes.homeostasisClampMaxInput,
    runNodes.homeostasisScopeSelect,
    runNodes.pruningEnabledToggle,
    runNodes.neurogenesisEnabledToggle,
  ];
  selectionControls.forEach((control) => {
    if (control) control.disabled = false;
  });

  const actionControls = [
    runNodes.startButton,
    runNodes.stopButton,
  ];
  actionControls.forEach((control) => {
    if (!control) return;
    control.disabled = !enabled;
  });
}

function renderFeatureChecklist(features) {
  if (!runNodes.featureList) return;
  if (!features || typeof features !== "object") {
    runNodes.featureList.innerHTML = "<li>No feature manifest loaded.</li>";
    return;
  }
  const learning = features.learning || {};
  const delays = features.delays || {};
  const modulators = features.modulators || {};
  const synapse = features.synapse || {};
  const advancedSynapse = features.advanced_synapse || {};
  const wrapper = features.wrapper || {};
  const excitability = features.excitability_modulation || {};
  const exploration = features.exploration || {};
  const rewardWindow = features.reward_window || {};
  const monitor = features.monitor || {};
  const monitorsEnabled = monitor.enabled !== false;
  const items = [
    `Learning: <strong>${learning.enabled ? "ON" : "OFF"}</strong>` +
      (learning.enabled ? ` (${learning.rule || "rule?"}, lr=${learning.lr ?? "?"})` : ""),
    `Delays: <strong>${delays.enabled ? "ON" : "OFF"}</strong>` +
      ` (max=${delays.max_delay_steps ?? "?"}, ring_len=${delays.ring_len ?? "?"})`,
    `Modulators: <strong>${modulators.enabled ? "ON" : "OFF"}</strong>` +
      ` (${(modulators.kinds || []).join(", ") || "none"})`,
    `Synapse backend: <strong>${synapse.backend || "unknown"}</strong>`,
    `Advanced synapse: <strong>${advancedSynapse.enabled ? "ON" : "OFF"}</strong>` +
      (advancedSynapse.enabled
        ? ` (conductance=${advancedSynapse.conductance_mode ? "yes" : "no"}, nmda=${advancedSynapse.nmda_voltage_block ? "yes" : "no"}, stp=${advancedSynapse.stp_enabled ? "yes" : "no"})`
        : ""),
    `Wrapper: <strong>${wrapper.enabled ? "ON" : "OFF"}</strong>`,
    `Excitability modulation: <strong>${excitability.enabled ? "ON" : "OFF"}</strong>`,
    `Exploration: <strong>${exploration.enabled ? "ON" : "OFF"}</strong>` +
      (exploration.enabled
        ? ` (${exploration.mode || "epsilon_greedy"}, ${exploration.epsilon_start ?? "?"}→${exploration.epsilon_end ?? "?"})`
        : ""),
    `Reward window: <strong>${(rewardWindow.steps || 0) > 0 ? "ON" : "OFF"}</strong>` +
      ` (${rewardWindow.steps ?? 0} step${Number(rewardWindow.steps || 0) === 1 ? "" : "s"})`,
    `Fused layout: <strong>${synapse.fused_layout || "unknown"}</strong>`,
    `Ring: <strong>${synapse.ring_strategy || "unknown"}</strong>` +
      ` (dtype=${synapse.ring_dtype || "none"})`,
    `Monitors: <strong>${monitorsEnabled ? "ON" : "OFF"}</strong>`,
    `Monitor policy: <strong>${monitor.sync_policy || monitor.mode || "unknown"}</strong>`,
  ];
  runNodes.featureList.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
}

function applyRunSpecToControls(spec) {
  if (!spec || typeof spec !== "object") return;
  const synapse = spec.synapse && typeof spec.synapse === "object" ? spec.synapse : {};
  const modulators = spec.modulators && typeof spec.modulators === "object" ? spec.modulators : {};
  const advancedSynapse =
    spec.advanced_synapse && typeof spec.advanced_synapse === "object" ? spec.advanced_synapse : {};
  const wrapper = spec.wrapper && typeof spec.wrapper === "object" ? spec.wrapper : {};
  const excitability =
    spec.excitability_modulation && typeof spec.excitability_modulation === "object"
      ? spec.excitability_modulation
      : {};
  const logic = spec.logic && typeof spec.logic === "object" ? spec.logic : {};
  const exploration = logic.exploration && typeof logic.exploration === "object" ? logic.exploration : {};
  const homeostasis = spec.homeostasis && typeof spec.homeostasis === "object" ? spec.homeostasis : {};
  const pruning = spec.pruning && typeof spec.pruning === "object" ? spec.pruning : {};
  const neurogenesis = spec.neurogenesis && typeof spec.neurogenesis === "object" ? spec.neurogenesis : {};
  if (runNodes.demoSelect && spec.demo_id) runNodes.demoSelect.value = String(spec.demo_id);
  if (runNodes.stepsInput && Number.isFinite(Number(spec.steps))) {
    runNodes.stepsInput.value = String(Math.max(1, Number(spec.steps)));
  }
  if (runNodes.deviceSelect && spec.device) runNodes.deviceSelect.value = String(spec.device);
  if (runNodes.fusedLayoutSelect) {
    runNodes.fusedLayoutSelect.value = String(synapse.fused_layout || spec.fused_layout || "auto");
  }
  if (runNodes.ringStrategySelect) {
    runNodes.ringStrategySelect.value = String(synapse.ring_strategy || spec.ring_strategy || "dense");
  }
  if (runNodes.learningToggle) {
    runNodes.learningToggle.checked = Boolean(spec.learning?.enabled);
  }
  if (runNodes.monitorsToggle) {
    runNodes.monitorsToggle.checked = spec.monitors_enabled !== false;
  }
  if (runNodes.modulatorToggle) {
    runNodes.modulatorToggle.checked = Boolean(modulators.enabled);
  }
  if (runNodes.logicBackendSelect) {
    runNodes.logicBackendSelect.value = String(spec.logic_backend || "harness");
  }
  if (runNodes.explorationEnabledToggle) {
    runNodes.explorationEnabledToggle.checked = Boolean(exploration.enabled);
  }
  if (runNodes.epsilonStartInput) {
    runNodes.epsilonStartInput.value = String(parseNumberOr(exploration.epsilon_start, 0.2));
  }
  if (runNodes.epsilonEndInput) {
    runNodes.epsilonEndInput.value = String(parseNumberOr(exploration.epsilon_end, 0.01));
  }
  if (runNodes.epsilonDecayTrialsInput) {
    runNodes.epsilonDecayTrialsInput.value = String(
      Math.max(1, parseInteger(exploration.epsilon_decay_trials) || 3000)
    );
  }
  if (runNodes.tieBreakSelect) {
    runNodes.tieBreakSelect.value = String(exploration.tie_break || "random_among_max");
  }
  if (runNodes.rewardDeliveryStepsInput) {
    runNodes.rewardDeliveryStepsInput.value = String(
      Math.max(0, parseInteger(logic.reward_delivery_steps) || 2)
    );
  }
  if (runNodes.advancedSynapseEnabledToggle) {
    runNodes.advancedSynapseEnabledToggle.checked = Boolean(advancedSynapse.enabled);
  }
  if (runNodes.advancedSynapseConductanceToggle) {
    runNodes.advancedSynapseConductanceToggle.checked = Boolean(advancedSynapse.conductance_mode);
  }
  if (runNodes.advancedSynapseNmdaBlockToggle) {
    runNodes.advancedSynapseNmdaBlockToggle.checked = Boolean(advancedSynapse.nmda_voltage_block);
  }
  if (runNodes.advancedSynapseStpToggle) {
    runNodes.advancedSynapseStpToggle.checked = Boolean(advancedSynapse.stp_enabled);
  }
  if (runNodes.receptorModeSelect) {
    runNodes.receptorModeSelect.value = String(synapse.receptor_mode || spec.receptor_mode || "exc_only");
  }
  if (runNodes.modulatorFieldTypeSelect) {
    runNodes.modulatorFieldTypeSelect.value = String(modulators.field_type || "global_scalar");
  }
  if (runNodes.modulatorKindsSelect) {
    const kinds = Array.isArray(modulators.kinds) ? modulators.kinds : [];
    setMultiSelectValues(runNodes.modulatorKindsSelect, kinds);
  }
  if (runNodes.wrapperEnabledToggle) {
    runNodes.wrapperEnabledToggle.checked = Boolean(wrapper.enabled);
  }
  if (runNodes.wrapperAchGainInput) {
    runNodes.wrapperAchGainInput.value = String(parseNumberOr(wrapper.ach_lr_gain, 0.0));
  }
  if (runNodes.wrapperNeGainInput) {
    runNodes.wrapperNeGainInput.value = String(parseNumberOr(wrapper.ne_lr_gain, 0.0));
  }
  if (runNodes.wrapperHtDecayInput) {
    runNodes.wrapperHtDecayInput.value = String(parseNumberOr(wrapper.ht_extra_weight_decay, 0.0));
  }
  if (runNodes.excitabilityEnabledToggle) {
    runNodes.excitabilityEnabledToggle.checked = Boolean(excitability.enabled);
  }
  if (runNodes.excitabilityAchGainInput) {
    runNodes.excitabilityAchGainInput.value = String(parseNumberOr(excitability.ach_gain, 0.0));
  }
  if (runNodes.excitabilityNeGainInput) {
    runNodes.excitabilityNeGainInput.value = String(parseNumberOr(excitability.ne_gain, 0.0));
  }
  if (runNodes.excitabilityHtGainInput) {
    runNodes.excitabilityHtGainInput.value = String(parseNumberOr(excitability.ht_gain, 0.0));
  }
  if (runNodes.homeostasisEnabledToggle) {
    runNodes.homeostasisEnabledToggle.checked = Boolean(homeostasis.enabled);
  }
  if (runNodes.homeostasisAlphaInput) {
    runNodes.homeostasisAlphaInput.value = String(parseNumberOr(homeostasis.alpha, 0.01));
  }
  if (runNodes.homeostasisEtaInput) {
    runNodes.homeostasisEtaInput.value = String(parseNumberOr(homeostasis.eta, 0.001));
  }
  if (runNodes.homeostasisTargetInput) {
    runNodes.homeostasisTargetInput.value = String(parseNumberOr(homeostasis.r_target, 0.05));
  }
  if (runNodes.homeostasisClampMinInput) {
    runNodes.homeostasisClampMinInput.value = String(parseNumberOr(homeostasis.clamp_min, 0.0));
  }
  if (runNodes.homeostasisClampMaxInput) {
    runNodes.homeostasisClampMaxInput.value = String(parseNumberOr(homeostasis.clamp_max, 0.05));
  }
  if (runNodes.homeostasisScopeSelect) {
    runNodes.homeostasisScopeSelect.value = String(homeostasis.scope || "per_neuron");
  }
  if (runNodes.pruningEnabledToggle) {
    runNodes.pruningEnabledToggle.checked = Boolean(pruning.enabled);
  }
  if (runNodes.neurogenesisEnabledToggle) {
    runNodes.neurogenesisEnabledToggle.checked = Boolean(neurogenesis.enabled);
  }
  updateAdvancedBiologyVisibility();
}

function selectedDemoDefaults() {
  const id = runNodes.demoSelect?.value;
  if (!id) return null;
  const found = runState.demos.find((demo) => String(demo.id) === String(id));
  if (!found || !found.defaults || typeof found.defaults !== "object") return null;
  try {
    return JSON.parse(JSON.stringify(found.defaults));
  } catch {
    return null;
  }
}

function buildRunSpecFromControls() {
  const base = selectedDemoDefaults() || {};
  const demoId = String(runNodes.demoSelect?.value || base.demo_id || "network");
  const isLogicDemo = isLogicDemoId(demoId);
  const steps = Math.max(1, Number(runNodes.stepsInput?.value || base.steps || 200));
  const device = runNodes.deviceSelect?.value || base.device || "cpu";
  const baseSynapse = base.synapse && typeof base.synapse === "object" ? base.synapse : {};
  const baseModulators = base.modulators && typeof base.modulators === "object" ? base.modulators : {};
  const baseAdvancedSynapse =
    base.advanced_synapse && typeof base.advanced_synapse === "object" ? base.advanced_synapse : {};
  const baseWrapper = base.wrapper && typeof base.wrapper === "object" ? base.wrapper : {};
  const baseExcitability =
    base.excitability_modulation && typeof base.excitability_modulation === "object"
      ? base.excitability_modulation
      : {};
  const baseLogic = base.logic && typeof base.logic === "object" ? base.logic : {};
  const baseExploration =
    baseLogic.exploration && typeof baseLogic.exploration === "object"
      ? baseLogic.exploration
      : {};
  const baseHomeostasis = base.homeostasis && typeof base.homeostasis === "object" ? base.homeostasis : {};
  const basePruning = base.pruning && typeof base.pruning === "object" ? base.pruning : {};
  const baseNeurogenesis = base.neurogenesis && typeof base.neurogenesis === "object" ? base.neurogenesis : {};

  const synapseBackend = String(baseSynapse.backend || base.synapse_backend || "spmm_fused");
  const fusedLayout = String(runNodes.fusedLayoutSelect?.value || baseSynapse.fused_layout || base.fused_layout || "auto");
  const ringStrategy = String(runNodes.ringStrategySelect?.value || baseSynapse.ring_strategy || base.ring_strategy || "dense");
  const receptorMode = String(runNodes.receptorModeSelect?.value || baseSynapse.receptor_mode || base.receptor_mode || "exc_only");
  const ringDtype = String(baseSynapse.ring_dtype || base.ring_dtype || "none");
  const storeSparseByDelay =
    baseSynapse.store_sparse_by_delay !== undefined
      ? baseSynapse.store_sparse_by_delay
      : (base.store_sparse_by_delay ?? null);

  const learningEnabled = Boolean(runNodes.learningToggle?.checked);
  const monitorsEnabled = runNodes.monitorsToggle ? Boolean(runNodes.monitorsToggle.checked) : base.monitors_enabled !== false;
  const modEnabled = Boolean(runNodes.modulatorToggle?.checked);
  const baseModKinds = Array.isArray(baseModulators.kinds)
    ? baseModulators.kinds.map((item) => String(item).trim()).filter(Boolean)
    : [];
  const selectedModKinds = getMultiSelectValues(runNodes.modulatorKindsSelect);
  const modKinds = modEnabled
    ? (selectedModKinds.length > 0 ? selectedModKinds : (baseModKinds.length > 0 ? baseModKinds : ["dopamine"]))
    : [];
  const modulatorFieldType = String(runNodes.modulatorFieldTypeSelect?.value || baseModulators.field_type || "global_scalar");
  const homeostasisEnabled = Boolean(runNodes.homeostasisEnabledToggle?.checked);
  const pruningEnabled = Boolean(runNodes.pruningEnabledToggle?.checked);
  const neurogenesisEnabled = Boolean(runNodes.neurogenesisEnabledToggle?.checked);
  const logicBackend = String(runNodes.logicBackendSelect?.value || base.logic_backend || "harness");
  const advancedSynapseEnabled = Boolean(runNodes.advancedSynapseEnabledToggle?.checked);
  const advancedSynapseConductance = Boolean(runNodes.advancedSynapseConductanceToggle?.checked);
  const advancedSynapseNmdaBlock = Boolean(runNodes.advancedSynapseNmdaBlockToggle?.checked);
  const advancedSynapseStp = Boolean(runNodes.advancedSynapseStpToggle?.checked);
  const wrapperEnabled = Boolean(runNodes.wrapperEnabledToggle?.checked);
  const excitabilityEnabled = Boolean(runNodes.excitabilityEnabledToggle?.checked);
  const explorationEnabled = Boolean(runNodes.explorationEnabledToggle?.checked);
  const epsilonStart = parseNumberOr(
    runNodes.epsilonStartInput?.value,
    parseNumberOr(baseExploration.epsilon_start, 0.2)
  );
  const epsilonEnd = parseNumberOr(
    runNodes.epsilonEndInput?.value,
    parseNumberOr(baseExploration.epsilon_end, 0.01)
  );
  const epsilonDecayTrials = Math.max(
    1,
    parseInteger(runNodes.epsilonDecayTrialsInput?.value) ||
      parseInteger(baseExploration.epsilon_decay_trials) ||
      3000
  );
  const tieBreak = String(runNodes.tieBreakSelect?.value || baseExploration.tie_break || "random_among_max");
  const rewardDeliverySteps = Math.max(
    0,
    parseInteger(runNodes.rewardDeliveryStepsInput?.value) ||
      parseInteger(baseLogic.reward_delivery_steps) ||
      2
  );
  const baseLogicLearningMode = String(base.logic_learning_mode || "rstdp").trim().toLowerCase();
  const logicLearningMode = isLogicDemo
    ? (learningEnabled
      ? (baseLogicLearningMode === "none" ? "rstdp" : baseLogicLearningMode)
      : "none")
    : baseLogicLearningMode;

  return {
    ...base,
    demo_id: demoId,
    steps,
    device,
    logic_backend: logicBackend,
    logic_learning_mode: isLogicDemo ? logicLearningMode : base.logic_learning_mode,
    fused_layout: fusedLayout,
    synapse_backend: synapseBackend,
    ring_strategy: ringStrategy,
    ring_dtype: ringDtype,
    store_sparse_by_delay: storeSparseByDelay,
    receptor_mode: receptorMode,
    monitors_enabled: monitorsEnabled,
    monitor_mode: "dashboard",
    logic: {
      ...baseLogic,
      reward_delivery_steps: rewardDeliverySteps,
      reward_delivery_clamp_input:
        baseLogic.reward_delivery_clamp_input === undefined
          ? true
          : Boolean(baseLogic.reward_delivery_clamp_input),
      exploration: {
        ...baseExploration,
        enabled: explorationEnabled,
        mode: "epsilon_greedy",
        epsilon_start: epsilonStart,
        epsilon_end: epsilonEnd,
        epsilon_decay_trials: epsilonDecayTrials,
        tie_break: tieBreak,
      },
    },
    synapse: {
      ...baseSynapse,
      backend: synapseBackend,
      fused_layout: fusedLayout,
      ring_strategy: ringStrategy,
      ring_dtype: ringDtype,
      store_sparse_by_delay: storeSparseByDelay,
      receptor_mode: receptorMode,
    },
    advanced_synapse: {
      ...baseAdvancedSynapse,
      enabled: advancedSynapseEnabled,
      conductance_mode: advancedSynapseConductance,
      nmda_voltage_block: advancedSynapseNmdaBlock,
      stp_enabled: advancedSynapseStp,
    },
    learning: {
      ...(base.learning || {}),
      enabled: learningEnabled,
    },
    modulators: {
      ...baseModulators,
      enabled: modEnabled,
      kinds: modKinds,
      field_type: modulatorFieldType,
    },
    wrapper: {
      ...baseWrapper,
      enabled: wrapperEnabled,
      ach_lr_gain: parseNumberOr(
        runNodes.wrapperAchGainInput?.value,
        parseNumberOr(baseWrapper.ach_lr_gain, 0.0)
      ),
      ne_lr_gain: parseNumberOr(
        runNodes.wrapperNeGainInput?.value,
        parseNumberOr(baseWrapper.ne_lr_gain, 0.0)
      ),
      ht_extra_weight_decay: parseNumberOr(
        runNodes.wrapperHtDecayInput?.value,
        parseNumberOr(baseWrapper.ht_extra_weight_decay, 0.0)
      ),
    },
    excitability_modulation: {
      ...baseExcitability,
      enabled: excitabilityEnabled,
      ach_gain: parseNumberOr(
        runNodes.excitabilityAchGainInput?.value,
        parseNumberOr(baseExcitability.ach_gain, 0.0)
      ),
      ne_gain: parseNumberOr(
        runNodes.excitabilityNeGainInput?.value,
        parseNumberOr(baseExcitability.ne_gain, 0.0)
      ),
      ht_gain: parseNumberOr(
        runNodes.excitabilityHtGainInput?.value,
        parseNumberOr(baseExcitability.ht_gain, 0.0)
      ),
    },
    homeostasis: {
      ...baseHomeostasis,
      enabled: homeostasisEnabled,
      rule: "rate_ema_threshold",
      alpha: parseNumberOr(runNodes.homeostasisAlphaInput?.value, parseNumberOr(baseHomeostasis.alpha, 0.01)),
      eta: parseNumberOr(runNodes.homeostasisEtaInput?.value, parseNumberOr(baseHomeostasis.eta, 0.001)),
      r_target: parseNumberOr(
        runNodes.homeostasisTargetInput?.value,
        parseNumberOr(baseHomeostasis.r_target, 0.05)
      ),
      clamp_min: parseNumberOr(
        runNodes.homeostasisClampMinInput?.value,
        parseNumberOr(baseHomeostasis.clamp_min, 0.0)
      ),
      clamp_max: parseNumberOr(
        runNodes.homeostasisClampMaxInput?.value,
        parseNumberOr(baseHomeostasis.clamp_max, 0.05)
      ),
      scope: String(runNodes.homeostasisScopeSelect?.value || baseHomeostasis.scope || "per_neuron"),
    },
    pruning: {
      ...basePruning,
      enabled: pruningEnabled,
    },
    neurogenesis: {
      ...baseNeurogenesis,
      enabled: neurogenesisEnabled,
    },
  };
}

async function apiJson(path, options) {
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timeoutId = controller
    ? setTimeout(() => {
        controller.abort();
      }, dataConfig.apiTimeoutMs)
    : null;
  try {
    const response = await fetch(path, {
      ...(options || {}),
      signal: controller ? controller.signal : undefined,
      cache: "no-store",
    });
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch {
    return null;
  } finally {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
  }
}

async function refreshDashboardApiBootstrap() {
  if (runState.apiBootstrapInFlight) return;
  runState.apiBootstrapInFlight = true;
  try {
    if (!dataConfig.useApi) {
      runState.demos = fallbackDemos();
      renderDemoSelectOptions(runState.demos);
      runState.apiAvailable = false;
      setRunStateText({ state: "api-disabled" });
      setRunControlsEnabled(false);
      return;
    }
    const demosPayload = await apiJson("/api/demos");
    const statusPayload = await apiJson("/api/run/status");
    if (!demosPayload || !Array.isArray(demosPayload.demos)) {
      runState.apiAvailable = false;
      runState.demos = fallbackDemos();
      renderDemoSelectOptions(runState.demos);
      setRunStateText({ state: "api-unavailable" });
      setRunControlsEnabled(false);
      return;
    }
    runState.apiAvailable = true;
    setRunControlsEnabled(true);
    runState.demos = normalizeDemoDefinitions(demosPayload.demos);
    if (runState.demos.length === 0) {
      runState.demos = fallbackDemos();
    }
    renderDemoSelectOptions(runState.demos);
    if (statusPayload && typeof statusPayload === "object") {
      runState.status = statusPayload;
      dataState.runStatus = statusPayload;
      setRunStateText(statusPayload);
      const queryRun = new URLSearchParams(window.location.search).get("run");
      if (!queryRun && statusPayload.run_dir) {
        applyRunDirectory(statusPayload.run_dir, { updateUrl: false });
      }
    } else {
      setRunStateText({ state: "api-ready" });
    }
  } finally {
    runState.apiBootstrapInFlight = false;
  }
}

async function refreshDashboardRunStatus() {
  if (!runState.apiAvailable) {
    if (dataConfig.useApi) {
      await refreshDashboardApiBootstrap();
    }
    return;
  }
  const statusPayload = await apiJson("/api/run/status");
  if (!statusPayload) {
    runState.apiAvailable = false;
    setRunStateText({ state: "api-unavailable" });
    setRunControlsEnabled(false);
    return;
  }
  runState.status = statusPayload;
  dataState.runStatus = statusPayload;
  setRunStateText(statusPayload);
  const queryRun = new URLSearchParams(window.location.search).get("run");
  if (!queryRun && statusPayload.run_dir) {
    applyRunDirectory(statusPayload.run_dir, { updateUrl: false });
  }
}

async function onStartRunClick() {
  if (!runState.apiAvailable && dataConfig.useApi) {
    await refreshDashboardApiBootstrap();
  }
  if (!runState.apiAvailable) {
    setRunStateText({ state: dataConfig.useApi ? "api-unavailable" : "api-disabled" });
    return;
  }
  const spec = buildRunSpecFromControls();
  const payload = await apiJson("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
  if (!payload) {
    setRunStateText({ state: "start-failed" });
    return;
  }
  if (payload.run_dir) {
    applyRunDirectory(payload.run_dir, { updateUrl: true });
  }
  await refreshDashboardRunStatus();
  await refreshData();
}

async function onStopRunClick() {
  if (!runState.apiAvailable && dataConfig.useApi) {
    await refreshDashboardApiBootstrap();
  }
  if (!runState.apiAvailable) {
    setRunStateText({ state: dataConfig.useApi ? "api-unavailable" : "api-disabled" });
    return;
  }
  await apiJson("/api/run/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  await refreshDashboardRunStatus();
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

  const { matrix, counts } = buildWeightMatrix(edges, 0, 0, maxDim);
  drawHeatmapMatrix(canvases.heatmapInput, matrix, clampMin, clampMax, counts);

  const secondary =
    uiControls.weightProjection2?.value ||
    projections.find((proj) => proj.toLowerCase().includes("hidden->output")) ||
    projections.find((proj) => proj !== selectedProj) ||
    projections[0];
  if (uiControls.weightProjectionLabel2) {
    uiControls.weightProjectionLabel2.textContent = secondary || "--";
  }
  const sEdges = dataState.weightsIndex[secondary]?.byStep[selectedStep] || [];
  const { matrix: sMatrix, counts: sCounts } = buildWeightMatrix(sEdges, 0, 0, maxDim);
  drawHeatmapMatrix(canvases.heatmapOutput, sMatrix, clampMin, clampMax, sCounts);
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

function computeLogicSpikeRatesFromTrials() {
  if (!dataState.trialsRows || dataState.trialsRows.length === 0) return null;
  const windowTrials = Math.max(20, Number(uiControls.rasterWindow?.value || 200));
  const trials = dataState.trialsRows.slice(-windowTrials);
  if (trials.length === 0) return null;

  let out0Total = 0;
  let out1Total = 0;
  let hiddenRateTotal = 0;
  trials.forEach((row) => {
    out0Total += Number(row.out_spikes_0 || 0);
    out1Total += Number(row.out_spikes_1 || 0);
    hiddenRateTotal += Number(row.hidden_mean_spikes || 0);
  });

  const simSteps = Math.max(
    1,
    Number(dataState.runConfig?.logic_sim_steps_per_trial || dataState.runConfig?.sim_steps_per_trial || 1)
  );
  const denom = trials.length * simSteps;
  return {
    labels: ["output_0", "output_1", "hidden_mean"],
    rates: [out0Total / denom, out1Total / denom, hiddenRateTotal / trials.length],
  };
}

function computePerPopSpikeRates() {
  if ((!dataState.spikesRows || dataState.spikesRows.length === 0) && dataState.trialsRows?.length) {
    return computeLogicSpikeRatesFromTrials();
  }
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

function drawSpikeRatesPanel() {
  const rates = computePerPopSpikeRates();
  if (rates && rates.rates.length) {
    drawBarsWithLabels(canvases.rate, rates.labels, rates.rates, theme.accent);
    return;
  }
  const ctx = setupCanvas(canvases.rate);
  if (!ctx) return;
  const { width, height } = canvases.rate.getBoundingClientRect();
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = theme.muted;
  ctx.fillText("Spike rates unavailable", 12, 20);
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

  drawSpikeRatesPanel();

  const weightSamples = extractWeightSamples();
  drawHistogram(canvases.weight, weightSamples);
  drawStateSpace();

  requestAnimationFrame(tick);
}

try {
  runState.demos = fallbackDemos();
  renderDemoSelectOptions(runState.demos);
  setRunStateText({ state: "initializing" });
  renderTaskPanel();
} catch (error) {
  console.error("Failed to pre-populate demo list", error);
  setRunStateText({ state: "ui-error" });
}

window.addEventListener("resize", () => {
  resizeAll();
  renderAuxPanels();
});
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
if (uiControls.edgeOpacityByDistance) {
  uiControls.edgeOpacityByDistance.addEventListener("change", () => {
    updateLegendNotes();
  });
}
if (uiControls.showDelayTooltip) {
  uiControls.showDelayTooltip.addEventListener("change", () => {
    updateLegendNotes();
  });
}
if (canvases.network) {
  canvases.network.addEventListener("mousemove", onNetworkHover);
  canvases.network.addEventListener("mouseleave", hideNetworkTooltip);
}
resizeAll();
syncNeuronViewControls();
updateNeuronControlsEnabled();
updateNeuronViewInfo();
updateLegendNotes();
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
if (uiControls.weightProjection2) {
  uiControls.weightProjection2.addEventListener("change", () => {});
}
if (uiControls.weightStep) {
  uiControls.weightStep.addEventListener("change", () => {});
}
if (runNodes.demoSelect) {
  runNodes.demoSelect.addEventListener("change", () => {
    const defaults = selectedDemoDefaults();
    if (defaults) {
      applyRunSpecToControls(defaults);
    }
  });
}
if (runNodes.startButton) {
  runNodes.startButton.addEventListener("click", () => {
    onStartRunClick();
  });
}
if (runNodes.stopButton) {
  runNodes.stopButton.addEventListener("click", () => {
    onStopRunClick();
  });
}
if (auxNodes.modulatorFieldSelect) {
  auxNodes.modulatorFieldSelect.addEventListener("change", () => {
    drawModulatorPanel();
  });
}
if (auxNodes.modulatorKindSelect) {
  auxNodes.modulatorKindSelect.addEventListener("change", () => {
    drawModulatorPanel();
  });
}
if (auxNodes.receptorProjectionSelect) {
  auxNodes.receptorProjectionSelect.addEventListener("change", () => {
    drawReceptorPanel();
  });
}
if (auxNodes.receptorMetricSelect) {
  auxNodes.receptorMetricSelect.addEventListener("change", () => {
    drawReceptorPanel();
  });
}

async function bootstrapDashboard() {
  await refreshDashboardApiBootstrap();
  await refreshData();
  setInterval(async () => {
    await refreshDashboardRunStatus();
    await refreshData();
  }, dataConfig.refreshMs);
  requestAnimationFrame(tick);
}

bootstrapDashboard().catch((error) => {
  console.error("Dashboard bootstrap failed", error);
  setRunStateText({ state: "ui-error" });
  setRunControlsEnabled(false);
});
