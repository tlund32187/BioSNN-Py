# BioSNN-Py

**Library-first scaffold** for a biologically inspired spiking “liquid” brain simulation toolkit.

- Repo name: `BioSNN-Py`
- Import package: `biosnn`
- Stable façade (public API): `biosnn.api`
- Internals: everything else under `biosnn.*` (not re-exported from `biosnn.api`)

## Goals (high level)
- SOLID-friendly modular architecture (contracts + implementations)
- Pluggable neuron models (GLIF, AdEx, multi-compartment, etc.)
- Synapses/receptors (AMPA/NMDA/GABA…), learning (STDP, triplet, 3-factor, metaplasticity, homeostasis)
- Neuromodulators (DA/ACh/NA), diffusion fields, astrocytes (tripartite synapse), energy accounting
- 3D spatial wiring, axonal delays (distance + myelination), pruning/growth/rewiring
- Online learning (eligibility traces / e-prop), multi-factor hooks
- Monitoring: CSV/console/graph outputs + rich debug hooks
- Backends: CPU + CUDA (e.g., PyTorch) with reproducibility controls

This scaffold intentionally contains **no domain implementation** yet—only structure, tooling, docs, and a minimal import test.

## Quickstart (Windows + VS Code)

1) Create and activate a virtual environment:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Optional: auto-detect CUDA and install the correct torch build:
```powershell
.\scripts\install_torch.ps1 -Dev
```

2) Install the library in editable mode with dev tools:
```powershell
python -m pip install -U pip
pip install -e ".[dev,torch]"
```

### Torch (CPU or CUDA)
Install the optional torch extra (CPU by default):
```powershell
pip install -e ".[torch]"
```

This extra also installs `numpy` to avoid optional torch warnings.

For CUDA-enabled wheels (example: CUDA 13.0 / cu130), use:
```powershell
pip install -e ".[dev,torch-cu130]" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

For CPU-only wheels, use:
```powershell
pip install -e ".[dev,torch-cpu]"
```

If you already have a CUDA-enabled torch in another Python, re-installing inside your `.venv` is required.

### Benchmark (optional)
```powershell
python scripts/bench_step.py --device cuda --steps 5000
```
Scaling benchmark (compile time + steps/sec + monitor overhead):
```powershell
python scripts/bench_scale.py --device all --n-pre 2000 --n-post 2000 --p 0.01
```

## Run modes (dashboard vs fast)
The CLI supports two run modes that trade off artifacts vs throughput:
- `dashboard` (default): full CSV artifacts for the dashboard (neuron/synapse/spikes/weights), best for visualization.
- `fast`: throughput-oriented, uses `fast_mode=True` and `compiled_mode=True`, minimal monitors (metrics only), and skips launching the dashboard server.

Use:
```powershell
python -m biosnn.runners.cli --demo network --mode fast
```

### Fast mode monitor compatibility
`fast_mode=True` does not support monitors that require merged tensors or global spikes:
- `NeuronCSVMonitor`
- `SynapseCSVMonitor`
- `SpikeEventsCSVMonitor`

`MetricsCSVMonitor` is compatible, as are custom monitors that only consume per-population tensors.

### Monitor safety rails
By default the demos enable safe defaults to avoid huge CSV output on large networks:
- `--monitor-safe-defaults` (enabled by default)
- `--monitor-neuron-sample` (default 512)
- `--monitor-edge-sample` (default 20000)

You can disable or override these with the CLI flags above.

### CUDA-clean demos
When `--device cuda` is selected (or CUDA is auto-detected), the demo defaults to the CUDA-friendly
`DelayedSparseMatmulSynapse`. `DelayedCurrentSynapse` is kept as a CPU reference path.

### Profiling (optional)
Use `--profile` to capture a short `torch.profiler` trace after the demo run:
```powershell
python -m biosnn.runners.cli --demo network --device cuda --profile --profile-steps 50
```
The trace is written to `profile.json` in the run directory.

### Fused delay buckets (sparse backend)
When compiling sparse delay mats, non-empty delay buckets are fused by default:
- Buckets are stacked into a single sparse matrix per compartment.
- Runtime performs one sparse matmul and a vectorized scatter into the delay ring buffer.

This removes the Python loop over delays and is the intended high-performance path. The legacy per-delay
loop remains as a fallback if fused artifacts are absent.

3) Run checks:
```powershell
ruff check .
mypy .
pytest
```
If `ruff` or `mypy` are missing, install the dev extra:
```powershell
pip install -e ".[dev]"
```

## Public API policy
- Only symbols re-exported from `biosnn.api` are considered public and semver-stable.
- Everything else may change freely.

See: `docs/public_api.md` and `docs/architecture/decisions/0001-library-first.md`
and `docs/architecture/decisions/0002-run-modes-and-fused-delays.md`

## Model docs
- `docs/models/adex_2c.md`
- `docs/models/glif.md`

## Guides
- `docs/guides/synapses.md`
- `docs/guides/network_builder.md`
- `docs/guides/presets.md`
- `docs/guides/adding_neuron_models.md`

## Dashboard (local)
The synapse dashboard is a static page under `docs/dashboard/`.

1) Start a simple web server from the repo root:
```powershell
python -m http.server 8000
```

2) Open the dashboard:
```
http://localhost:8000/docs/dashboard/
```

### Live data
By default the dashboard reads:
- `docs/dashboard/data/neuron.csv`
- `docs/dashboard/data/synapse.csv`
- `docs/dashboard/data/topology.json` (optional)

You can override with query params:
```
http://localhost:8000/docs/dashboard/?neuron=PATH&synapse=PATH&topology=PATH
```
