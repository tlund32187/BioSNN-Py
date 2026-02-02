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

2) Install the library in editable mode with dev tools:
```powershell
python -m pip install -U pip
pip install -e ".[dev]"
```

3) Run checks:
```powershell
ruff check .
mypy .
pytest
```

## Public API policy
- Only symbols re-exported from `biosnn.api` are considered public and semver-stable.
- Everything else may change freely.

See: `docs/public_api.md` and `docs/architecture/decisions/0001-library-first.md`
