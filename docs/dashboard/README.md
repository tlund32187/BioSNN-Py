# Synapse Dashboard

This dashboard reads CSV monitor outputs (neurons + synapses) and renders a live
view of connections and spiking activity.


## CLI demo run
```powershell
python -m biosnn.runners.cli --steps 500 --dt 0.001
```
Optional flags:
- `--no-open`
- `--no-server`

## Quick start
1) Start a static file server from the repo root:
```powershell
python -m http.server 8000
```

2) Open the dashboard:
```
http://localhost:8000/docs/dashboard/
```

## Live data files
By default the dashboard looks for:
- `docs/dashboard/data/neuron.csv`
- `docs/dashboard/data/synapse.csv`
- `docs/dashboard/data/spikes.csv`
- `docs/dashboard/data/metrics.csv`
- `docs/dashboard/data/weights.csv`
- `docs/dashboard/data/topology.json` (optional)

You can override with query parameters:
```
http://localhost:8000/docs/dashboard/?neuron=PATH&synapse=PATH&spikes=PATH&metrics=PATH&weights=PATH&topology=PATH
```

## How to generate CSVs
Use the CSV monitors from BioSNN and write into the dashboard data folder.

```python
from pathlib import Path

from biosnn.monitors.csv import NeuronCSVMonitor, SynapseCSVMonitor

out_dir = Path("docs/dashboard/data")
out_dir.mkdir(parents=True, exist_ok=True)

neuron_monitor = NeuronCSVMonitor(
    out_dir / "neuron.csv",
    tensor_keys=("v_soma", "refrac_left", "spike_hold_left", "theta"),
    sample_indices=list(range(32)),
    include_spikes=True,
)

synapse_monitor = SynapseCSVMonitor(
    out_dir / "synapse.csv",
    sample_indices=list(range(64)),
)
```

## Export topology + synapse snapshot
Use the helper to export the topology JSON and a synapse CSV snapshot together.

```python
from biosnn.io.dashboard_export import export_dashboard_snapshot

export_dashboard_snapshot(
    topology,
    weights=state.weights,
    out_dir="docs/dashboard/data",
    neuron_tensors={
        "v_soma": neuron_state.v_soma,
        "refrac_left": neuron_state.refrac_left,
    },
    neuron_spikes=step_result.spikes,
)
```

## Export neuron snapshot from a model step
If you already have the model + state + step result, you can snapshot a neuron CSV in one call.

```python
from biosnn.io.dashboard_export import export_neuron_snapshot

export_neuron_snapshot(
    model,
    state,
    step_result,
    path="docs/dashboard/data/neuron.csv",
    sample_indices=list(range(32)),
)
```

## Topology JSON (optional)
Provide a topology file to render your true graph layout.

```json
{
  "nodes": [
    {"x": 0.1, "y": 0.2, "layer": 0},
    {"x": 0.5, "y": 0.5, "layer": 1}
  ],
  "edges": [
    {"from": 0, "to": 1, "weight": 0.6, "receptor": "ampa"}
  ]
}
```

Notes:
- `x`/`y` are normalized (0..1). `layer` is 0=input, 1=hidden, 2=output.
- `weight` is used for edge color intensity (positive = excitatory, negative = inhibitory).

Population view:
- If topology.json contains {"mode": "population"}, the dashboard defaults to the population view.
- Use the View toggle to switch between neurons and populations.

## New dashboard artifacts
- Spike raster: `spikes.csv` via `SpikeEventsCSVMonitor`
- Metrics time-series: `metrics.csv` via `MetricsCSVMonitor`
- Projection weights: `weights.csv` via `ProjectionWeightsCSVMonitor`

Runner flags:
- `--no-open`: do not open the browser
- `--no-server`: run simulation and write artifacts only (no HTTP server)
