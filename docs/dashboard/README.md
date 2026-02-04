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
Use the public API to run a demo network and point the dashboard at its
artifact folder (query params). The CLI runner already writes the required
CSV files into `artifacts/run_*`.

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
