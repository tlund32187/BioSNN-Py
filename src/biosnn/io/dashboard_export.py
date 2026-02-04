"""Dashboard export helpers (CSV + topology JSON)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from biosnn.contracts.monitors import StepEvent
from biosnn.contracts.neurons import INeuronModel, NeuronStepResult
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.io.graph.population_topology_json import build_population_topology_payload
from biosnn.io.graph.topology_json import build_topology_payload
from biosnn.monitors.csv import NeuronCSVMonitor, SynapseCSVMonitor


def export_topology_json(
    topology: SynapseTopology,
    *,
    path: str | Path | None = None,
    weights: Tensor | None = None,
    pre_layer: int = 0,
    post_layer: int = 2,
) -> dict[str, Any]:
    """Export a topology JSON payload for the dashboard.

    Returns a dict with "nodes" and "edges". If path is provided, writes JSON to disk.
    """

    payload = build_topology_payload(
        topology,
        weights=weights,
        pre_layer=pre_layer,
        post_layer=post_layer,
    )

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload




def export_population_topology_json(
    populations: Sequence[Any],
    projections: Sequence[Any],
    *,
    path: str | Path | None = None,
    weights_by_projection: Mapping[str, Tensor] | None = None,
    include_neuron_topology: bool = False,
) -> dict[str, Any]:
    """Export a population-level topology JSON payload for the dashboard."""

    payload = build_population_topology_payload(
        populations,
        projections,
        weights_by_projection=weights_by_projection,
        include_neuron_topology=include_neuron_topology,
    )

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def export_synapse_csv(
    weights: Tensor,
    *,
    path: str | Path,
    stats: Sequence[str] | None = None,
    sample_indices: Sequence[int] | None = None,
    scalars: Mapping[str, float] | None = None,
) -> Path:
    """Write a synapse CSV snapshot for the dashboard."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_indices = _clamp_indices(weights, sample_indices)
    monitor = SynapseCSVMonitor(path, stats=stats, sample_indices=safe_indices)
    event = StepEvent(t=0.0, dt=0.0, tensors={"weights": weights}, scalars=scalars)
    monitor.on_step(event)
    monitor.close()
    return path


def export_neuron_csv(
    tensors: Mapping[str, Tensor],
    *,
    path: str | Path,
    spikes: Tensor | None = None,
    stats: Sequence[str] | None = None,
    sample_indices: Sequence[int] | None = None,
    scalars: Mapping[str, float] | None = None,
    t: float = 0.0,
    dt: float = 0.0,
) -> Path:
    """Write a neuron CSV snapshot for the dashboard."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_indices = _clamp_indices(_first_tensor(tensors), sample_indices)
    monitor = NeuronCSVMonitor(
        path,
        tensor_keys=tuple(tensors.keys()) if tensors else None,
        stats=stats,
        include_spikes=spikes is not None,
        sample_indices=safe_indices,
    )
    event = StepEvent(t=t, dt=dt, spikes=spikes, tensors=tensors, scalars=scalars)
    monitor.on_step(event)
    monitor.close()
    return path


def export_neuron_snapshot(
    model: INeuronModel,
    state: Any,
    result: NeuronStepResult,
    *,
    path: str | Path,
    stats: Sequence[str] | None = None,
    sample_indices: Sequence[int] | None = None,
    scalars: Mapping[str, float] | None = None,
    t: float = 0.0,
    dt: float = 0.0,
) -> Path:
    """Write a neuron CSV snapshot from a model + step result."""

    tensors = model.state_tensors(state)
    return export_neuron_csv(
        tensors,
        path=path,
        spikes=result.spikes,
        stats=stats,
        sample_indices=sample_indices,
        scalars=scalars,
        t=t,
        dt=dt,
    )




def export_dashboard_bundle(
    *,
    out_dir: Path,
    topology_payload: dict | None = None,
    neuron_csv_src: Path | None = None,
    synapse_csv_src: Path | None = None,
    spikes_csv_src: Path | None = None,
    metrics_csv_src: Path | None = None,
    weights_csv_src: Path | None = None,
) -> None:
    """Bundle dashboard artifacts into a single folder."""

    out_dir.mkdir(parents=True, exist_ok=True)

    if topology_payload is not None:
        (out_dir / "topology.json").write_text(json.dumps(topology_payload, indent=2), encoding="utf-8")

    _copy_if_present(neuron_csv_src, out_dir / "neuron.csv")
    _copy_if_present(synapse_csv_src, out_dir / "synapse.csv")
    _copy_if_present(spikes_csv_src, out_dir / "spikes.csv")
    _copy_if_present(metrics_csv_src, out_dir / "metrics.csv")
    _copy_if_present(weights_csv_src, out_dir / "weights.csv")


def export_dashboard_snapshot(
    topology: SynapseTopology,
    weights: Tensor,
    *,
    out_dir: str | Path,
    topology_name: str = "topology.json",
    synapse_name: str = "synapse.csv",
    neuron_name: str = "neuron.csv",
    neuron_tensors: Mapping[str, Tensor] | None = None,
    neuron_spikes: Tensor | None = None,
    neuron_stats: Sequence[str] | None = None,
    neuron_samples: int | None = 32,
    synapse_stats: Sequence[str] | None = None,
    synapse_samples: int | None = 64,
    scalars: Mapping[str, float] | None = None,
    neuron_scalars: Mapping[str, float] | None = None,
) -> dict[str, Path]:
    """Export topology JSON + synapse CSV into a dashboard data folder."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topology_path = out_dir / topology_name
    synapse_path = out_dir / synapse_name
    neuron_path = out_dir / neuron_name

    export_topology_json(topology, weights=weights, path=topology_path)
    sample_indices = list(range(synapse_samples)) if synapse_samples else None
    export_synapse_csv(
        weights,
        path=synapse_path,
        stats=synapse_stats,
        sample_indices=sample_indices,
        scalars=scalars,
    )

    paths: dict[str, Path] = {"topology": topology_path, "synapse": synapse_path}
    if neuron_tensors is not None:
        neuron_indices = list(range(neuron_samples)) if neuron_samples else None
        export_neuron_csv(
            neuron_tensors,
            path=neuron_path,
            spikes=neuron_spikes,
            stats=neuron_stats,
            sample_indices=neuron_indices,
            scalars=neuron_scalars,
        )
        paths["neuron"] = neuron_path

    return paths


def _copy_if_present(src: Path | None, dest: Path) -> None:
    if src is None:
        return
    if not src.exists():
        return
    dest.write_bytes(src.read_bytes())


def _clamp_indices(values: Tensor | None, indices: Sequence[int] | None) -> list[int] | None:
    if indices is None:
        return None
    if values is None:
        return list(indices)
    length = _tensor_length(values)
    return [idx for idx in indices if idx < length]


def _tensor_length(values: Tensor) -> int:
    if hasattr(values, "numel"):
        return int(values.numel())
    try:
        return len(values)
    except TypeError:
        return 0


def _first_tensor(tensors: Mapping[str, Tensor]) -> Tensor | None:
    for value in tensors.values():
        return value
    return None
