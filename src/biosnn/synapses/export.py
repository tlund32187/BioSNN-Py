"""Topology export helpers for visualization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from biosnn.contracts.monitors import StepEvent
from biosnn.contracts.neurons import INeuronModel, NeuronStepResult
from biosnn.contracts.synapses import ReceptorKind, SynapseTopology
from biosnn.contracts.tensor import Tensor
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

    node_payload, pre_offset, post_offset = _build_nodes(topology, pre_layer, post_layer)
    edge_payload = _build_edges(topology, weights, pre_offset, post_offset)
    payload = {"nodes": node_payload, "edges": edge_payload}

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


def _build_nodes(
    topology: SynapseTopology,
    pre_layer: int,
    post_layer: int,
) -> tuple[list[dict[str, float]], int, int]:
    pre_pos = _to_list(topology.pre_pos)
    post_pos = _to_list(topology.post_pos)

    n_pre = len(pre_pos) if pre_pos is not None else _infer_n(topology.pre_idx)
    n_post = len(post_pos) if post_pos is not None else _infer_n(topology.post_idx)

    if pre_pos is None:
        pre_pos = _layout_column(n_pre, x=0.2)
    if post_pos is None:
        post_pos = _layout_column(n_post, x=0.8)

    normalized = _normalize_positions(pre_pos + post_pos)
    pre_norm = normalized[:n_pre]
    post_norm = normalized[n_pre:]

    nodes: list[dict[str, float]] = []
    for idx, (x, y) in enumerate(pre_norm):
        nodes.append({"x": x, "y": y, "layer": pre_layer, "index": idx})
    for idx, (x, y) in enumerate(post_norm):
        nodes.append({"x": x, "y": y, "layer": post_layer, "index": n_pre + idx})

    return nodes, 0, n_pre


def _build_edges(
    topology: SynapseTopology,
    weights: Tensor | None,
    pre_offset: int,
    post_offset: int,
) -> list[dict[str, Any]]:
    pre_idx = _to_int_list(topology.pre_idx)
    post_idx = _to_int_list(topology.post_idx)
    if len(pre_idx) != len(post_idx):
        raise ValueError("pre_idx and post_idx must be the same length")

    weight_source = weights if weights is not None else topology.weights
    weight_list = _to_float_list(weight_source)
    if weight_list is None:
        weight_list = [0.0] * len(pre_idx)

    receptor_names = _receptor_names(topology)

    edges: list[dict[str, Any]] = []
    for i, (pre, post) in enumerate(zip(pre_idx, post_idx, strict=True)):
        edge = {
            "from": pre + pre_offset,
            "to": post + post_offset,
            "weight": weight_list[i] if i < len(weight_list) else 0.0,
        }
        if receptor_names is not None:
            edge["receptor"] = receptor_names[i] if i < len(receptor_names) else "unknown"
        edges.append(edge)

    return edges


def _receptor_names(topology: SynapseTopology) -> list[str] | None:
    if topology.receptor is None:
        return None
    receptor_ids = _to_int_list(topology.receptor)
    kinds = topology.receptor_kinds or (ReceptorKind.AMPA, ReceptorKind.NMDA, ReceptorKind.GABA)
    names = [kind.value for kind in kinds]
    resolved = []
    for idx in receptor_ids:
        if idx < 0 or idx >= len(names):
            resolved.append("unknown")
        else:
            resolved.append(names[idx])
    return resolved


def _layout_column(count: int, *, x: float) -> list[list[float]]:
    if count <= 0:
        return []
    return [[x, (i + 1) / (count + 1)] for i in range(count)]


def _normalize_positions(points: Sequence[Sequence[float]]) -> list[list[float]]:
    if not points:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max(max_x - min_x, 1e-9)
    range_y = max(max_y - min_y, 1e-9)
    return [[(p[0] - min_x) / range_x, (p[1] - min_y) / range_y] for p in points]


def _infer_n(tensor: Tensor) -> int:
    values = _to_int_list(tensor)
    if not values:
        return 0
    return max(values) + 1


def _to_list(tensor: Tensor | None) -> list[list[float]] | None:
    if tensor is None:
        return None
    data = tensor
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "tolist"):
        data = data.tolist()
    return [[float(pair[0]), float(pair[1])] for pair in data]


def _to_int_list(tensor: Tensor | None) -> list[int]:
    if tensor is None:
        return []
    data = tensor
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "tolist"):
        data = data.tolist()
    return [int(value) for value in data]


def _to_float_list(tensor: Tensor | None) -> list[float] | None:
    if tensor is None:
        return None
    data = tensor
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "tolist"):
        data = data.tolist()
    return [float(value) for value in data]
