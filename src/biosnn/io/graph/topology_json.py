"""Topology graph payload builder for dashboard export."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from biosnn.contracts.synapses import ReceptorKind, SynapseTopology
from biosnn.contracts.tensor import Tensor


def build_topology_payload(
    topology: SynapseTopology,
    *,
    weights: Tensor | None = None,
    pre_layer: int = 0,
    post_layer: int = 2,
) -> dict[str, Any]:
    """Build a topology payload with nodes and edges for visualization."""

    node_payload, pre_offset, post_offset = _build_nodes(topology, pre_layer, post_layer)
    edge_payload = _build_edges(topology, weights, pre_offset, post_offset)
    return {"nodes": node_payload, "edges": edge_payload}


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


__all__ = ["build_topology_payload"]
