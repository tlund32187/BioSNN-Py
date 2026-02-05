"""Population-level topology payload builder for dashboard export."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import ReceptorKind, SynapseTopology
from biosnn.contracts.tensor import Tensor


def build_population_topology_payload(
    populations: Sequence[Mapping[str, Any]] | Sequence[Any],
    projections: Sequence[Mapping[str, Any]] | Sequence[Any],
    *,
    weights_by_projection: Mapping[str, Tensor] | None = None,
    layout: str = "layers",
    include_neuron_topology: bool = False,
) -> dict[str, Any]:
    """Build a population-level topology payload for visualization."""

    _ = layout
    pop_specs = list(populations)
    proj_specs = list(projections)

    nodes = _build_population_nodes(pop_specs)
    edges = _build_population_edges(proj_specs, weights_by_projection)
    payload = {
        "mode": "population",
        "nodes": nodes,
        "edges": edges,
        "meta": {"created_by": "biosnn", "version": 1},
    }
    if include_neuron_topology:
        neuron_nodes, neuron_edges = _build_neuron_topology(pop_specs, proj_specs, nodes)
        payload["neuron_nodes"] = neuron_nodes
        payload["neuron_edges"] = neuron_edges
    return payload


def _build_neuron_topology(
    pop_specs: Sequence[Any],
    proj_specs: Sequence[Any],
    pop_nodes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    offsets: dict[str, int] = {}
    total = 0
    for pop, node in zip(pop_specs, pop_nodes, strict=True):
        offsets[_pop_name(pop)] = total
        total += int(node.get("n_neurons") or 0)

    neuron_nodes: list[dict[str, Any]] = []
    for pop, node in zip(pop_specs, pop_nodes, strict=True):
        n = int(node.get("n_neurons") or 0)
        if n <= 0:
            continue
        pop_name = _pop_name(pop)
        base_x = float(node.get("x") or 0.5)
        base_y = float(node.get("y") or 0.5)
        spread = min(0.8, 0.6 / max(n, 1))
        pos_list = _pop_positions(pop)
        for i in range(n):
            y_offset = 0.0 if n == 1 else (i / (n - 1) - 0.5) * spread
            x_jitter = ((_hash_int(i + offsets[pop_name]) % 1000) / 1000 - 0.5) * 0.05
            pos = _neuron_pos(pos_list, i, base_x, base_y)
            neuron_nodes.append(
                {
                    "index": offsets[pop_name] + i,
                    "pop": pop_name,
                    "local_idx": i,
                    "x": _clamp01(base_x + x_jitter),
                    "y": _clamp01(base_y + y_offset),
                    "layer": int(node.get("layer") or 0),
                    "pos": pos,
                }
            )

    neuron_edges: list[dict[str, Any]] = []
    for proj in proj_specs:
        topo = _proj_topology(proj)
        pre = _proj_pre(proj)
        post = _proj_post(proj)
        pre_offset = offsets.get(pre, 0)
        post_offset = offsets.get(post, 0)
        pre_idx = _to_int_list(topo.pre_idx)
        post_idx = _to_int_list(topo.post_idx)
        weights = _to_float_list(topo.weights) or [0.0] * len(pre_idx)
        delay_steps = _to_int_list(topo.delay_steps)
        receptor_names = _receptor_names(topo)
        for idx, (src, dst) in enumerate(zip(pre_idx, post_idx, strict=True)):
            edge: dict[str, Any] = {
                "from": pre_offset + src,
                "to": post_offset + dst,
                "weight": weights[idx] if idx < len(weights) else 0.0,
            }
            if delay_steps:
                edge["delay_steps"] = delay_steps[idx] if idx < len(delay_steps) else 0
            if receptor_names is not None:
                edge["receptor"] = receptor_names[idx] if idx < len(receptor_names) else "unknown"
            neuron_edges.append(edge)

    return neuron_nodes, neuron_edges


def _hash_int(value: int) -> int:
    value ^= (value << 13) & 0xFFFFFFFF
    value ^= (value >> 17) & 0xFFFFFFFF
    value ^= (value << 5) & 0xFFFFFFFF
    return value & 0xFFFFFFFF


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _build_population_nodes(pop_specs: Sequence[Any]) -> list[dict[str, Any]]:
    raw_layers: list[int | None] = []
    for idx, pop in enumerate(pop_specs):
        layer = _pop_layer(pop)
        if layer is None:
            layer = _infer_layer_from_name(_pop_name(pop), idx)
        raw_layers.append(layer)

    max_layer = max((layer for layer in raw_layers if layer is not None), default=-1)
    next_layer = max_layer + 1
    layers: list[int] = []
    for layer in raw_layers:
        if layer is None:
            layers.append(next_layer)
            next_layer += 1
        else:
            layers.append(layer)

    layer_values = sorted(set(layers))
    layer_to_col = {layer: idx for idx, layer in enumerate(layer_values)}
    layer_counts: dict[int, int] = Counter(layer_to_col[layer] for layer in layers)
    layer_indices: dict[int, int] = {layer: 0 for layer in layer_values}
    total_layers = max(len(layer_values), 1)

    nodes: list[dict[str, Any]] = []
    for idx, pop in enumerate(pop_specs):
        layer = layers[idx]
        col = layer_to_col[layer]
        count = layer_counts[col]
        pos_index = layer_indices[layer]
        layer_indices[layer] += 1

        x = (col + 1) / (total_layers + 1)
        y = (pos_index + 1) / (count + 1)

        node_payload: dict[str, Any] = {
            "id": _pop_name(pop),
            "label": _pop_label(pop),
            "layer": int(layer),
            "n_neurons": _pop_count(pop),
            "model": _pop_model(pop),
            "x": float(x),
            "y": float(y),
        }
        frame_payload = _frame_payload(_pop_frame(pop))
        if frame_payload is not None:
            node_payload["frame"] = frame_payload
        role = _pop_meta_value(pop, "role")
        group = _pop_meta_value(pop, "group")
        if role is not None:
            node_payload["role"] = role
        if group is not None:
            node_payload["group"] = group
        nodes.append(node_payload)

    return nodes


def _build_population_edges(
    proj_specs: Sequence[Any],
    weights_by_projection: Mapping[str, Tensor] | None,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []

    for proj in proj_specs:
        topo = _proj_topology(proj)
        edge_count = len(_to_int_list(topo.pre_idx))
        weights = None
        proj_name = _proj_name(proj)
        if weights_by_projection and proj_name in weights_by_projection:
            weights = weights_by_projection[proj_name]
        elif topo.weights is not None:
            weights = topo.weights

        weight_values = _to_float_list(weights) or []
        mean_weight, std_weight = _mean_std(weight_values)

        delay_values = _to_float_list(topo.delay_steps) or []
        mean_delay = float(sum(delay_values) / len(delay_values)) if delay_values else None
        min_delay = float(min(delay_values)) if delay_values else None
        max_delay = float(max(delay_values)) if delay_values else None

        dist_values = _to_float_list(getattr(topo, "edge_dist", None)) or []
        mean_dist = float(sum(dist_values) / len(dist_values)) if dist_values else None
        corr_weight_dist = (
            _corr_sample(weight_values, dist_values, max_samples=2000)
            if weight_values and dist_values
            else None
        )

        receptor_counts = _receptor_counts(topo)
        target_counts = _target_counts(topo)

        edge_payload = {
            "id": proj_name,
            "from": _proj_pre(proj),
            "to": _proj_post(proj),
            "n_synapses": int(edge_count),
            "mean_weight": float(mean_weight),
            "std_weight": float(std_weight),
        }
        if mean_delay is not None:
            edge_payload["mean_delay_steps"] = float(mean_delay)
        if min_delay is not None:
            edge_payload["min_delay_steps"] = float(min_delay)
        if max_delay is not None:
            edge_payload["max_delay_steps"] = float(max_delay)
        if mean_dist is not None:
            edge_payload["mean_edge_dist"] = float(mean_dist)
        if corr_weight_dist is not None:
            edge_payload["corr_weight_dist"] = float(corr_weight_dist)
        if receptor_counts:
            edge_payload["receptor_counts"] = receptor_counts
        if target_counts:
            edge_payload["target_counts"] = target_counts

        edges.append(edge_payload)

    return edges


def _pop_name(pop: Any) -> str:
    if hasattr(pop, "name"):
        return str(pop.name)
    if isinstance(pop, Mapping):
        return str(pop.get("name") or pop.get("id") or pop.get("label") or "population")
    return "population"


def _pop_label(pop: Any) -> str:
    if isinstance(pop, Mapping):
        label = pop.get("label")
        if label is not None:
            return str(label)
    return _pop_name(pop)


def _pop_count(pop: Any) -> int:
    if hasattr(pop, "n"):
        return int(pop.n)
    if isinstance(pop, Mapping):
        return int(pop.get("n") or pop.get("n_neurons") or 0)
    return 0


def _pop_model(pop: Any) -> str:
    if hasattr(pop, "model"):
        model = pop.model
        if hasattr(model, "name"):
            return str(model.name)
        return cast(str, model.__class__.__name__.lower())
    if isinstance(pop, Mapping):
        name = pop.get("model_name") or pop.get("model") or pop.get("modelType")
        if name is not None:
            return str(name)
    return "unknown"


def _pop_layer(pop: Any) -> int | None:
    if hasattr(pop, "layer"):
        return int(pop.layer)
    if hasattr(pop, "meta") and pop.meta and isinstance(pop.meta, Mapping) and "layer" in pop.meta:
        return int(pop.meta["layer"])
    if isinstance(pop, Mapping):
        if "layer" in pop:
            return int(pop["layer"])
        meta = pop.get("meta")
        if isinstance(meta, Mapping) and "layer" in meta:
            return int(meta["layer"])
    return None


def _pop_meta_value(pop: Any, key: str) -> Any:
    if hasattr(pop, "meta") and pop.meta and isinstance(pop.meta, Mapping) and key in pop.meta:
        return pop.meta[key]
    if isinstance(pop, Mapping):
        meta = pop.get("meta")
        if isinstance(meta, Mapping) and key in meta:
            return meta[key]
    return None


def _pop_frame(pop: Any) -> Any | None:
    if hasattr(pop, "frame"):
        return pop.frame
    if isinstance(pop, Mapping):
        frame = pop.get("frame")
        if frame is not None:
            return frame
        meta = pop.get("meta")
        if isinstance(meta, Mapping):
            return meta.get("frame")
    if hasattr(pop, "meta") and pop.meta and isinstance(pop.meta, Mapping):
        return pop.meta.get("frame")
    return None


def _frame_payload(frame: Any | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    if isinstance(frame, Mapping):
        origin = frame.get("origin")
        extent = frame.get("extent")
        layout = frame.get("layout")
    else:
        origin = getattr(frame, "origin", None)
        extent = getattr(frame, "extent", None)
        layout = getattr(frame, "layout", None)
    if origin is None or extent is None or layout is None:
        return None
    return {
        "origin": _vec3(origin),
        "extent": _vec3(extent),
        "layout": str(layout),
    }


def _vec3(value: Any) -> list[float]:
    if isinstance(value, Mapping) and {"x", "y", "z"}.issubset(value):
        return [float(value["x"]), float(value["y"]), float(value["z"])]
    try:
        seq = list(value)
    except TypeError:
        return [0.0, 0.0, 0.0]
    if len(seq) >= 3:
        return [float(seq[0]), float(seq[1]), float(seq[2])]
    if len(seq) == 2:
        return [float(seq[0]), float(seq[1]), 0.0]
    if len(seq) == 1:
        return [float(seq[0]), 0.0, 0.0]
    return [0.0, 0.0, 0.0]


def _pop_positions(pop: Any) -> list[list[float]] | None:
    if hasattr(pop, "positions"):
        return _to_positions_list(pop.positions)
    if isinstance(pop, Mapping):
        return _to_positions_list(pop.get("positions"))
    return None


def _neuron_pos(
    pos_list: list[list[float]] | None,
    idx: int,
    fallback_x: float,
    fallback_y: float,
) -> list[float]:
    if pos_list is not None and 0 <= idx < len(pos_list):
        pos = pos_list[idx]
        if len(pos) >= 3:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
        if len(pos) == 2:
            return [float(pos[0]), float(pos[1]), 0.0]
        if len(pos) == 1:
            return [float(pos[0]), 0.0, 0.0]
    return [float(fallback_x), float(fallback_y), 0.0]


def _to_positions_list(tensor: Any) -> list[list[float]] | None:
    if tensor is None:
        return None
    data: Any = tensor
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "tolist"):
        data = data.tolist()
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        return None
    if not data:
        return []
    first = data[0]
    if isinstance(first, (int, float)):
        if len(data) >= 3:
            return [[float(data[0]), float(data[1]), float(data[2])]]
        if len(data) == 2:
            return [[float(data[0]), float(data[1]), 0.0]]
        if len(data) == 1:
            return [[float(data[0]), 0.0, 0.0]]
        return []
    positions: list[list[float]] = []
    for item in data:
        if hasattr(item, "x") and hasattr(item, "y") and hasattr(item, "z"):
            positions.append([float(item.x), float(item.y), float(item.z)])
            continue
        if isinstance(item, Mapping) and {"x", "y"}.issubset(item):
            positions.append([float(item["x"]), float(item["y"]), float(item.get("z", 0.0))])
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) >= 3:
                positions.append([float(item[0]), float(item[1]), float(item[2])])
            elif len(item) == 2:
                positions.append([float(item[0]), float(item[1]), 0.0])
            elif len(item) == 1:
                positions.append([float(item[0]), 0.0, 0.0])
    return positions


def _infer_layer_from_name(name: str, idx: int) -> int | None:
    lowered = name.lower()
    if "input" in lowered:
        return 0
    if "hidden" in lowered:
        return 1
    if "output" in lowered or lowered.startswith("out"):
        return 2
    return None


def _proj_name(proj: Any) -> str:
    if hasattr(proj, "name"):
        return str(proj.name)
    if isinstance(proj, Mapping):
        name = proj.get("name") or proj.get("id")
        if name is not None:
            return str(name)
    return f"{_proj_pre(proj)}->{_proj_post(proj)}"


def _proj_pre(proj: Any) -> str:
    if hasattr(proj, "pre"):
        return str(proj.pre)
    if isinstance(proj, Mapping):
        return str(proj.get("pre") or proj.get("from") or "pre")
    return "pre"


def _proj_post(proj: Any) -> str:
    if hasattr(proj, "post"):
        return str(proj.post)
    if isinstance(proj, Mapping):
        return str(proj.get("post") or proj.get("to") or "post")
    return "post"


def _proj_topology(proj: Any) -> SynapseTopology:
    if hasattr(proj, "topology"):
        return cast(SynapseTopology, proj.topology)
    if isinstance(proj, Mapping) and "topology" in proj:
        return cast(SynapseTopology, proj["topology"])
    raise ValueError("Projection missing topology")


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((val - mean) ** 2 for val in values) / len(values)
    return float(mean), float(variance**0.5)


def _corr_sample(values_a: Sequence[float], values_b: Sequence[float], *, max_samples: int) -> float | None:
    n = min(len(values_a), len(values_b))
    if n == 0:
        return None
    sample_a: list[float]
    sample_b: list[float]
    if max_samples <= 0 or n <= max_samples:
        sample_a = [float(val) for val in values_a[:n]]
        sample_b = [float(val) for val in values_b[:n]]
    else:
        step = n / max_samples
        sample_a = []
        sample_b = []
        for i in range(max_samples):
            idx = int(i * step)
            sample_a.append(float(values_a[idx]))
            sample_b.append(float(values_b[idx]))
    mean_a = sum(sample_a) / len(sample_a)
    mean_b = sum(sample_b) / len(sample_b)
    var_a = 0.0
    var_b = 0.0
    cov = 0.0
    for a, b in zip(sample_a, sample_b, strict=True):
        da = a - mean_a
        db = b - mean_b
        var_a += da * da
        var_b += db * db
        cov += da * db
    if var_a <= 0.0 or var_b <= 0.0:
        return 0.0
    return float(cov / (var_a**0.5 * var_b**0.5))


def _receptor_counts(topology: SynapseTopology) -> dict[str, int] | None:
    if topology.receptor is None:
        return None
    receptor_ids = _to_int_list(topology.receptor)
    kinds = topology.receptor_kinds or (ReceptorKind.AMPA, ReceptorKind.NMDA, ReceptorKind.GABA)
    names = [kind.value for kind in kinds]
    counts: dict[str, int] = {}
    for idx in receptor_ids:
        if idx < 0 or idx >= len(names):
            counts["unknown"] = counts.get("unknown", 0) + 1
        else:
            name = names[idx]
            counts[name] = counts.get(name, 0) + 1
    return counts


def _target_counts(topology: SynapseTopology) -> dict[str, int] | None:
    if topology.target_compartments is None:
        return None
    values = _to_int_list(topology.target_compartments)
    comp_names = [comp.value for comp in Compartment]
    counts: dict[str, int] = {}
    for idx in values:
        if idx < 0 or idx >= len(comp_names):
            counts["unknown"] = counts.get("unknown", 0) + 1
        else:
            name = comp_names[idx]
            counts[name] = counts.get(name, 0) + 1
    return counts


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


def _to_int_list(tensor: Tensor | None) -> list[int]:
    if tensor is None:
        return []
    data: Any = tensor
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
    data: Any = tensor
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "tolist"):
        data = data.tolist()
    return [float(value) for value in data]


__all__ = ["build_population_topology_payload"]
