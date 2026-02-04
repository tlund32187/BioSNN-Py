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
        for i in range(n):
            y_offset = 0.0 if n == 1 else (i / (n - 1) - 0.5) * spread
            x_jitter = ((_hash_int(i + offsets[pop_name]) % 1000) / 1000 - 0.5) * 0.05
            neuron_nodes.append(
                {
                    "index": offsets[pop_name] + i,
                    "pop": pop_name,
                    "local_idx": i,
                    "x": _clamp01(base_x + x_jitter),
                    "y": _clamp01(base_y + y_offset),
                    "layer": int(node.get("layer") or 0),
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
        receptor_names = _receptor_names(topo)
        for idx, (src, dst) in enumerate(zip(pre_idx, post_idx, strict=True)):
            edge: dict[str, Any] = {
                "from": pre_offset + src,
                "to": post_offset + dst,
                "weight": weights[idx] if idx < len(weights) else 0.0,
            }
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

        nodes.append(
            {
                "id": _pop_name(pop),
                "label": _pop_label(pop),
                "layer": int(layer),
                "n_neurons": _pop_count(pop),
                "model": _pop_model(pop),
                "x": float(x),
                "y": float(y),
            }
        )

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
