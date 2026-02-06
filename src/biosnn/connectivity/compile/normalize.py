"""Normalization helpers for topology compilation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, SupportsInt, cast

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch


@dataclass(frozen=True, slots=True)
class _NormalizedTopologyInputs:
    pre_idx: Tensor
    post_idx: Tensor
    delay_steps: Tensor | None
    edge_dist: Tensor | None
    receptor: Tensor | None
    target_compartments: Tensor | None
    weights: Tensor | None
    device_obj: Any
    dtype_obj: Any


def _normalize_topology_inputs(
    topology: SynapseTopology,
    device: Any,
    dtype: Any,
) -> tuple[_NormalizedTopologyInputs, dict[str, Any]]:
    torch = require_torch()
    device_obj = torch.device(device) if device is not None else None
    dtype_obj = _resolve_dtype(torch, dtype)

    pre_idx = _to_long(topology.pre_idx, device_obj, name="pre_idx")
    post_idx = _to_long(topology.post_idx, device_obj, name="post_idx")
    if pre_idx is None or post_idx is None:
        raise ValueError("pre_idx and post_idx must be provided")
    pre_idx = cast(Tensor, pre_idx)
    post_idx = cast(Tensor, post_idx)
    delay_steps = _to_long(topology.delay_steps, device_obj, name="delay_steps")
    edge_dist = _to_float(topology.edge_dist, device_obj, name="edge_dist")
    receptor = _to_long(topology.receptor, device_obj, name="receptor")
    target_compartments = _to_long(
        topology.target_compartments, device_obj, name="target_compartments"
    )

    weights = topology.weights
    if weights is not None:
        weight_dtype = None
        if dtype_obj is not None and getattr(dtype_obj, "is_floating_point", False):
            weight_dtype = dtype_obj
        elif hasattr(weights, "dtype") and getattr(weights.dtype, "is_floating_point", False):
            weight_dtype = weights.dtype
        else:
            weight_dtype = torch.get_default_dtype()
        if hasattr(weights, "to"):
            weights = weights.to(device=device_obj, dtype=weight_dtype)
        else:
            weights = torch.tensor(weights, device=device_obj, dtype=weight_dtype)

    base_meta = dict(topology.meta) if topology.meta else {}
    if "n_pre" not in base_meta:
        base_meta["n_pre"] = _infer_size(pre_idx)
    if "n_post" not in base_meta:
        base_meta["n_post"] = _infer_size(post_idx)
    if "max_delay_steps" not in base_meta:
        base_meta["max_delay_steps"] = _max_delay(delay_steps)

    if target_compartments is not None and "target_comp_ids" not in base_meta:
        comp_ids = target_compartments
        ids = comp_ids.detach()
        if hasattr(ids, "unique"):
            ids = ids.unique()
        if hasattr(ids, "cpu"):
            ids = ids.cpu()
        if hasattr(ids, "tolist"):
            base_meta["target_comp_ids"] = [int(val) for val in ids.tolist()]

    _estimate_ring_meta(
        base_meta,
        topology=topology,
        target_compartments=target_compartments,
        dtype_obj=dtype_obj,
        weights=weights,
    )

    normalized = _NormalizedTopologyInputs(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        edge_dist=edge_dist,
        receptor=receptor,
        target_compartments=target_compartments,
        weights=weights,
        device_obj=device_obj,
        dtype_obj=dtype_obj,
    )
    return normalized, base_meta


def _estimate_ring_meta(
    base_meta: dict[str, Any],
    *,
    topology: SynapseTopology,
    target_compartments: Tensor | None,
    dtype_obj: Any,
    weights: Tensor | None,
) -> None:
    torch = require_torch()
    ring_len = int(base_meta.get("max_delay_steps", 0)) + 1
    n_post = int(base_meta.get("n_post", 0))
    n_comp = _ring_compartment_count(base_meta, topology, target_compartments)
    bytes_per_value = _dtype_bytes(torch, dtype_obj, weights)
    if ring_len <= 0 or n_post <= 0 or n_comp <= 0 or bytes_per_value <= 0:
        estimated_bytes = 0
    else:
        estimated_bytes = int(ring_len * n_post * n_comp * bytes_per_value)
    base_meta["ring_len"] = int(ring_len)
    base_meta["estimated_ring_bytes"] = estimated_bytes
    base_meta["estimated_ring_mib"] = estimated_bytes / (1024.0 * 1024.0)


def _ring_compartment_count(
    base_meta: Mapping[str, Any],
    topology: SynapseTopology,
    target_compartments: Tensor | None,
) -> int:
    _ = topology
    if target_compartments is None:
        return 1
    comp_ids = base_meta.get("target_comp_ids")
    if isinstance(comp_ids, list) and comp_ids:
        return len(comp_ids)
    try:
        return len(Compartment)
    except TypeError:
        return 1


def _dtype_bytes(torch: Any, dtype_obj: Any, weights: Tensor | None) -> int:
    dtype = None
    if weights is not None and hasattr(weights, "dtype"):
        dtype = weights.dtype
    elif dtype_obj is not None:
        dtype = dtype_obj
    else:
        dtype = torch.get_default_dtype()

    if getattr(dtype, "is_floating_point", False) or getattr(dtype, "is_complex", False):
        try:
            return int(torch.finfo(dtype).bits // 8)
        except (TypeError, ValueError):
            pass
    if hasattr(dtype, "itemsize"):
        try:
            return int(dtype.itemsize)
        except (TypeError, ValueError):
            pass
    try:
        return int(torch.iinfo(dtype).bits // 8)
    except (TypeError, ValueError):
        return 8


def _resolve_dtype(torch: Any, dtype: Any) -> Any | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        name = dtype
        if name.startswith("torch."):
            name = name.split(".", 1)[1]
        if not hasattr(torch, name):
            raise ValueError(f"Unknown torch dtype: {dtype}")
        return getattr(torch, name)
    return dtype


def _resolve_sparse_dtype(torch: Any, dtype_obj: Any, weights: Tensor | None) -> Any:
    if dtype_obj is not None and getattr(dtype_obj, "is_floating_point", False):
        return dtype_obj
    if weights is not None and hasattr(weights, "dtype") and getattr(weights.dtype, "is_floating_point", False):
        return weights.dtype
    return torch.get_default_dtype()


def _to_long(tensor: Tensor | None, device: Any, *, name: str) -> Tensor | None:
    if tensor is None:
        return None
    torch = require_torch()
    if hasattr(tensor, "dim") and tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got {tuple(tensor.shape)}")
    if hasattr(tensor, "to"):
        if device is not None:
            return cast(Tensor, tensor.to(device=device, dtype=torch.long))
        return cast(Tensor, tensor.to(dtype=torch.long))
    return cast(Tensor, torch.tensor(tensor, device=device, dtype=torch.long))


def _to_float(tensor: Tensor | None, device: Any, *, name: str) -> Tensor | None:
    if tensor is None:
        return None
    torch = require_torch()
    if hasattr(tensor, "dim") and tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got {tuple(tensor.shape)}")
    if hasattr(tensor, "to"):
        if device is not None:
            return cast(Tensor, tensor.to(device=device, dtype=torch.float32))
        return cast(Tensor, tensor.to(dtype=torch.float32))
    return cast(Tensor, torch.tensor(tensor, device=device, dtype=torch.float32))


def _infer_size(indices: Tensor | None) -> int:
    if indices is None:
        return 0
    if hasattr(indices, "numel") and indices.numel():
        max_val = indices.detach()
        if hasattr(max_val, "max"):
            max_val = max_val.max()
        if hasattr(max_val, "cpu"):
            max_val = max_val.cpu()
            if hasattr(max_val, "tolist"):
                max_list = max_val.tolist()
                if isinstance(max_list, list):
                    scalar = max_list[0] if max_list else 0
                    return int(cast(SupportsInt, scalar)) + 1
                return int(cast(SupportsInt, max_list)) + 1
        return int(max_val) + 1
    return 0


def _max_delay(delay_steps: Tensor | None) -> int:
    if delay_steps is None:
        return 0
    if hasattr(delay_steps, "numel") and delay_steps.numel():
        max_val = delay_steps.detach()
        if hasattr(max_val, "max"):
            max_val = max_val.max()
        if hasattr(max_val, "cpu"):
            max_val = max_val.cpu()
            if hasattr(max_val, "tolist"):
                max_list = max_val.tolist()
                if isinstance(max_list, list):
                    scalar = max_list[0] if max_list else 0
                    return int(cast(SupportsInt, scalar))
                return int(cast(SupportsInt, max_list))
        return int(max_val)
    return 0


def _edge_count(indices: Tensor | None) -> int:
    if indices is None:
        return 0
    if hasattr(indices, "numel"):
        return int(indices.numel())
    try:
        return len(indices)
    except TypeError:
        return 0


def _compartments_from_meta(
    target_compartment: Compartment,
    target_compartments: Tensor | None,
) -> tuple[Compartment, ...]:
    if target_compartments is None:
        return (target_compartment,)
    comp_ids = target_compartments.detach()
    if hasattr(comp_ids, "unique"):
        comp_ids = comp_ids.unique()
    if hasattr(comp_ids, "cpu"):
        comp_ids = comp_ids.cpu()
    raw = comp_ids.tolist() if hasattr(comp_ids, "tolist") else []
    if not isinstance(raw, list):
        raw = [raw]
    comp_order = tuple(Compartment)
    comps: list[Compartment] = []
    for comp_id in raw:
        try:
            idx = int(comp_id)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(comp_order):
            comp = comp_order[idx]
            if comp not in comps:
                comps.append(comp)
    return tuple(comps)


__all__ = [
    "_NormalizedTopologyInputs",
    "_normalize_topology_inputs",
    "_resolve_dtype",
    "_resolve_sparse_dtype",
    "_to_long",
    "_to_float",
    "_infer_size",
    "_max_delay",
    "_edge_count",
    "_compartments_from_meta",
]
