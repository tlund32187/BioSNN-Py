"""Topology compilation utilities."""

from __future__ import annotations

from typing import Any, SupportsInt, cast

from biosnn.biophysics.models._torch_utils import require_torch
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor


def compile_topology(
    topology: SynapseTopology,
    device: Any,
    dtype: Any,
) -> SynapseTopology:
    """Cast/move topology tensors to the requested device/dtype and populate meta."""

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

    meta = dict(topology.meta) if topology.meta else {}
    if "n_pre" not in meta:
        meta["n_pre"] = _infer_size(pre_idx)
    if "n_post" not in meta:
        meta["n_post"] = _infer_size(post_idx)
    if "max_delay_steps" not in meta:
        meta["max_delay_steps"] = _max_delay(delay_steps)

    if target_compartments is not None and "target_comp_ids" not in meta:
        comp_ids = target_compartments
        ids = comp_ids.detach()
        if hasattr(ids, "unique"):
            ids = ids.unique()
        if hasattr(ids, "cpu"):
            ids = ids.cpu()
        if hasattr(ids, "tolist"):
            meta["target_comp_ids"] = [int(val) for val in ids.tolist()]

    edges_by_delay = meta.get("edges_by_delay")
    rebuild_edges = True
    if isinstance(edges_by_delay, list) and edges_by_delay:
        first = edges_by_delay[0]
        if hasattr(first, "device"):
            if device_obj is None or first.device == device_obj:
                rebuild_edges = False
        else:
            rebuild_edges = False

    if rebuild_edges:
        meta["edges_by_delay"] = _bucket_edges_by_delay(
            delay_steps=delay_steps,
            edge_count=_edge_count(pre_idx),
            max_delay=int(meta["max_delay_steps"]),
            device=device_obj,
        )

    rebuild_adj = False
    pre_ptr_meta = meta.get("pre_ptr")
    edge_idx_meta = meta.get("edge_idx")
    if pre_ptr_meta is None or edge_idx_meta is None:
        rebuild_adj = True
    else:
        if (
            device_obj is not None
            and hasattr(pre_ptr_meta, "device")
            and pre_ptr_meta.device != device_obj
        ):
            rebuild_adj = True
        if (
            device_obj is not None
            and hasattr(edge_idx_meta, "device")
            and edge_idx_meta.device != device_obj
        ):
            rebuild_adj = True

    if rebuild_adj:
        pre_ptr, edge_idx = _build_pre_adjacency(
            pre_idx=pre_idx,
            n_pre=int(meta["n_pre"]),
            device=device_obj,
        )
        meta["pre_ptr"] = pre_ptr
        meta["edge_idx"] = edge_idx

    object.__setattr__(topology, "pre_idx", pre_idx)
    object.__setattr__(topology, "post_idx", post_idx)
    object.__setattr__(topology, "delay_steps", delay_steps)
    object.__setattr__(topology, "receptor", receptor)
    object.__setattr__(topology, "target_compartments", target_compartments)
    object.__setattr__(topology, "weights", weights)
    object.__setattr__(topology, "meta", meta)
    return topology


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
    if hasattr(delay_steps, "numel") and not delay_steps.numel():
        return 0
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


def _edge_count(indices: Tensor | None) -> int:
    if indices is None:
        return 0
    if hasattr(indices, "numel"):
        return int(indices.numel())
    try:
        return len(indices)
    except TypeError:
        return 0


def _bucket_edges_by_delay(
    *,
    delay_steps: Tensor | None,
    edge_count: int,
    max_delay: int,
    device: Any,
) -> list[Tensor]:
    torch = require_torch()
    if edge_count <= 0:
        return [torch.empty((0,), device=device, dtype=torch.long)]
    if delay_steps is None:
        return [torch.arange(edge_count, device=device, dtype=torch.long)]
    buckets: list[Tensor] = []
    for delay in range(max_delay + 1):
        mask = delay_steps == delay
        if mask.any():
            buckets.append(mask.nonzero(as_tuple=False).flatten())
        else:
            buckets.append(torch.empty((0,), device=device, dtype=torch.long))
    return buckets


def _build_pre_adjacency(
    *,
    pre_idx: Tensor,
    n_pre: int,
    device: Any,
) -> tuple[Tensor, Tensor]:
    torch = require_torch()
    edge_count = _edge_count(pre_idx)
    if edge_count <= 0 or n_pre <= 0:
        pre_ptr = torch.zeros((n_pre + 1,), device=device, dtype=torch.long)
        edge_idx = torch.empty((0,), device=device, dtype=torch.long)
        return pre_ptr, edge_idx

    edge_idx = torch.argsort(pre_idx)
    pre_sorted = pre_idx.index_select(0, edge_idx)
    counts = torch.bincount(pre_sorted, minlength=n_pre)
    pre_ptr = torch.zeros((n_pre + 1,), device=device, dtype=torch.long)
    pre_ptr[1:] = torch.cumsum(counts, 0)
    return pre_ptr, edge_idx


__all__ = ["compile_topology"]
