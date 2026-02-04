"""Topology compilation utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, SupportsInt, cast

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import ReceptorKind, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch


def compile_topology(
    topology: SynapseTopology,
    device: Any,
    dtype: Any,
    *,
    build_edges_by_delay: bool = False,
    build_pre_adjacency: bool = False,
    build_sparse_delay_mats: bool = False,
    build_bucket_edge_mapping: bool = False,
) -> SynapseTopology:
    """Cast/move topology tensors to the requested device/dtype and populate meta."""

    torch = require_torch()
    if build_bucket_edge_mapping:
        build_sparse_delay_mats = True
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

    if build_edges_by_delay:
        edges_by_delay = meta.get("edges_by_delay")
        rebuild_edges = True
        if isinstance(edges_by_delay, list):
            first = edges_by_delay[0] if edges_by_delay else None
            if first is None:
                rebuild_edges = False
            elif hasattr(first, "device"):
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

    if build_sparse_delay_mats:
        sparse_mats = meta.get("W_by_delay_by_comp") or meta.get("W_by_delay")
        rebuild_sparse = True
        if isinstance(sparse_mats, list):
            first = sparse_mats[0] if sparse_mats else None
            if first is None:
                rebuild_sparse = False
            elif hasattr(first, "device"):
                if device_obj is None or first.device == device_obj:
                    rebuild_sparse = False
            else:
                rebuild_sparse = False
        if build_bucket_edge_mapping and (
            meta.get("edge_bucket_comp") is None
            or meta.get("edge_bucket_delay") is None
            or meta.get("edge_bucket_pos") is None
        ):
            rebuild_sparse = True

        if rebuild_sparse:
            weights_tensor = weights if weights is not None else None
            receptor_scale = meta.get("receptor_scale") if isinstance(meta, dict) else None
            (
                mats_by_comp,
                values_by_comp,
                indices_by_comp,
                scale_by_comp,
                edge_scale,
                edge_bucket_comp,
                edge_bucket_delay,
                edge_bucket_pos,
            ) = _build_sparse_delay_mats_by_comp(
                pre_idx=pre_idx,
                post_idx=post_idx,
                weights=weights_tensor,
                delay_steps=delay_steps,
                max_delay=int(meta["max_delay_steps"]),
                n_pre=int(meta["n_pre"]),
                n_post=int(meta["n_post"]),
                device=device_obj,
                dtype=_resolve_sparse_dtype(torch, dtype_obj, weights_tensor),
                target_compartment=topology.target_compartment,
                target_compartments=target_compartments,
                receptor_ids=receptor,
                receptor_kinds=topology.receptor_kinds,
                receptor_scale=receptor_scale,
                build_bucket_edge_mapping=build_bucket_edge_mapping,
            )
            keep_coo = bool(meta.get("keep_sparse_coo", False))
            if keep_coo:
                meta["W_by_delay_by_comp"] = mats_by_comp
            meta["values_by_comp"] = values_by_comp
            meta["indices_by_comp"] = indices_by_comp
            meta["scale_by_comp"] = scale_by_comp

            nonempty_coo: dict[Compartment, list[tuple[int, Tensor]]] = {}
            for comp, mats in mats_by_comp.items():
                nonempty_coo[comp] = [
                    (delay, mat)
                    for delay, mat in enumerate(mats)
                    if mat is not None
                ]
            if keep_coo:
                meta["nonempty_mats_by_comp"] = nonempty_coo

            csr_supported = hasattr(torch.Tensor, "to_sparse_csr")
            mats_by_comp_csr: dict[Compartment, list[Tensor | None]] = {}
            if csr_supported:
                nonempty_csr: dict[Compartment, list[tuple[int, Tensor]]] = {}
                for comp, mats in mats_by_comp.items():
                    mats_csr: list[Tensor | None] = [None for _ in range(len(mats))]
                    for delay, mat in enumerate(mats):
                        if mat is None:
                            continue
                        try:
                            mats_csr[delay] = mat.to_sparse_csr()
                        except Exception:
                            mats_csr[delay] = None
                    mats_by_comp_csr[comp] = mats_csr
                    nonempty_csr[comp] = [
                        (delay, cast(Tensor, mat))
                        for delay, mat in enumerate(mats_csr)
                        if mat is not None
                    ]
                meta["W_by_delay_by_comp_csr"] = mats_by_comp_csr
                meta["nonempty_mats_by_comp_csr"] = nonempty_csr
            if build_bucket_edge_mapping:
                meta["edge_scale"] = edge_scale
                meta["edge_bucket_comp"] = edge_bucket_comp
                meta["edge_bucket_delay"] = edge_bucket_delay
                meta["edge_bucket_pos"] = edge_bucket_pos
            if target_compartments is None and keep_coo:
                meta["W_by_delay"] = mats_by_comp.get(topology.target_compartment)
            if target_compartments is None and csr_supported:
                meta["W_by_delay_csr"] = mats_by_comp_csr.get(topology.target_compartment)

    if build_pre_adjacency:
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


def _build_sparse_delay_mats_by_comp(
    *,
    pre_idx: Tensor,
    post_idx: Tensor,
    weights: Tensor | None,
    delay_steps: Tensor | None,
    max_delay: int,
    n_pre: int,
    n_post: int,
    device: Any,
    dtype: Any,
    target_compartment: Compartment,
    target_compartments: Tensor | None,
    receptor_ids: Tensor | None,
    receptor_kinds: tuple[Any, ...] | None,
    receptor_scale: Mapping[Any, float] | None,
    build_bucket_edge_mapping: bool,
) -> tuple[
    dict[Compartment, list[Tensor | None]],
    dict[Compartment, list[Tensor | None]],
    dict[Compartment, list[Tensor | None]],
    dict[Compartment, list[Tensor | None]],
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
]:
    torch = require_torch()
    depth = max_delay + 1
    edge_count = int(pre_idx.numel())
    if n_pre <= 0 or n_post <= 0:
        empty = [None for _ in range(depth)]
        empty_edges = torch.empty((0,), device=device, dtype=torch.long)
        return (
            {target_compartment: list(empty)},
            {target_compartment: list(empty)},
            {target_compartment: list(empty)},
            {target_compartment: list(empty)},
            torch.empty((0,), device=device, dtype=dtype) if build_bucket_edge_mapping else None,
            empty_edges if build_bucket_edge_mapping else None,
            empty_edges if build_bucket_edge_mapping else None,
            empty_edges if build_bucket_edge_mapping else None,
        )
    if delay_steps is None:
        delays = torch.zeros_like(pre_idx, dtype=torch.long)
    else:
        delays = delay_steps.to(dtype=torch.long)
    if weights is None:
        values = torch.ones((pre_idx.numel(),), device=device, dtype=dtype)
    else:
        values = weights.to(device=device, dtype=dtype)

    if receptor_ids is not None and receptor_scale is not None:
        kinds = receptor_kinds or (ReceptorKind.AMPA, ReceptorKind.NMDA, ReceptorKind.GABA)
        table_vals = [receptor_scale.get(kind, 1.0) for kind in kinds]
        table = torch.tensor(table_vals, device=device, dtype=dtype)
        scale_all = table.index_select(0, receptor_ids)
    else:
        scale_all = torch.ones_like(values, device=device, dtype=dtype)

    values_all = values * scale_all

    mats_by_comp: dict[Compartment, list[Tensor | None]] = {}
    values_by_comp: dict[Compartment, list[Tensor | None]] = {}
    indices_by_comp: dict[Compartment, list[Tensor | None]] = {}
    scale_by_comp: dict[Compartment, list[Tensor | None]] = {}

    for comp in _compartments_from_meta(target_compartment, target_compartments):
        mats_by_comp[comp] = [None for _ in range(depth)]
        values_by_comp[comp] = [None for _ in range(depth)]
        indices_by_comp[comp] = [None for _ in range(depth)]
        scale_by_comp[comp] = [None for _ in range(depth)]

    comp_order = tuple(Compartment)
    comp_id_map = {comp: comp_order.index(comp) for comp in mats_by_comp}
    edge_bucket_comp = (
        torch.full((edge_count,), -1, device=device, dtype=torch.long)
        if build_bucket_edge_mapping
        else None
    )
    edge_bucket_delay = (
        torch.full((edge_count,), -1, device=device, dtype=torch.long)
        if build_bucket_edge_mapping
        else None
    )
    edge_bucket_pos = (
        torch.full((edge_count,), -1, device=device, dtype=torch.long)
        if build_bucket_edge_mapping
        else None
    )

    comp_masks: dict[Compartment, Tensor | None]
    if target_compartments is None:
        comp_masks = {target_compartment: None}
    else:
        comp_order = tuple(Compartment)
        comp_masks = {comp: target_compartments == comp_order.index(comp) for comp in mats_by_comp}

    for comp, comp_mask in comp_masks.items():
        for delay in range(depth):
            delay_mask = delays == delay
            mask = delay_mask & comp_mask if comp_mask is not None else delay_mask
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).flatten()
            rows = post_idx.index_select(0, idx)
            cols = pre_idx.index_select(0, idx)
            vals = values_all.index_select(0, idx)
            scale_vals = scale_all.index_select(0, idx)
            indices = torch.stack([rows, cols], dim=0)
            mat = torch.sparse_coo_tensor(
                indices,
                vals,
                size=(n_post, n_pre),
                device=device,
                dtype=dtype,
            )
            mat = mat.coalesce()
            mats_by_comp[comp][delay] = mat
            values_by_comp[comp][delay] = mat.values()
            indices_by_comp[comp][delay] = mat.indices()
            scale_by_comp[comp][delay] = scale_vals
            if build_bucket_edge_mapping:
                assert edge_bucket_comp is not None
                assert edge_bucket_delay is not None
                assert edge_bucket_pos is not None
                comp_id = comp_id_map[comp]
                edge_bucket_comp.index_fill_(0, idx, comp_id)
                edge_bucket_delay.index_fill_(0, idx, delay)
                mat_indices = mat.indices()
                mat_keys = mat_indices[0] * n_pre + mat_indices[1]
                edge_keys = rows * n_pre + cols
                pos = torch.searchsorted(mat_keys, edge_keys)
                edge_bucket_pos.index_copy_(0, idx, pos)

    return (
        mats_by_comp,
        values_by_comp,
        indices_by_comp,
        scale_by_comp,
        scale_all if build_bucket_edge_mapping else None,
        edge_bucket_comp,
        edge_bucket_delay,
        edge_bucket_pos,
    )


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
