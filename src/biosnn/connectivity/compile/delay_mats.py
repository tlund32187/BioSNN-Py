"""Sparse per-delay matrix builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import ReceptorKind
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch

from .normalize import _compartments_from_meta


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
        selected = mask.nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            buckets.append(torch.empty((0,), device=device, dtype=torch.long))
        else:
            buckets.append(selected)
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
            idx = mask.nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
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


__all__ = ["_bucket_edges_by_delay", "_build_sparse_delay_mats_by_comp"]
