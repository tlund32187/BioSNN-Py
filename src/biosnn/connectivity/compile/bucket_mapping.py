"""Edge bucket mapping helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch

from .normalize import _edge_count


def _build_edge_bucket_fused_pos(
    *,
    edge_bucket_comp: Tensor,
    edge_bucket_delay: Tensor,
    edge_bucket_pos: Tensor,
    fused_offsets_by_comp: Mapping[Compartment, Mapping[int, int]],
    max_delay: int,
) -> Tensor:
    torch = require_torch()
    comp_order = tuple(Compartment)
    n_comp = len(comp_order)
    offset_table = torch.full(
        (n_comp, max_delay + 1),
        -1,
        device=edge_bucket_comp.device,
        dtype=edge_bucket_pos.dtype,
    )
    for comp, offsets in fused_offsets_by_comp.items():
        if comp not in comp_order:
            continue
        comp_id = comp_order.index(comp)
        for delay, offset in offsets.items():
            if 0 <= delay <= max_delay:
                offset_table[comp_id, delay] = int(offset)

    fused_pos = cast(Tensor, torch.full_like(edge_bucket_pos, -1))
    valid = (
        (edge_bucket_comp >= 0)
        & (edge_bucket_comp < n_comp)
        & (edge_bucket_delay >= 0)
        & (edge_bucket_delay <= max_delay)
    )
    valid_idx = valid.nonzero(as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        return fused_pos
    comp_ids = edge_bucket_comp.index_select(0, valid_idx)
    delay_ids = edge_bucket_delay.index_select(0, valid_idx)
    offsets = offset_table[comp_ids, delay_ids]
    offset_idx = (offsets >= 0).nonzero(as_tuple=False).flatten()
    if offset_idx.numel() == 0:
        return fused_pos
    fused_idx = valid_idx.index_select(0, offset_idx)
    pos_vals = edge_bucket_pos.index_select(0, fused_idx)
    fused_pos.index_copy_(0, fused_idx, pos_vals + offsets.index_select(0, offset_idx))
    return fused_pos


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


__all__ = ["_build_edge_bucket_fused_pos", "_build_pre_adjacency"]
