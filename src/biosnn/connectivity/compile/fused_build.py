"""Fused sparse mat builders and routing metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import ReceptorKind
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch

from .normalize import _compartments_from_meta


def _build_fused_sparse_direct(
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
    receptor_kinds: tuple[ReceptorKind, ...] | None,
    receptor_scale: Mapping[ReceptorKind, float] | None,
    build_bucket_edge_mapping: bool,
) -> tuple[
    dict[Compartment, Tensor],
    dict[Compartment, Tensor],
    dict[Compartment, int],
    dict[Compartment, dict[int, int]],
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
]:
    torch = require_torch()
    edge_count = int(pre_idx.numel())
    if n_pre <= 0 or n_post <= 0 or edge_count <= 0:
        empty_edges = torch.empty((0,), device=device, dtype=torch.long)
        empty_vals = torch.empty((0,), device=device, dtype=dtype)
        return (
            {},
            {},
            {},
            {},
            empty_vals if build_bucket_edge_mapping else None,
            empty_edges if build_bucket_edge_mapping else None,
            empty_edges if build_bucket_edge_mapping else None,
            empty_edges if build_bucket_edge_mapping else None,
            empty_edges if build_bucket_edge_mapping else None,
        )

    delays = delay_steps.to(dtype=torch.long) if delay_steps is not None else torch.zeros_like(pre_idx)
    if weights is None:
        values = torch.ones((edge_count,), device=device, dtype=dtype)
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

    fused_by_comp: dict[Compartment, Tensor] = {}
    delays_by_comp: dict[Compartment, Tensor] = {}
    n_post_by_comp: dict[Compartment, int] = {}
    offsets_by_comp: dict[Compartment, dict[int, int]] = {}

    comp_order = tuple(Compartment)
    comps = _compartments_from_meta(target_compartment, target_compartments)
    comp_id_map = {comp: comp_order.index(comp) for comp in comps}

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
    edge_bucket_fused_pos = None

    comp_masks: dict[Compartment, Tensor | None]
    if target_compartments is None:
        comp_masks = {comps[0]: None} if comps else {}
    else:
        comp_masks = {
            comp: target_compartments == comp_order.index(comp)
            for comp in comps
        }

    for comp, comp_mask in comp_masks.items():
        comp_idx = (
            torch.arange(edge_count, device=device, dtype=torch.long)
            if comp_mask is None
            else comp_mask.nonzero(as_tuple=False).flatten()
        )
        if comp_idx.numel() == 0:
            fused_by_comp[comp] = torch.sparse_coo_tensor(
                torch.empty((2, 0), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=dtype),
                size=(0, n_pre),
                device=device,
                dtype=dtype,
            ).coalesce()
            delays_by_comp[comp] = torch.empty((0,), device=device, dtype=torch.long)
            n_post_by_comp[comp] = n_post
            offsets_by_comp[comp] = {}
            continue

        pre_sel = pre_idx.index_select(0, comp_idx)
        post_sel = post_idx.index_select(0, comp_idx)
        delay_sel = delays.index_select(0, comp_idx)
        values_sel = values_all.index_select(0, comp_idx)

        delay_unique = torch.unique(delay_sel, sorted=True)
        block_ids = torch.arange(delay_unique.numel(), device=device, dtype=torch.long)
        delay_table = torch.full((max_delay + 1,), -1, device=device, dtype=torch.long)
        delay_table.index_copy_(0, delay_unique, block_ids)
        block_idx = delay_table.index_select(0, delay_sel)

        fused_row = post_sel + block_idx * n_post
        fused_col = pre_sel
        fused_indices = torch.stack([fused_row, fused_col], dim=0)
        fused_mat = torch.sparse_coo_tensor(
            fused_indices,
            values_sel,
            size=(int(delay_unique.numel()) * n_post, n_pre),
            device=device,
            dtype=dtype,
        ).coalesce()

        fused_by_comp[comp] = fused_mat
        delays_by_comp[comp] = delay_unique
        n_post_by_comp[comp] = n_post
        offsets: dict[int, int] = {}

        if build_bucket_edge_mapping:
            assert edge_bucket_comp is not None
            assert edge_bucket_delay is not None
            assert edge_bucket_pos is not None
            edge_bucket_comp.index_fill_(0, comp_idx, comp_id_map[comp])
            edge_bucket_delay.index_copy_(0, comp_idx, delay_sel)

            nnz_total = 0
            for delay_val in delay_unique.tolist():
                delay_mask = delay_sel == int(delay_val)
                selected = delay_mask.nonzero(as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                idx = comp_idx.index_select(0, selected)
                rows = post_sel.index_select(0, selected)
                cols = pre_sel.index_select(0, selected)
                keys = rows * n_pre + cols
                unique_keys = torch.unique(keys, sorted=True)
                pos = torch.searchsorted(unique_keys, keys)
                edge_bucket_pos.index_copy_(0, idx, pos)
                offsets[int(delay_val)] = nnz_total
                nnz_total += int(unique_keys.numel())

            # edge_bucket_fused_pos is computed later via offset-table gather.

        offsets_by_comp[comp] = offsets
    return (
        fused_by_comp,
        delays_by_comp,
        n_post_by_comp,
        offsets_by_comp,
        scale_all if build_bucket_edge_mapping else None,
        edge_bucket_comp,
        edge_bucket_delay,
        edge_bucket_pos,
        edge_bucket_fused_pos,
    )


def _build_fused_sparse_by_comp(
    *,
    indices_by_comp: Mapping[Compartment, list[Tensor | None]],
    values_by_comp: Mapping[Compartment, list[Tensor | None]],
    n_post: int,
    n_pre: int,
    device: Any,
    dtype: Any,
) -> tuple[
    dict[Compartment, Tensor],
    dict[Compartment, Tensor],
    dict[Compartment, int],
    dict[Compartment, dict[int, int]],
]:
    torch = require_torch()
    fused_by_comp: dict[Compartment, Tensor] = {}
    delays_by_comp: dict[Compartment, Tensor] = {}
    n_post_by_comp: dict[Compartment, int] = {}
    offsets_by_comp: dict[Compartment, dict[int, int]] = {}

    for comp, indices_by_delay in indices_by_comp.items():
        if not isinstance(indices_by_delay, list):
            continue
        values_by_delay = values_by_comp.get(comp)
        if not isinstance(values_by_delay, list):
            continue
        indices_list: list[Tensor] = []
        values_list: list[Tensor] = []
        delays: list[int] = []
        offsets: dict[int, int] = {}
        bucket_idx = 0
        nnz_total = 0
        for delay, indices in enumerate(indices_by_delay):
            if indices is None:
                continue
            if delay >= len(values_by_delay):
                continue
            values = values_by_delay[delay]
            if values is None or indices.numel() == 0 or values.numel() == 0:
                continue
            rows = indices[0] + bucket_idx * n_post
            cols = indices[1]
            indices_list.append(torch.stack([rows, cols], dim=0))
            values_list.append(values)
            delays.append(delay)
            offsets[delay] = nnz_total
            nnz_total += int(values.numel())
            bucket_idx += 1
        if not delays:
            continue
        fused_indices = indices_list[0] if len(indices_list) == 1 else torch.cat(indices_list, dim=1)
        fused_values = values_list[0] if len(values_list) == 1 else torch.cat(values_list)
        fused = torch.sparse_coo_tensor(
            fused_indices,
            fused_values,
            size=(bucket_idx * n_post, n_pre),
            device=device,
            dtype=dtype,
        ).coalesce()
        fused_by_comp[comp] = fused
        delays_by_comp[comp] = torch.tensor(delays, device=device, dtype=torch.long)
        n_post_by_comp[comp] = n_post
        offsets_by_comp[comp] = offsets

    return fused_by_comp, delays_by_comp, n_post_by_comp, offsets_by_comp


def _ensure_fused_routing_meta(meta: dict[str, Any], device: Any) -> None:
    torch = require_torch()
    fused_delays = meta.get("fused_W_delays_by_comp")
    if not isinstance(fused_delays, dict):
        return
    if (
        isinstance(meta.get("fused_is_immediate_by_comp"), dict)
        and isinstance(meta.get("fused_immediate_blocks_idx_by_comp"), dict)
        and isinstance(meta.get("fused_delayed_blocks_idx_by_comp"), dict)
    ):
        return
    is_immediate_by_comp: dict[Compartment, Tensor] = {}
    immediate_idx_by_comp: dict[Compartment, Tensor] = {}
    delayed_idx_by_comp: dict[Compartment, Tensor] = {}
    n_blocks_by_comp: dict[Compartment, int] = {}
    delays_long_by_comp: dict[Compartment, Tensor] = {}
    d_by_comp: dict[Compartment, int] = {}
    for comp, delays in fused_delays.items():
        if delays is None:
            continue
        if (
            hasattr(delays, "device")
            and device is not None
            and delays.device != device
            and getattr(delays.device, "type", None) != getattr(device, "type", None)
        ):
            raise RuntimeError(
                "fused_W_delays_by_comp is on a different device; "
                "compile_topology(..., build_sparse_delay_mats=True) with the synapse device."
            )
        delays_tensor = cast(Tensor, delays)
        delays_long = (
            delays_tensor
            if delays_tensor.dtype == torch.long
            else cast(Tensor, delays_tensor.to(dtype=torch.long))
        )
        delays_long_by_comp[cast(Compartment, comp)] = delays_long
        mask = delays_long == 0
        is_immediate_by_comp[cast(Compartment, comp)] = mask
        immediate_idx_by_comp[cast(Compartment, comp)] = mask.nonzero(as_tuple=False).flatten()
        delayed_idx_by_comp[cast(Compartment, comp)] = (~mask).nonzero(as_tuple=False).flatten()
        if hasattr(delays_long, "numel"):
            d_val = int(delays_long.numel())
            n_blocks_by_comp[cast(Compartment, comp)] = d_val
            d_by_comp[cast(Compartment, comp)] = d_val
    meta["fused_is_immediate_by_comp"] = is_immediate_by_comp
    meta["fused_immediate_blocks_idx_by_comp"] = immediate_idx_by_comp
    meta["fused_delayed_blocks_idx_by_comp"] = delayed_idx_by_comp
    meta["fused_W_n_blocks_by_comp"] = n_blocks_by_comp
    meta["fused_W_delays_long_by_comp"] = delays_long_by_comp
    meta["fused_D_by_comp"] = d_by_comp


def _ensure_fused_layout_meta(
    meta: dict[str, Any],
    *,
    build_fused_csr: bool,
    device_obj: Any,
    torch: Any,
) -> None:
    fused_by_comp = meta.get("fused_W_by_comp")
    if not isinstance(fused_by_comp, dict):
        return
    if "fused_W_by_comp_coo" not in meta:
        meta["fused_W_by_comp_coo"] = fused_by_comp
    if build_fused_csr and "fused_W_by_comp_csr" not in meta:
        fused_csr = _build_fused_csr(fused_by_comp, torch)
        meta["fused_W_by_comp_csr"] = fused_csr
    if "fused_layout_preference" not in meta:
        fused_csr_meta = meta.get("fused_W_by_comp_csr")
        prefer_csr = bool(build_fused_csr and _device_is_cpu(device_obj) and fused_csr_meta)
        if prefer_csr:
            meta["fused_layout_preference"] = "csr"
        else:
            meta["fused_layout_preference"] = "coo"


def _store_fused_sparse_meta(
    meta: dict[str, Any],
    fused_by_comp: dict[Compartment, Tensor],
    *,
    build_fused_csr: bool,
    device_obj: Any,
    torch: Any,
) -> None:
    meta["fused_W_by_comp"] = fused_by_comp
    meta["fused_W_by_comp_coo"] = fused_by_comp
    if build_fused_csr:
        fused_csr = _build_fused_csr(fused_by_comp, torch)
        meta["fused_W_by_comp_csr"] = fused_csr
    prefer_csr = bool(build_fused_csr and _device_is_cpu(device_obj) and meta.get("fused_W_by_comp_csr"))
    if prefer_csr:
        meta["fused_layout_preference"] = "csr"
    else:
        meta["fused_layout_preference"] = "coo"


def _build_fused_csr(
    fused_by_comp: dict[Compartment, Tensor], torch: Any
) -> dict[Compartment, Tensor]:
    fused_csr: dict[Compartment, Tensor] = {}
    if not hasattr(torch.Tensor, "to_sparse_csr"):
        return fused_csr
    for comp, mat in fused_by_comp.items():
        if mat is None:
            continue
        try:
            fused_csr[comp] = cast(Tensor, mat.to_sparse_csr())
        except Exception:
            continue
    return fused_csr


def _device_is_cpu(device: Any) -> bool:
    if device is None:
        return True
    try:
        return bool(getattr(device, "type", None) == "cpu")
    except Exception:
        return False


__all__ = [
    "_build_fused_sparse_direct",
    "_build_fused_sparse_by_comp",
    "_ensure_fused_routing_meta",
    "_ensure_fused_layout_meta",
    "_store_fused_sparse_meta",
]
