"""Topology compilation submodules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch

from .bucket_mapping import _build_edge_bucket_fused_pos, _build_pre_adjacency
from .delay_mats import _bucket_edges_by_delay, _build_sparse_delay_mats_by_comp
from .finalize import _finalize_topology
from .fused_build import (
    _build_fused_sparse_by_comp,
    _build_fused_sparse_direct,
    _ensure_fused_layout_meta,
    _ensure_fused_routing_meta,
    _store_fused_sparse_meta,
)
from .normalize import (
    _edge_count,
    _normalize_topology_inputs,
    _NormalizedTopologyInputs,
    _resolve_sparse_dtype,
)


def _build_compiled_artifacts(
    topology: SynapseTopology,
    normalized: _NormalizedTopologyInputs,
    base_meta: Mapping[str, Any],
    *,
    build_edges_by_delay: bool,
    build_pre_adjacency: bool,
    build_sparse_delay_mats: bool,
    build_bucket_edge_mapping: bool,
    fuse_delay_buckets: bool,
    store_sparse_by_delay: bool,
    build_fused_csr: bool = False,
) -> dict[str, Any]:
    torch = require_torch()
    meta = dict(base_meta)
    device_obj = normalized.device_obj
    dtype_obj = normalized.dtype_obj
    pre_idx = normalized.pre_idx
    post_idx = normalized.post_idx
    delay_steps = normalized.delay_steps
    target_compartments = normalized.target_compartments
    receptor = normalized.receptor
    weights = normalized.weights

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
            if fuse_delay_buckets and not store_sparse_by_delay:
                (
                    fused_by_comp,
                    fused_delays_by_comp,
                    fused_n_post_by_comp,
                    fused_offsets_by_comp,
                    edge_scale,
                    edge_bucket_comp,
                    edge_bucket_delay,
                    edge_bucket_pos,
                    _edge_bucket_fused_pos,
                ) = _build_fused_sparse_direct(
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
                _store_fused_sparse_meta(
                    meta,
                    fused_by_comp,
                    build_fused_csr=build_fused_csr,
                    device_obj=device_obj,
                    torch=torch,
                )
                meta["fused_W_delays_by_comp"] = fused_delays_by_comp
                meta["fused_W_n_post_by_comp"] = fused_n_post_by_comp
                if build_bucket_edge_mapping:
                    meta["edge_scale"] = edge_scale
                    meta["edge_bucket_comp"] = edge_bucket_comp
                    meta["edge_bucket_delay"] = edge_bucket_delay
                    meta["edge_bucket_pos"] = edge_bucket_pos
                    if (
                        edge_bucket_comp is not None
                        and edge_bucket_delay is not None
                        and edge_bucket_pos is not None
                    ):
                        meta["edge_bucket_fused_pos"] = _build_edge_bucket_fused_pos(
                            edge_bucket_comp=edge_bucket_comp,
                            edge_bucket_delay=edge_bucket_delay,
                            edge_bucket_pos=edge_bucket_pos,
                            fused_offsets_by_comp=fused_offsets_by_comp,
                            max_delay=int(meta["max_delay_steps"]),
                        )
                for key in (
                    "W_by_delay_by_comp",
                    "W_by_delay",
                    "W_by_delay_by_comp_csr",
                    "W_by_delay_csr",
                    "values_by_comp",
                    "indices_by_comp",
                    "scale_by_comp",
                    "nonempty_mats_by_comp",
                    "nonempty_mats_by_comp_csr",
                ):
                    meta.pop(key, None)
            else:
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
                if store_sparse_by_delay:
                    if keep_coo:
                        meta["W_by_delay_by_comp"] = mats_by_comp
                    meta["values_by_comp"] = values_by_comp
                    meta["indices_by_comp"] = indices_by_comp
                    meta["scale_by_comp"] = scale_by_comp

                if fuse_delay_buckets:
                    (
                        fused_by_comp,
                        fused_delays_by_comp,
                        fused_n_post_by_comp,
                        fused_offsets_by_comp,
                    ) = _build_fused_sparse_by_comp(
                        indices_by_comp=indices_by_comp,
                        values_by_comp=values_by_comp,
                        n_post=int(meta["n_post"]),
                        n_pre=int(meta["n_pre"]),
                        device=device_obj,
                        dtype=_resolve_sparse_dtype(torch, dtype_obj, weights_tensor),
                    )
                    _store_fused_sparse_meta(
                        meta,
                        fused_by_comp,
                        build_fused_csr=build_fused_csr,
                        device_obj=device_obj,
                        torch=torch,
                    )
                    meta["fused_W_delays_by_comp"] = fused_delays_by_comp
                    meta["fused_W_n_post_by_comp"] = fused_n_post_by_comp

                if store_sparse_by_delay:
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
                    if (
                        fuse_delay_buckets
                        and edge_bucket_comp is not None
                        and edge_bucket_delay is not None
                        and edge_bucket_pos is not None
                    ):
                        edge_bucket_fused_pos = _build_edge_bucket_fused_pos(
                            edge_bucket_comp=edge_bucket_comp,
                            edge_bucket_delay=edge_bucket_delay,
                            edge_bucket_pos=edge_bucket_pos,
                            fused_offsets_by_comp=fused_offsets_by_comp,
                            max_delay=int(meta["max_delay_steps"]),
                        )
                        meta["edge_bucket_fused_pos"] = edge_bucket_fused_pos
                if store_sparse_by_delay and target_compartments is None and keep_coo:
                    meta["W_by_delay"] = mats_by_comp.get(topology.target_compartment)
                if store_sparse_by_delay and target_compartments is None and csr_supported:
                    meta["W_by_delay_csr"] = mats_by_comp_csr.get(topology.target_compartment)

                if not store_sparse_by_delay:
                    for key in (
                        "W_by_delay_by_comp",
                        "W_by_delay",
                        "W_by_delay_by_comp_csr",
                        "W_by_delay_csr",
                        "values_by_comp",
                        "indices_by_comp",
                        "scale_by_comp",
                        "nonempty_mats_by_comp",
                        "nonempty_mats_by_comp_csr",
                    ):
                        meta.pop(key, None)

        if fuse_delay_buckets:
            fused_ready = (
                isinstance(meta.get("fused_W_by_comp"), dict)
                and isinstance(meta.get("fused_W_delays_by_comp"), dict)
                and isinstance(meta.get("fused_W_n_post_by_comp"), dict)
            )
            if not fused_ready:
                indices_by_comp_meta = meta.get("indices_by_comp")
                values_by_comp_meta = meta.get("values_by_comp")
                if isinstance(indices_by_comp_meta, dict) and isinstance(values_by_comp_meta, dict):
                    (
                        fused_by_comp,
                        fused_delays_by_comp,
                        fused_n_post_by_comp,
                        fused_offsets_by_comp,
                    ) = _build_fused_sparse_by_comp(
                        indices_by_comp=indices_by_comp_meta,
                        values_by_comp=values_by_comp_meta,
                        n_post=int(meta["n_post"]),
                        n_pre=int(meta["n_pre"]),
                        device=device_obj,
                        dtype=_resolve_sparse_dtype(torch, dtype_obj, weights),
                    )
                    _store_fused_sparse_meta(
                        meta,
                        fused_by_comp,
                        build_fused_csr=build_fused_csr,
                        device_obj=device_obj,
                        torch=torch,
                    )
                    meta["fused_W_delays_by_comp"] = fused_delays_by_comp
                    meta["fused_W_n_post_by_comp"] = fused_n_post_by_comp
                    if (
                        build_bucket_edge_mapping
                        and meta.get("edge_bucket_comp") is not None
                        and meta.get("edge_bucket_delay") is not None
                        and meta.get("edge_bucket_pos") is not None
                    ):
                        edge_bucket_comp = cast(Tensor, meta.get("edge_bucket_comp"))
                        edge_bucket_delay = cast(Tensor, meta.get("edge_bucket_delay"))
                        edge_bucket_pos = cast(Tensor, meta.get("edge_bucket_pos"))
                        edge_bucket_fused_pos = _build_edge_bucket_fused_pos(
                            edge_bucket_comp=edge_bucket_comp,
                            edge_bucket_delay=edge_bucket_delay,
                            edge_bucket_pos=edge_bucket_pos,
                            fused_offsets_by_comp=fused_offsets_by_comp,
                            max_delay=int(meta["max_delay_steps"]),
                        )
                        meta["edge_bucket_fused_pos"] = edge_bucket_fused_pos

    if fuse_delay_buckets:
        _ensure_fused_layout_meta(
            meta,
            build_fused_csr=build_fused_csr,
            device_obj=device_obj,
            torch=torch,
        )
        _ensure_fused_routing_meta(meta, device_obj)

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
        if rebuild_adj:
            pre_ptr, edge_idx = _build_pre_adjacency(
                pre_idx=pre_idx,
                n_pre=int(meta["n_pre"]),
                device=device_obj,
            )
            meta["pre_ptr"] = pre_ptr
            meta["edge_idx"] = edge_idx

    return meta


__all__ = [
    "_NormalizedTopologyInputs",
    "_normalize_topology_inputs",
    "_build_compiled_artifacts",
    "_finalize_topology",
]
