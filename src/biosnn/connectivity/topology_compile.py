"""Topology compilation utilities."""

from __future__ import annotations

from typing import Any

from biosnn.connectivity.compile import (
    _build_compiled_artifacts,
    _finalize_topology,
    _normalize_topology_inputs,
)
from biosnn.connectivity.compile.fused_build import _build_fused_sparse_by_comp
from biosnn.contracts.synapses import SynapseTopology


def compile_topology(
    topology: SynapseTopology,
    device: Any,
    dtype: Any,
    *,
    build_edges_by_delay: bool = False,
    build_pre_adjacency: bool = False,
    build_sparse_delay_mats: bool = False,
    build_bucket_edge_mapping: bool = False,
    fuse_delay_buckets: bool = True,
    store_sparse_by_delay: bool = True,
    build_fused_csr: bool = False,
) -> SynapseTopology:
    """Cast/move topology tensors to the requested device/dtype and populate meta.

    Returns a new SynapseTopology instance; the input topology is left unchanged.
    When building sparse delay mats, fused delay buckets are created by default
    (disable via fuse_delay_buckets=False).
    Set store_sparse_by_delay=False to keep only fused sparse artifacts and
    edge bucket mappings. Set build_fused_csr=True to store an optional CSR
    representation alongside the fused COO.
    """

    if build_bucket_edge_mapping:
        build_sparse_delay_mats = True
    normalized, base_meta = _normalize_topology_inputs(topology, device, dtype)
    compiled_meta = _build_compiled_artifacts(
        topology,
        normalized,
        base_meta,
        build_edges_by_delay=build_edges_by_delay,
        build_pre_adjacency=build_pre_adjacency,
        build_sparse_delay_mats=build_sparse_delay_mats,
        build_bucket_edge_mapping=build_bucket_edge_mapping,
        fuse_delay_buckets=fuse_delay_buckets,
        store_sparse_by_delay=store_sparse_by_delay,
        build_fused_csr=build_fused_csr,
    )
    return _finalize_topology(topology, normalized, base_meta, compiled_meta)


__all__ = [
    "compile_topology",
    "_build_compiled_artifacts",
    "_normalize_topology_inputs",
    "_finalize_topology",
    "_build_fused_sparse_by_comp",
]
