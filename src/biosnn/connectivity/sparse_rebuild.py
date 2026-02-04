"""Utilities to rebuild sparse delay matrices from topology weights."""

from __future__ import annotations

from typing import Any

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.synapses import SynapseTopology


def rebuild_sparse_delay_mats(
    topology: SynapseTopology,
    device: Any,
    dtype: Any,
    *,
    build_bucket_edge_mapping: bool | None = None,
) -> SynapseTopology:
    """Rebuild sparse delay matrices from topology.weights.

    This is a maintenance/debug utility and is not intended for the hot path.
    """

    meta = dict(topology.meta) if topology.meta else {}
    if build_bucket_edge_mapping is None:
        build_bucket_edge_mapping = any(
            key in meta
            for key in ("edge_bucket_comp", "edge_bucket_delay", "edge_bucket_pos")
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

    if build_bucket_edge_mapping:
        for key in ("edge_scale", "edge_bucket_comp", "edge_bucket_delay", "edge_bucket_pos"):
            meta.pop(key, None)

    object.__setattr__(topology, "meta", meta)

    return compile_topology(
        topology,
        device=device,
        dtype=dtype,
        build_sparse_delay_mats=True,
        build_bucket_edge_mapping=bool(build_bucket_edge_mapping),
    )


__all__ = ["rebuild_sparse_delay_mats"]
