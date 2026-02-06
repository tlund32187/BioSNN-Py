"""Finalize compiled topology objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from biosnn.contracts.synapses import SynapseTopology

from .normalize import _NormalizedTopologyInputs


def _finalize_topology(
    topology: SynapseTopology,
    normalized: _NormalizedTopologyInputs,
    base_meta: Mapping[str, Any],
    compiled_meta_updates: Mapping[str, Any],
) -> SynapseTopology:
    new_meta = dict(base_meta)
    new_meta.update(compiled_meta_updates)
    removed = set(base_meta) - set(compiled_meta_updates)
    for key in removed:
        new_meta.pop(key, None)

    return replace(
        topology,
        pre_idx=normalized.pre_idx,
        post_idx=normalized.post_idx,
        delay_steps=normalized.delay_steps,
        edge_dist=normalized.edge_dist,
        receptor=normalized.receptor,
        target_compartments=normalized.target_compartments,
        weights=normalized.weights,
        meta=new_meta,
    )


__all__ = ["_finalize_topology"]
