"""Deprecated export shim. Use biosnn.io.dashboard_export instead."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, cast

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        _MODULE = importlib.import_module("biosnn.io.dashboard_export")
    return _MODULE


def _warn() -> None:
    warnings.warn(
        "biosnn.synapses.export is deprecated; use biosnn.io.dashboard_export",
        DeprecationWarning,
        stacklevel=2,
    )


def export_topology_json(
    topology,
    *,
    path=None,
    weights=None,
    pre_layer: int = 0,
    post_layer: int = 2,
) -> dict[str, Any]:
    _warn()
    return cast(
        dict[str, Any],
        _module().export_topology_json(
            topology,
            path=path,
            weights=weights,
            pre_layer=pre_layer,
            post_layer=post_layer,
        ),
    )


def export_synapse_csv(
    weights,
    *,
    path,
    stats: Sequence[str] | None = None,
    sample_indices: Sequence[int] | None = None,
    scalars: Mapping[str, float] | None = None,
):
    _warn()
    return cast(
        Any,
        _module().export_synapse_csv(
            weights,
            path=path,
            stats=stats,
            sample_indices=sample_indices,
            scalars=scalars,
        ),
    )


def export_neuron_csv(
    tensors: Mapping[str, Any],
    *,
    path,
    spikes=None,
    stats: Sequence[str] | None = None,
    sample_indices: Sequence[int] | None = None,
    scalars: Mapping[str, float] | None = None,
    t: float = 0.0,
    dt: float = 0.0,
):
    _warn()
    return cast(
        Any,
        _module().export_neuron_csv(
            tensors,
            path=path,
            spikes=spikes,
            stats=stats,
            sample_indices=sample_indices,
            scalars=scalars,
            t=t,
            dt=dt,
        ),
    )


def export_neuron_snapshot(
    model,
    state: Any,
    result,
    *,
    path,
    stats: Sequence[str] | None = None,
    sample_indices: Sequence[int] | None = None,
    scalars: Mapping[str, float] | None = None,
    t: float = 0.0,
    dt: float = 0.0,
):
    _warn()
    return cast(
        Any,
        _module().export_neuron_snapshot(
            model,
            state,
            result,
            path=path,
            stats=stats,
            sample_indices=sample_indices,
            scalars=scalars,
            t=t,
            dt=dt,
        ),
    )


def export_dashboard_snapshot(
    topology,
    weights,
    *,
    out_dir,
    topology_name: str = "topology.json",
    synapse_name: str = "synapse.csv",
    neuron_name: str = "neuron.csv",
    neuron_tensors: Mapping[str, Any] | None = None,
    neuron_spikes=None,
    neuron_stats: Sequence[str] | None = None,
    neuron_samples: int | None = 32,
    synapse_stats: Sequence[str] | None = None,
    synapse_samples: int | None = 64,
    scalars: Mapping[str, float] | None = None,
    neuron_scalars: Mapping[str, float] | None = None,
):
    _warn()
    return cast(
        Any,
        _module().export_dashboard_snapshot(
            topology,
            weights,
            out_dir=out_dir,
            topology_name=topology_name,
            synapse_name=synapse_name,
            neuron_name=neuron_name,
            neuron_tensors=neuron_tensors,
            neuron_spikes=neuron_spikes,
            neuron_stats=neuron_stats,
            neuron_samples=neuron_samples,
            synapse_stats=synapse_stats,
            synapse_samples=synapse_samples,
            scalars=scalars,
            neuron_scalars=neuron_scalars,
        ),
    )


__all__ = [
    "export_dashboard_snapshot",
    "export_neuron_csv",
    "export_neuron_snapshot",
    "export_synapse_csv",
    "export_topology_json",
]
