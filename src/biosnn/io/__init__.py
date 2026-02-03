"""I/O utilities for exporting dashboards and artifacts."""

from __future__ import annotations

from typing import Any

__all__ = [
    "export_dashboard_snapshot",
    "export_neuron_csv",
    "export_neuron_snapshot",
    "export_synapse_csv",
    "export_topology_json",
    "export_population_topology_json",
    "export_dashboard_bundle",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from biosnn.io import dashboard_export

        return getattr(dashboard_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
