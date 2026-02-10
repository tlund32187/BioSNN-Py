"""Simulation engine contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from biosnn.contracts.monitors import IMonitor


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    dt: float = 1e-3
    seed: int | None = None
    device: str | None = None
    dtype: str | None = None
    max_ring_mib: float | None = 2048.0
    enable_pruning: bool = False
    prune_interval_steps: int = 250
    usage_alpha: float = 0.01
    w_min: float = 0.01
    usage_min: float = 0.01
    k_min_out: int = 0
    k_min_in: int = 0
    max_prune_fraction_per_interval: float = 0.1
    pruning_verbose: bool = False
    enable_neurogenesis: bool = False
    growth_interval_steps: int = 500
    add_neurons_per_event: int = 4
    newborn_plasticity_multiplier: float = 1.5
    newborn_duration_steps: int = 250
    max_total_neurons: int = 20000
    neurogenesis_verbose: bool = False
    meta: Mapping[str, Any] | None = None


@runtime_checkable
class ISimulationEngine(Protocol):
    """High-level engine interface (build/reset/step/run)."""

    name: str

    def reset(self, *, config: SimulationConfig) -> None:
        ...

    def attach_monitors(self, monitors: Sequence[IMonitor]) -> None:
        ...

    def step(self) -> Mapping[str, Any]:
        """Advance the engine by one dt and return optional metrics."""
        ...

    def run(self, steps: int) -> None:
        ...
