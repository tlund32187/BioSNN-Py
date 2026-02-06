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
