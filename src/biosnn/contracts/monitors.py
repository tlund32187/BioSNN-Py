"""Monitoring contracts.

Monitors receive structured step events and may write CSV, emit live graphs,
compute metrics, or collect debugging artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, runtime_checkable

from biosnn.contracts.tensor import Tensor

Scalar: TypeAlias = float | int | Tensor  # noqa: UP040


@dataclass(frozen=True, slots=True)
class StepEvent:
    """A single simulation step event."""

    t: float
    dt: float
    spikes: Tensor | None = None
    tensors: Mapping[str, Tensor] | None = None
    scalars: Mapping[str, Scalar] | None = None
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MonitorRequirements:
    """Monitor-declared payload needs for StepEvent construction."""

    needs_spikes: bool = False
    needs_v_soma: bool = False
    needs_projection_weights: bool = False
    needs_projection_drive: bool = False
    needs_synapse_state: bool = False
    needs_modulators: bool = False
    needs_learning_state: bool = False
    needs_homeostasis_state: bool = False
    needs_population_state: bool = False
    needs_scalars: bool = False
    needs_population_slices: bool = False

    @classmethod
    def all(cls) -> MonitorRequirements:
        return cls(
            needs_spikes=True,
            needs_v_soma=True,
            needs_projection_weights=True,
            needs_projection_drive=True,
            needs_synapse_state=True,
            needs_modulators=True,
            needs_learning_state=True,
            needs_homeostasis_state=True,
            needs_population_state=True,
            needs_scalars=True,
            needs_population_slices=True,
        )

    @classmethod
    def none(cls) -> MonitorRequirements:
        return cls()

    def merge(self, other: MonitorRequirements) -> MonitorRequirements:
        return MonitorRequirements(
            needs_spikes=self.needs_spikes or other.needs_spikes,
            needs_v_soma=self.needs_v_soma or other.needs_v_soma,
            needs_projection_weights=self.needs_projection_weights or other.needs_projection_weights,
            needs_projection_drive=self.needs_projection_drive or other.needs_projection_drive,
            needs_synapse_state=self.needs_synapse_state or other.needs_synapse_state,
            needs_modulators=self.needs_modulators or other.needs_modulators,
            needs_learning_state=self.needs_learning_state or other.needs_learning_state,
            needs_homeostasis_state=self.needs_homeostasis_state or other.needs_homeostasis_state,
            needs_population_state=self.needs_population_state or other.needs_population_state,
            needs_scalars=self.needs_scalars or other.needs_scalars,
            needs_population_slices=self.needs_population_slices or other.needs_population_slices,
        )


@runtime_checkable
class IMonitor(Protocol):
    """Observer of simulation steps."""

    name: str

    def on_step(self, event: StepEvent) -> None:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class IMonitorRequirements(Protocol):
    """Optional monitor StepEvent requirements hook."""

    def requirements(self) -> MonitorRequirements:
        ...


@runtime_checkable
class IMonitorCompilationRequirements(Protocol):
    """Optional hook for monitor-driven engine artifact requirements."""

    def compilation_requirements(self) -> Mapping[str, bool]:
        """Return monitor requirements consumed by engine compile planning.

        Expected keys (if applicable):
        - wants_fused_sparse
        - wants_by_delay_sparse
        - wants_bucket_edge_mapping
        - wants_weights_snapshot_each_step
        - wants_projection_drive_tensor
        """
        ...
