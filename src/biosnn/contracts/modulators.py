"""Neuromodulator contracts.

A modulator field is responsible for tracking diffusion/decay dynamics and for
providing sampled modulator values at neuron/synapse locations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from biosnn.contracts.tensor import Tensor


class ModulatorKind(StrEnum):
    """Canonical modulator types.

    Keep this small and stable; add new modulators cautiously because they tend
    to leak into many subsystems (plasticity, excitability, energy, monitoring).
    """

    DOPAMINE = "dopamine"
    ACETYLCHOLINE = "acetylcholine"
    NORADRENALINE = "noradrenaline"
    SEROTONIN = "serotonin"


@dataclass(frozen=True, slots=True)
class ModulatorRelease:
    """A release event (e.g., DA burst) emitted by some controller/agent.

    Interpretation is left to the field implementation:
    - positions: (R, 3) xyz positions (meters or arbitrary units, but be consistent)
    - amount: (R,) scalar intensity per release site
    - radius: scalar (optional) influence radius
    """

    kind: ModulatorKind
    positions: Tensor
    amount: Tensor
    radius: float | None = None
    meta: Mapping[str, Any] | None = None


@runtime_checkable
class IModulatorField(Protocol):
    """Diffusion/decay field for neuromodulators (DA/ACh/NA/etc.)."""

    name: str

    def init_state(self, *, ctx: Any) -> Any:
        """Create internal field state (grids, particles, caches, etc.)."""
        ...

    def step(
        self,
        state: Any,
        *,
        releases: Sequence[ModulatorRelease],
        dt: float,
        t: float,
        ctx: Any,
    ) -> Any:
        """Advance field dynamics by one timestep."""
        ...

    def sample_at(
        self,
        state: Any,
        *,
        positions: Tensor,
        kind: ModulatorKind,
        ctx: Any,
    ) -> Tensor:
        """Sample the modulator value at given xyz positions."""
        ...

    def state_tensors(self, state: Any) -> Mapping[str, Tensor]:
        """Expose tensors for monitoring/debugging (e.g., grids, stats)."""
        ...
