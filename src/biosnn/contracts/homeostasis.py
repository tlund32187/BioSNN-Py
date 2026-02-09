"""Homeostasis contracts for activity stabilization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from biosnn.contracts.neurons import INeuronModel, StepContext
from biosnn.contracts.tensor import Tensor


@dataclass(frozen=True, slots=True)
class HomeostasisPopulation:
    """Population reference passed to homeostasis initialization."""

    name: str
    model: INeuronModel
    state: Any
    n: int


@runtime_checkable
class IHomeostasisRule(Protocol):
    """Population-level homeostasis interface."""

    name: str

    def init(
        self,
        populations: Sequence[HomeostasisPopulation],
        *,
        device: Any,
        dtype: Any,
        ctx: StepContext,
    ) -> None:
        """Initialize internal rule state from population references."""
        ...

    def step(
        self,
        spikes_by_pop: Mapping[str, Tensor],
        *,
        dt: float,
        ctx: StepContext,
    ) -> Mapping[str, Tensor]:
        """Update homeostasis state from population spikes."""
        ...

    def state_tensors(self) -> Mapping[str, Tensor]:
        """Expose tensors for monitoring/debugging."""
        ...


__all__ = ["HomeostasisPopulation", "IHomeostasisRule"]

