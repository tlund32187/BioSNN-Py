"""Learning rule contracts (STDP, metaplasticity, 3-factor, e-prop hooks)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from biosnn.contracts.modulators import ModulatorKind

if TYPE_CHECKING:  # pragma: no cover
    import torch
    Tensor = torch.Tensor
else:
    Tensor = Any


@dataclass(frozen=True, slots=True)
class LearningBatch:
    """Learning inputs for one update step.

    This is intentionally generic so it can support:
    - pair/triplet STDP using pre/post spikes and traces
    - 3-factor updates via modulator signals
    - eligibility propagation (e-prop) via additional traces

    Implementations can require particular keys inside extras.
    """

    pre_spikes: Tensor
    post_spikes: Tensor
    weights: Tensor  # shape [E] or [E, ...] depending on synapse representation
    modulators: Mapping[ModulatorKind, Tensor] | None = None
    extras: Mapping[str, Tensor] | None = None
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LearningStepResult:
    """Learning outputs (e.g., weight deltas)."""

    d_weights: Tensor
    extras: Mapping[str, Tensor] | None = None


@runtime_checkable
class ILearningRule(Protocol):
    """Learning rule operating on synapses/edges."""

    name: str

    def init_state(self, e: int, *, ctx: Any) -> Any:
        ...

    def step(
        self,
        state: Any,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: Any,
    ) -> tuple[Any, LearningStepResult]:
        ...

    def state_tensors(self, state: Any) -> Mapping[str, Tensor]:
        ...
