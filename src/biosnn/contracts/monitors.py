"""Monitoring contracts.

Monitors receive structured step events and may write CSV, emit live graphs,
compute metrics, or collect debugging artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    import torch
    Tensor = torch.Tensor
else:
    Tensor = Any


@dataclass(frozen=True, slots=True)
class StepEvent:
    """A single simulation step event."""

    t: float
    dt: float
    spikes: Tensor | None = None
    tensors: Mapping[str, Tensor] | None = None
    scalars: Mapping[str, float] | None = None
    meta: Mapping[str, Any] | None = None


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
