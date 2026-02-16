"""Global scalar neuromodulator field."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.modulators import IModulatorField, ModulatorKind, ModulatorRelease
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(frozen=True, slots=True)
class GlobalScalarParams:
    decay_tau: float = 1.0


@dataclass(slots=True)
class GlobalScalarState:
    levels: Tensor  # shape [K]


class GlobalScalarField(IModulatorField):
    """Global scalar field with exponential decay per kind."""

    name = "global_scalar"

    def __init__(
        self,
        *,
        kinds: tuple[ModulatorKind, ...] = (ModulatorKind.DOPAMINE,),
        params: GlobalScalarParams | None = None,
    ) -> None:
        self.kinds = kinds
        self._kind_to_idx = {kind: idx for idx, kind in enumerate(self.kinds)}
        self.params = params or GlobalScalarParams()
        # Cached decay scalar — lazily initialised on first step() call so
        # device/dtype match the actual state tensor.  Invalidated when dt
        # changes (rare).
        self._cached_decay: Tensor | None = None
        self._cached_decay_dt: float | None = None

    def init_state(self, *, ctx: Any) -> GlobalScalarState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        levels = torch.zeros((len(self.kinds),), device=device, dtype=dtype)
        self._cached_decay = None  # reset cache on new state
        return GlobalScalarState(levels=levels)

    def step(
        self,
        state: GlobalScalarState,
        *,
        releases: Sequence[ModulatorRelease],
        dt: float,
        t: float,
        ctx: Any,
    ) -> GlobalScalarState:
        torch = require_torch()
        if self.params.decay_tau > 0:
            # Re-use a cached scalar to avoid creating a new CUDA tensor
            # every step.  The cache is invalidated when dt changes.
            decay = self._cached_decay
            if decay is None or self._cached_decay_dt != dt:
                decay = torch.exp(
                    torch.tensor(-dt / self.params.decay_tau, device=state.levels.device)
                )
                self._cached_decay = decay
                self._cached_decay_dt = dt
            state.levels.mul_(decay)

        for release in releases:
            idx = self._kind_to_idx.get(release.kind)
            if idx is None:
                continue
            amount = release.amount
            if hasattr(amount, "sum"):
                delta = amount.sum().to(device=state.levels.device, dtype=state.levels.dtype)
            else:
                delta = torch.tensor(
                    float(amount), device=state.levels.device, dtype=state.levels.dtype
                )
            state.levels[idx] = state.levels[idx] + delta

        return state

    def sample_at(
        self,
        state: GlobalScalarState,
        *,
        positions: Tensor,
        kind: ModulatorKind,
        ctx: Any,
    ) -> Tensor:
        torch = require_torch()
        idx = self._kind_to_idx.get(kind)
        if idx is None:
            zeros = torch.zeros(
                (positions.shape[0],), device=positions.device, dtype=positions.dtype
            )
            return cast(Tensor, zeros)
        level = state.levels[idx].to(device=positions.device, dtype=positions.dtype)
        expanded = level.expand(positions.shape[0])
        return cast(Tensor, expanded)

    def state_tensors(self, state: GlobalScalarState) -> dict[str, Tensor]:
        return {"levels": state.levels}


__all__ = ["GlobalScalarField", "GlobalScalarParams", "GlobalScalarState"]
