"""Global scalar neuromodulator field."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from biosnn.biophysics.models._torch_utils import require_torch, resolve_device_dtype
from biosnn.contracts.modulators import IModulatorField, ModulatorKind, ModulatorRelease
from biosnn.contracts.tensor import Tensor


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
        self.params = params or GlobalScalarParams()

    def init_state(self, *, ctx: Any) -> GlobalScalarState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        levels = torch.zeros((len(self.kinds),), device=device, dtype=dtype)
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
            decay = torch.exp(torch.tensor(-dt / self.params.decay_tau, device=state.levels.device))
            state.levels.mul_(decay)

        for release in releases:
            if release.kind not in self.kinds:
                continue
            idx = self.kinds.index(release.kind)
            amount = release.amount
            if hasattr(amount, "sum"):
                delta = amount.sum().to(device=state.levels.device, dtype=state.levels.dtype)
            else:
                delta = torch.tensor(float(amount), device=state.levels.device, dtype=state.levels.dtype)
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
        if kind not in self.kinds:
            zeros = torch.zeros((positions.shape[0],), device=positions.device, dtype=positions.dtype)
            return cast(Tensor, zeros)
        idx = self.kinds.index(kind)
        level = state.levels[idx].to(device=positions.device, dtype=positions.dtype)
        expanded = level.expand(positions.shape[0])
        return cast(Tensor, expanded)

    def state_tensors(self, state: GlobalScalarState) -> dict[str, Tensor]:
        return {"levels": state.levels}


__all__ = ["GlobalScalarField", "GlobalScalarParams", "GlobalScalarState"]
