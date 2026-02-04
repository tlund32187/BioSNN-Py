"""Three-factor Hebbian learning rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(frozen=True, slots=True)
class ThreeFactorHebbianParams:
    lr: float = 1e-3
    weight_decay: float = 0.0
    clamp_min: float | None = None
    clamp_max: float | None = None


@dataclass(slots=True)
class ThreeFactorHebbianState:
    last_mean_dw: Tensor


class ThreeFactorHebbianRule(ILearningRule):
    """Simple three-factor Hebbian update with optional dopamine gating."""

    name = "three_factor_hebbian"
    supports_sparse = True

    def __init__(self, params: ThreeFactorHebbianParams | None = None) -> None:
        self.params = params or ThreeFactorHebbianParams()

    def init_state(self, e: int, *, ctx: Any) -> ThreeFactorHebbianState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        return ThreeFactorHebbianState(last_mean_dw=torch.zeros((), device=device, dtype=dtype))

    def step(
        self,
        state: ThreeFactorHebbianState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: Any,
    ) -> tuple[ThreeFactorHebbianState, LearningStepResult]:
        torch = require_torch()

        pre = batch.pre_spikes
        post = batch.post_spikes

        if batch.modulators and ModulatorKind.DOPAMINE in batch.modulators:
            gate = batch.modulators[ModulatorKind.DOPAMINE]
        else:
            gate = torch.ones_like(batch.weights)

        if hasattr(pre, "to"):
            pre = pre.to(dtype=batch.weights.dtype)
        if hasattr(post, "to"):
            post = post.to(dtype=batch.weights.dtype)

        dw = pre * post
        dw = dw * gate if hasattr(dw, "mul") else dw * float(gate)

        dw = dw * self.params.lr
        if self.params.weight_decay:
            dw = dw - (self.params.weight_decay * batch.weights)

        if hasattr(dw, "mean"):
            state.last_mean_dw = dw.mean()
        else:
            state.last_mean_dw = torch.tensor(float(dw), device=batch.weights.device, dtype=batch.weights.dtype)

        return state, LearningStepResult(d_weights=dw)

    def state_tensors(self, state: ThreeFactorHebbianState) -> dict[str, Tensor]:
        return {"last_mean_dw": state.last_mean_dw}


__all__ = ["ThreeFactorHebbianParams", "ThreeFactorHebbianRule", "ThreeFactorHebbianState"]
