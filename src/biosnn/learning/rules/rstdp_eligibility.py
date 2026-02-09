"""Reward-modulated STDP with local per-edge eligibility traces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(frozen=True, slots=True)
class RStdpEligibilityParams:
    lr: float = 0.08
    tau_e: float = 0.05
    a_plus: float = 1.0
    a_minus: float = 0.6
    w_min: float | None = -1.0
    w_max: float | None = 1.0
    weight_decay: float = 1e-4
    dopamine_scale: float = 1.0
    baseline: float = 0.0


@dataclass(slots=True)
class RStdpEligibilityState:
    eligibility: Tensor
    last_mean_abs_dw: Tensor
    last_mean_abs_eligibility: Tensor


class RStdpEligibilityRule(ILearningRule):
    """Three-factor eligibility rule with dopamine-gated updates."""

    name = "rstdp_eligibility"
    supports_sparse = True

    def __init__(self, params: RStdpEligibilityParams | None = None) -> None:
        self.params = params or RStdpEligibilityParams()

    def init_state(self, e: int, *, ctx: Any) -> RStdpEligibilityState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        return RStdpEligibilityState(
            eligibility=torch.zeros((e,), device=device, dtype=dtype),
            last_mean_abs_dw=torch.zeros((), device=device, dtype=dtype),
            last_mean_abs_eligibility=torch.zeros((), device=device, dtype=dtype),
        )

    def step(
        self,
        state: RStdpEligibilityState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: Any,
    ) -> tuple[RStdpEligibilityState, LearningStepResult]:
        _ = (t, ctx)
        torch = require_torch()
        weights = batch.weights
        if state.eligibility.shape != weights.shape:
            raise ValueError(
                "Eligibility shape must match weights shape: "
                f"{tuple(state.eligibility.shape)} != {tuple(weights.shape)}"
            )

        pre = _as_dtype(batch.pre_spikes, like=weights)
        post = _as_dtype(batch.post_spikes, like=weights)

        decay = math.exp(-float(dt) / max(float(self.params.tau_e), 1e-12))
        state.eligibility.mul_(decay)
        state.eligibility.add_(float(self.params.a_plus) * pre * post)
        state.eligibility.add_(-float(self.params.a_minus) * pre * (1.0 - post))

        dopamine = _resolve_dopamine(batch=batch, like=weights)
        effective_dopamine = dopamine * float(self.params.dopamine_scale) + float(self.params.baseline)

        dw = state.eligibility * effective_dopamine
        dw.mul_(float(self.params.lr))
        if self.params.weight_decay:
            dw.add_(weights, alpha=-float(self.params.weight_decay) * float(self.params.lr))

        dw = _apply_weight_bounds(dw=dw, weights=weights, w_min=self.params.w_min, w_max=self.params.w_max)

        state.last_mean_abs_dw = dw.abs().mean() if dw.numel() else torch.zeros_like(state.last_mean_abs_dw)
        state.last_mean_abs_eligibility = (
            state.eligibility.abs().mean()
            if state.eligibility.numel()
            else torch.zeros_like(state.last_mean_abs_eligibility)
        )

        return state, LearningStepResult(
            d_weights=dw,
            extras={
                "mean_abs_dw": state.last_mean_abs_dw,
                "mean_abs_eligibility": state.last_mean_abs_eligibility,
            },
        )

    def state_tensors(self, state: RStdpEligibilityState) -> dict[str, Tensor]:
        return {
            "eligibility": state.eligibility,
            "mean_abs_dw": state.last_mean_abs_dw,
            "mean_abs_eligibility": state.last_mean_abs_eligibility,
        }


def _as_dtype(value: Tensor, *, like: Tensor) -> Tensor:
    if value.dtype == like.dtype and value.device == like.device:
        return value
    return value.to(device=like.device, dtype=like.dtype)


def _resolve_dopamine(*, batch: LearningBatch, like: Tensor) -> Tensor:
    torch = require_torch()
    if not batch.modulators or ModulatorKind.DOPAMINE not in batch.modulators:
        return cast(Tensor, torch.zeros_like(like))

    dopamine = _as_dtype(batch.modulators[ModulatorKind.DOPAMINE], like=like)
    if dopamine.numel() == like.numel():
        return dopamine
    if dopamine.numel() == 1:
        return cast(Tensor, torch.full_like(like, float(dopamine.reshape(()).item())))

    if batch.extras and "post_idx" in batch.extras:
        post_idx = cast(Tensor, batch.extras["post_idx"]).to(device=like.device, dtype=torch.long)
        if int(dopamine.numel()) > 0 and int(post_idx.numel()) == int(like.numel()):
            max_idx = int(post_idx.max().item()) if post_idx.numel() else -1
            if max_idx < int(dopamine.numel()):
                return cast(Tensor, dopamine.index_select(0, post_idx))

    raise ValueError(
        "Dopamine tensor must be scalar, edge-sized, or post-sized with post_idx mapping."
    )


def _apply_weight_bounds(
    *,
    dw: Tensor,
    weights: Tensor,
    w_min: float | None,
    w_max: float | None,
) -> Tensor:
    if w_min is None and w_max is None:
        return dw
    candidate = weights + dw
    candidate = candidate.clamp(min=w_min, max=w_max)
    return candidate - weights


__all__ = [
    "RStdpEligibilityParams",
    "RStdpEligibilityRule",
    "RStdpEligibilityState",
]
