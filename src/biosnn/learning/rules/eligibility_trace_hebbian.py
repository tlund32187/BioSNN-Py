"""Three-factor Hebbian rule with optional sparse-friendly eligibility traces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(frozen=True, slots=True)
class EligibilityTraceHebbianParams:
    lr: float = 1e-3
    weight_decay: float = 0.0
    clamp_min: float | None = None
    clamp_max: float | None = None
    dopamine_scale: float = 1.0
    baseline_gate: float = 0.0
    enable_eligibility: bool = False
    tau_e: float = 0.05
    a_plus: float = 1.0
    a_minus: float = 0.0
    renorm_min_scale: float = 1e-6


@dataclass(slots=True)
class EligibilityTraceHebbianState:
    eligibility: Tensor
    eligibility_scale: Tensor
    last_mean_abs_dw: Tensor
    last_mean_abs_eligibility: Tensor


class EligibilityTraceHebbianRule(ILearningRule):
    """Three-factor Hebbian update with optional per-edge eligibility traces.

    Sparse mode:
    - If ``batch.extras['active_edges']`` is present and matches ``batch.weights``,
      only active edges are updated.
    - A scalar ``eligibility_scale`` is used so full-vector decay is avoided in sparse steps.
    """

    name = "eligibility_trace_hebbian"
    supports_sparse = True

    def __init__(self, params: EligibilityTraceHebbianParams | None = None) -> None:
        self.params = params or EligibilityTraceHebbianParams()

    def init_state(self, e: int, *, ctx: Any) -> EligibilityTraceHebbianState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        return EligibilityTraceHebbianState(
            eligibility=torch.zeros((e,), device=device, dtype=dtype),
            eligibility_scale=torch.ones((), device=device, dtype=dtype),
            last_mean_abs_dw=torch.zeros((), device=device, dtype=dtype),
            last_mean_abs_eligibility=torch.zeros((), device=device, dtype=dtype),
        )

    def step(
        self,
        state: EligibilityTraceHebbianState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: Any,
    ) -> tuple[EligibilityTraceHebbianState, LearningStepResult]:
        _ = (t, ctx)
        torch = require_torch()
        weights = batch.weights

        active_edges = _active_edges_from_batch(batch, like=state.eligibility)
        sparse_active = (
            active_edges is not None
            and int(batch.weights.numel()) == int(active_edges.numel())
            and int(batch.pre_spikes.numel()) == int(active_edges.numel())
            and int(batch.post_spikes.numel()) == int(active_edges.numel())
        )

        pre = _as_dtype(batch.pre_spikes, like=weights)
        post = _as_dtype(batch.post_spikes, like=weights)
        gate = _resolve_gate(batch=batch, like=weights)
        gate = gate * float(self.params.dopamine_scale) + float(self.params.baseline_gate)

        if self.params.enable_eligibility:
            decay = math.exp(-float(dt) / max(float(self.params.tau_e), 1e-12))
            state.eligibility_scale.mul_(decay)
            _renormalize_if_needed(state=state, min_scale=float(self.params.renorm_min_scale))

            stdp_delta = float(self.params.a_plus) * pre * post
            stdp_delta.add_(pre * (1.0 - post), alpha=-float(self.params.a_minus))

            if sparse_active:
                assert active_edges is not None
                prev = state.eligibility.index_select(0, active_edges)
                elig_active = prev * state.eligibility_scale
                elig_active.add_(stdp_delta)
                new_raw = elig_active / state.eligibility_scale
                state.eligibility.index_copy_(0, active_edges, new_raw)
                local_term = elig_active
                state.last_mean_abs_eligibility = (
                    elig_active.abs().mean()
                    if elig_active.numel()
                    else torch.zeros_like(state.last_mean_abs_eligibility)
                )
            else:
                if state.eligibility.shape != weights.shape:
                    raise ValueError(
                        "Eligibility shape must match dense batch weights when sparse metadata is absent: "
                        f"{tuple(state.eligibility.shape)} != {tuple(weights.shape)}"
                    )
                elig_full = state.eligibility * state.eligibility_scale
                elig_full.add_(stdp_delta)
                state.eligibility.copy_(elig_full / state.eligibility_scale)
                local_term = elig_full
                state.last_mean_abs_eligibility = (
                    elig_full.abs().mean()
                    if elig_full.numel()
                    else torch.zeros_like(state.last_mean_abs_eligibility)
                )
        else:
            local_term = pre * post
            state.last_mean_abs_eligibility = torch.zeros_like(state.last_mean_abs_eligibility)

        dw = local_term * gate
        dw.mul_(float(self.params.lr))
        if self.params.weight_decay:
            dw.add_(weights, alpha=-float(self.params.weight_decay) * float(self.params.lr))

        dw = _apply_weight_bounds(
            dw=dw,
            weights=weights,
            w_min=self.params.clamp_min,
            w_max=self.params.clamp_max,
        )

        state.last_mean_abs_dw = (
            dw.abs().mean() if dw.numel() else torch.zeros_like(state.last_mean_abs_dw)
        )
        extras: dict[str, Tensor] = {
            "mean_abs_dw": state.last_mean_abs_dw,
            "mean_abs_eligibility": state.last_mean_abs_eligibility,
            "eligibility_scale": state.eligibility_scale,
        }
        if sparse_active and active_edges is not None:
            extras["active_edges"] = active_edges

        return state, LearningStepResult(d_weights=dw, extras=extras)

    def state_tensors(self, state: EligibilityTraceHebbianState) -> dict[str, Tensor]:
        return {
            "eligibility": state.eligibility,
            "eligibility_scale": state.eligibility_scale,
            "mean_abs_dw": state.last_mean_abs_dw,
            "mean_abs_eligibility": state.last_mean_abs_eligibility,
        }


def _as_dtype(value: Tensor, *, like: Tensor) -> Tensor:
    if value.dtype == like.dtype and value.device == like.device:
        return value
    return value.to(device=like.device, dtype=like.dtype)


def _active_edges_from_batch(batch: LearningBatch, *, like: Tensor) -> Tensor | None:
    if not batch.extras or "active_edges" not in batch.extras:
        return None
    active = batch.extras["active_edges"]
    torch = require_torch()
    if active.device != like.device or active.dtype != torch.long:
        return active.to(device=like.device, dtype=torch.long)
    return active


def _resolve_gate(*, batch: LearningBatch, like: Tensor) -> Tensor:
    torch = require_torch()
    if not batch.modulators or ModulatorKind.DOPAMINE not in batch.modulators:
        return cast(Tensor, torch.ones_like(like))

    dopamine = _as_dtype(batch.modulators[ModulatorKind.DOPAMINE], like=like)
    if dopamine.numel() == like.numel():
        return dopamine
    if dopamine.numel() == 1:
        return cast(Tensor, dopamine.reshape(()).expand_as(like))
    if batch.extras and "post_idx" in batch.extras:
        post_idx = cast(Tensor, batch.extras["post_idx"]).to(
            device=like.device, dtype=torch.long
        )
        if post_idx.numel() == like.numel():
            return cast(Tensor, dopamine.index_select(0, post_idx))

    raise ValueError(
        "Dopamine tensor must be scalar, edge-sized, or post-sized with post_idx mapping."
    )


def _renormalize_if_needed(*, state: EligibilityTraceHebbianState, min_scale: float) -> None:
    torch = require_torch()
    eps = torch.full_like(state.eligibility_scale, float(min_scale))
    renorm = state.eligibility_scale.abs() < eps
    factor = torch.where(renorm, state.eligibility_scale, torch.ones_like(state.eligibility_scale))
    state.eligibility.mul_(factor)
    state.eligibility_scale.copy_(
        torch.where(renorm, torch.ones_like(state.eligibility_scale), state.eligibility_scale)
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
    "EligibilityTraceHebbianParams",
    "EligibilityTraceHebbianRule",
    "EligibilityTraceHebbianState",
]
