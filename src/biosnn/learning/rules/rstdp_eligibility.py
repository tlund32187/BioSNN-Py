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
    lazy_decay_enabled: bool = True
    lazy_decay_step_dtype: str = "int32"
    lazy_decay_dense: bool = False
    tau_pre: float = 0.0
    tau_post: float = 0.0


@dataclass(slots=True)
class RStdpEligibilityState:
    eligibility: Tensor
    last_update_step: Tensor | None
    step: int
    last_mean_abs_dw: Tensor
    last_mean_abs_eligibility: Tensor
    pre_trace: Tensor | None = None
    post_trace: Tensor | None = None


class RStdpEligibilityRule(ILearningRule):
    """Three-factor eligibility rule with dopamine-gated updates."""

    name = "rstdp_eligibility"

    def __init__(self, params: RStdpEligibilityParams | None = None) -> None:
        self.params = params or RStdpEligibilityParams()
        # Trace-based STDP requires dense learning: sparse mode only
        # includes edges where pre spiked THIS step, but traces carry
        # information from recent (not current) spikes.  Dense mode
        # must be used so that decayed traces contribute to eligibility.
        traces_enabled = self.params.tau_pre > 0.0 and self.params.tau_post > 0.0
        self.supports_sparse: bool = not traces_enabled

    def init_state(self, e: int, *, ctx: Any) -> RStdpEligibilityState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        last_update_step = None
        if self.params.lazy_decay_enabled:
            step_dtype = _resolve_step_dtype(torch, self.params.lazy_decay_step_dtype)
            last_update_step = torch.zeros((e,), device=device, dtype=step_dtype)
        pre_trace = None
        post_trace = None
        if self.params.tau_pre > 0.0 and self.params.tau_post > 0.0:
            pre_trace = torch.zeros((e,), device=device, dtype=dtype)
            post_trace = torch.zeros((e,), device=device, dtype=dtype)
        return RStdpEligibilityState(
            eligibility=torch.zeros((e,), device=device, dtype=dtype),
            last_update_step=last_update_step,
            step=0,
            last_mean_abs_dw=torch.zeros((), device=device, dtype=dtype),
            last_mean_abs_eligibility=torch.zeros((), device=device, dtype=dtype),
            pre_trace=pre_trace,
            post_trace=post_trace,
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
        pre = _as_dtype(batch.pre_spikes, like=weights)
        post = _as_dtype(batch.post_spikes, like=weights)
        active_edges = _active_edges_from_batch(batch, like=state.eligibility)
        sparse_mode = active_edges is not None

        tau_e = float(self.params.tau_e)
        if tau_e <= 0.0:
            raise ValueError("tau_e must be > 0 for eligibility decay.")
        decay = math.exp(-float(dt) / tau_e)

        use_traces = state.pre_trace is not None and state.post_trace is not None
        if use_traces:
            assert state.pre_trace is not None
            assert state.post_trace is not None
            decay_pre = math.exp(-float(dt) / float(self.params.tau_pre))
            decay_post = math.exp(-float(dt) / float(self.params.tau_post))
            # Decay ALL trace entries (dense) each step
            state.pre_trace.mul_(decay_pre)
            state.post_trace.mul_(decay_post)
            if sparse_mode:
                assert active_edges is not None
                # Read traces for active edges BEFORE adding current spikes
                pre_t = state.pre_trace.index_select(0, active_edges)
                post_t = state.post_trace.index_select(0, active_edges)
                # Compute inc using traces (temporal STDP window)
                inc = float(self.params.a_plus) * (pre_t * post)
                inc.add_(pre * post_t, alpha=-float(self.params.a_minus))
                # Update traces with current spikes for active edges
                state.pre_trace.index_add_(0, active_edges, pre)
                state.post_trace.index_add_(0, active_edges, post)
            else:
                # Compute inc using traces BEFORE adding current spikes
                inc = float(self.params.a_plus) * (state.pre_trace * post)
                inc.add_(pre * state.post_trace, alpha=-float(self.params.a_minus))
                # Update traces with current spikes
                state.pre_trace.add_(pre)
                state.post_trace.add_(post)
        else:
            # Original instantaneous coincidence mode
            inc = float(self.params.a_plus) * (pre * post)
            inc.add_(pre * (1.0 - post), alpha=-float(self.params.a_minus))

        if sparse_mode:
            assert active_edges is not None
            if active_edges.numel() != weights.numel():
                raise ValueError(
                    "Sparse learning requires active_edges shape to match batch weights shape: "
                    f"{tuple(active_edges.shape)} vs {tuple(weights.shape)}"
                )
            if self.params.lazy_decay_enabled:
                if state.last_update_step is None:
                    raise ValueError(
                        "Lazy decay is enabled but RStdpEligibilityState.last_update_step is missing."
                    )
                last = state.last_update_step.index_select(0, active_edges)
                delta_steps = (float(state.step) - last).to(dtype=state.eligibility.dtype)
                decay_active = torch.exp(delta_steps * (-float(dt) / tau_e))
                elig_active = state.eligibility.index_select(0, active_edges)
                elig_active.mul_(decay_active)
                elig_active.add_(inc)
                state.eligibility.index_copy_(0, active_edges, elig_active)
                state.last_update_step.index_copy_(
                    0,
                    active_edges,
                    torch.full_like(last, fill_value=int(state.step)),
                )
            else:
                state.eligibility.mul_(decay)
                elig_active = state.eligibility.index_select(0, active_edges)
                elig_active.add_(inc)
                state.eligibility.index_copy_(0, active_edges, elig_active)
            e_local = elig_active
            weights_local = weights
        else:
            state.eligibility.mul_(decay)
            if state.eligibility.shape != weights.shape:
                raise ValueError(
                    "Dense learning requires eligibility shape to match weights shape: "
                    f"{tuple(state.eligibility.shape)} != {tuple(weights.shape)}"
                )
            state.eligibility.add_(inc)
            e_local = state.eligibility
            weights_local = weights
            if state.last_update_step is not None:
                state.last_update_step.fill_(int(state.step))

        dopamine = _resolve_dopamine(batch=batch, like=weights_local)
        effective_dopamine = dopamine * float(self.params.dopamine_scale) + float(
            self.params.baseline
        )

        dw = e_local * effective_dopamine
        dw.mul_(float(self.params.lr))
        if self.params.weight_decay:
            dw.add_(weights_local, alpha=-float(self.params.weight_decay) * float(self.params.lr))

        dw = _apply_weight_bounds(
            dw=dw,
            weights=weights_local,
            w_min=self.params.w_min,
            w_max=self.params.w_max,
        )

        state.last_mean_abs_dw = (
            dw.abs().mean() if dw.numel() else torch.zeros_like(state.last_mean_abs_dw)
        )
        state.last_mean_abs_eligibility = (
            e_local.abs().mean()
            if e_local.numel()
            else torch.zeros_like(state.last_mean_abs_eligibility)
        )
        extras: dict[str, Tensor] = {
            "mean_abs_dw": state.last_mean_abs_dw,
            "mean_abs_eligibility": state.last_mean_abs_eligibility,
        }
        if sparse_mode and active_edges is not None:
            extras["active_edges"] = active_edges
        state.step += 1
        return state, LearningStepResult(d_weights=dw, extras=extras)

    def state_tensors(self, state: RStdpEligibilityState) -> dict[str, Tensor]:
        d: dict[str, Tensor] = {
            "eligibility": state.eligibility,
            "mean_abs_dw": state.last_mean_abs_dw,
            "mean_abs_eligibility": state.last_mean_abs_eligibility,
        }
        if state.pre_trace is not None:
            d["pre_trace"] = state.pre_trace
        if state.post_trace is not None:
            d["post_trace"] = state.post_trace
        return d


def _as_dtype(value: Tensor, *, like: Tensor) -> Tensor:
    if value.dtype == like.dtype and value.device == like.device:
        return value
    return value.to(device=like.device, dtype=like.dtype)


def _resolve_step_dtype(torch: Any, raw: str) -> Any:
    token = str(raw).split(".", 1)[-1].strip().lower()
    if token in {"int32", "int64"}:
        return getattr(torch, token)
    return torch.int32


def _active_edges_from_batch(batch: LearningBatch, *, like: Tensor) -> Tensor | None:
    if not batch.extras or "active_edges" not in batch.extras:
        return None
    torch = require_torch()
    active_edges = batch.extras["active_edges"]
    if active_edges.device == like.device and active_edges.dtype == torch.long:
        return active_edges
    return active_edges.to(device=like.device, dtype=torch.long)


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
