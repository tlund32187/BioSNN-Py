"""Sparse-capable three-factor eligibility STDP rule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(frozen=True, slots=True)
class ThreeFactorEligibilityStdpParams:
    lr: float = 1e-3
    tau_e: float = 0.05
    a_plus: float = 1.0
    a_minus: float = 0.6
    weight_decay: float = 0.0
    clamp_min: float | None = None
    clamp_max: float | None = None
    modulator_kind: ModulatorKind = ModulatorKind.DOPAMINE


@dataclass(slots=True)
class ThreeFactorEligibilityStdpState:
    eligibility: Tensor
    last_mean_dw: Tensor


class ThreeFactorEligibilityStdpRule(ILearningRule):
    """Eligibility STDP with optional sparse active-edge updates.

    Design:
    - Keep eligibility as full per-edge tensor ``[E]`` on-device.
    - Dense mode: update full ``eligibility``.
    - Sparse mode: decay full eligibility, then update only ``active_edges``.
    """

    name = "three_factor_eligibility_stdp"
    supports_sparse = True

    def __init__(self, params: ThreeFactorEligibilityStdpParams | None = None) -> None:
        self.params = params or ThreeFactorEligibilityStdpParams()

    def init_state(self, e: int, *, ctx: Any) -> ThreeFactorEligibilityStdpState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        return ThreeFactorEligibilityStdpState(
            eligibility=torch.zeros((e,), device=device, dtype=dtype),
            last_mean_dw=torch.zeros((), device=device, dtype=dtype),
        )

    def step(
        self,
        state: ThreeFactorEligibilityStdpState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: Any,
    ) -> tuple[ThreeFactorEligibilityStdpState, LearningStepResult]:
        _ = (t, ctx)
        torch = require_torch()
        weights = batch.weights
        pre = _as_dtype(batch.pre_spikes, like=weights)
        post = _as_dtype(batch.post_spikes, like=weights)

        active_edges = _active_edges_from_batch(batch, like=state.eligibility)
        sparse_mode = active_edges is not None

        decay = math.exp(-float(dt) / max(float(self.params.tau_e), 1e-12))
        state.eligibility.mul_(decay)

        inc = float(self.params.a_plus) * (pre * post)
        inc.add_(pre * (1.0 - post), alpha=-float(self.params.a_minus))

        if sparse_mode:
            assert active_edges is not None
            if state.eligibility.numel() == 0 and active_edges.numel() > 0:
                raise ValueError("Eligibility state is empty while sparse active edges are present.")
            if active_edges.numel() != weights.numel():
                raise ValueError(
                    "Sparse learning requires active_edges shape to match batch weights shape: "
                    f"{tuple(active_edges.shape)} vs {tuple(weights.shape)}"
                )
            elig_active = state.eligibility.index_select(0, active_edges)
            elig_active.add_(inc)
            state.eligibility.index_copy_(0, active_edges, elig_active)
            e_local = elig_active
        else:
            if state.eligibility.shape != weights.shape:
                raise ValueError(
                    "Dense learning requires eligibility shape to match weights shape: "
                    f"{tuple(state.eligibility.shape)} != {tuple(weights.shape)}"
                )
            state.eligibility.add_(inc)
            e_local = state.eligibility

        gate = _resolve_gate(batch=batch, like=weights, kind=self.params.modulator_kind)
        dw = e_local * gate
        dw.mul_(float(self.params.lr))
        if self.params.weight_decay:
            dw.add_(weights, alpha=-float(self.params.weight_decay))

        dw = _apply_weight_bounds(
            dw=dw,
            weights=weights,
            clamp_min=self.params.clamp_min,
            clamp_max=self.params.clamp_max,
        )
        state.last_mean_dw = dw.mean() if dw.numel() else torch.zeros_like(state.last_mean_dw)

        extras: dict[str, Tensor] = {"mean_dw": state.last_mean_dw}
        if sparse_mode and active_edges is not None:
            extras["active_edges"] = active_edges
        return state, LearningStepResult(d_weights=dw, extras=extras)

    def state_tensors(self, state: ThreeFactorEligibilityStdpState) -> dict[str, Tensor]:
        return {
            "eligibility": state.eligibility,
            "mean_dw": state.last_mean_dw,
        }


def _as_dtype(value: Tensor, *, like: Tensor) -> Tensor:
    if value.dtype == like.dtype and value.device == like.device:
        return value
    return value.to(device=like.device, dtype=like.dtype)


def _active_edges_from_batch(batch: LearningBatch, *, like: Tensor) -> Tensor | None:
    if not batch.extras or "active_edges" not in batch.extras:
        return None
    torch = require_torch()
    active_edges = batch.extras["active_edges"]
    if active_edges.device == like.device and active_edges.dtype == torch.long:
        return active_edges
    return active_edges.to(device=like.device, dtype=torch.long)


def _resolve_gate(*, batch: LearningBatch, like: Tensor, kind: ModulatorKind) -> Tensor:
    torch = require_torch()
    if not batch.modulators or kind not in batch.modulators:
        return cast(Tensor, torch.zeros_like(like))

    gate = _as_dtype(batch.modulators[kind], like=like)
    if gate.numel() == like.numel():
        return gate
    if gate.numel() == 1:
        return cast(Tensor, gate.reshape(()).expand_as(like))
    if batch.extras and "post_idx" in batch.extras and batch.extras["post_idx"].numel() == like.numel():
        post_idx = batch.extras["post_idx"].to(device=like.device, dtype=torch.long)
        try:
            return cast(Tensor, gate.index_select(0, post_idx))
        except RuntimeError as exc:
            raise ValueError(
                "Modulator tensor could not be projected by post_idx into edge-space."
            ) from exc
    raise ValueError(
        "Modulator tensor must be scalar, edge-sized, or post-sized with post_idx mapping."
    )


def _apply_weight_bounds(
    *,
    dw: Tensor,
    weights: Tensor,
    clamp_min: float | None,
    clamp_max: float | None,
) -> Tensor:
    if clamp_min is None and clamp_max is None:
        return dw
    candidate = (weights + dw).clamp(min=clamp_min, max=clamp_max)
    return candidate - weights


__all__ = [
    "ThreeFactorEligibilityStdpParams",
    "ThreeFactorEligibilityStdpRule",
    "ThreeFactorEligibilityStdpState",
]
