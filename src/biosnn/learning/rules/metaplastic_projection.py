"""Metaplastic wrapper that scales learning-rate by reward variance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(frozen=True, slots=True)
class MetaplasticProjectionParams:
    enabled: bool = False
    reward_beta: float = 0.01
    variance_gain: float = 1.0
    eta_init: float = 1.0
    eta_min: float = 0.25
    eta_max: float = 2.0


@dataclass(slots=True)
class MetaplasticProjectionState:
    inner_state: Any
    eta: Tensor
    reward_mean: Tensor
    reward_var: Tensor
    last_reward: Tensor


class MetaplasticProjectionRule(ILearningRule):
    """Wraps a base learning rule and applies a projection-wise metaplastic scalar."""

    name = "metaplastic_projection"
    supports_sparse = True

    def __init__(
        self,
        base_rule: ILearningRule,
        params: MetaplasticProjectionParams | None = None,
    ) -> None:
        self.base_rule = base_rule
        self.params = params or MetaplasticProjectionParams()
        self.name = f"{getattr(base_rule, 'name', 'learning_rule')}_metaplastic"
        self.supports_sparse = bool(getattr(base_rule, "supports_sparse", False))

    def init_state(self, e: int, *, ctx: Any) -> MetaplasticProjectionState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        inner = self.base_rule.init_state(e, ctx=ctx)
        eta_init = float(self.params.eta_init)
        return MetaplasticProjectionState(
            inner_state=inner,
            eta=torch.full((), eta_init, device=device, dtype=dtype),
            reward_mean=torch.zeros((), device=device, dtype=dtype),
            reward_var=torch.zeros((), device=device, dtype=dtype),
            last_reward=torch.zeros((), device=device, dtype=dtype),
        )

    def step(
        self,
        state: MetaplasticProjectionState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: Any,
    ) -> tuple[MetaplasticProjectionState, LearningStepResult]:
        state.inner_state, inner_result = self.base_rule.step(
            state.inner_state,
            batch,
            dt=dt,
            t=t,
            ctx=ctx,
        )

        if self.params.enabled:
            reward = _resolve_reward(batch=batch, like=state.eta)
            beta = float(self.params.reward_beta)
            delta = reward - state.reward_mean
            state.reward_mean.add_(delta, alpha=beta)
            state.reward_var.mul_(1.0 - beta)
            state.reward_var.add_(delta * delta, alpha=beta)

            denom = 1.0 + float(self.params.variance_gain) * state.reward_var
            target_eta = denom.reciprocal()
            state.eta.copy_(target_eta.clamp(min=self.params.eta_min, max=self.params.eta_max))
            state.last_reward.copy_(reward)
        else:
            state.eta.fill_(float(self.params.eta_init))
            state.reward_mean.zero_()
            state.reward_var.zero_()
            state.last_reward.zero_()

        scaled_dw = inner_result.d_weights * state.eta
        extras = dict(inner_result.extras) if inner_result.extras is not None else {}
        extras.update(
            {
                "eta": state.eta,
                "reward_mean": state.reward_mean,
                "reward_var": state.reward_var,
                "reward": state.last_reward,
            }
        )
        return state, LearningStepResult(d_weights=scaled_dw, extras=extras)

    def state_tensors(self, state: MetaplasticProjectionState) -> Mapping[str, Tensor]:
        tensors = dict(self.base_rule.state_tensors(state.inner_state))
        tensors.update(
            {
                "eta": state.eta,
                "reward_mean": state.reward_mean,
                "reward_var": state.reward_var,
                "reward": state.last_reward,
            }
        )
        return tensors


def _resolve_reward(*, batch: LearningBatch, like: Tensor) -> Tensor:
    torch = require_torch()
    if batch.meta and "reward" in batch.meta:
        return _as_scalar_tensor(batch.meta["reward"], like=like)
    if batch.extras and "reward" in batch.extras:
        return _as_scalar_tensor(batch.extras["reward"], like=like)
    if batch.modulators and ModulatorKind.DOPAMINE in batch.modulators:
        dop = batch.modulators[ModulatorKind.DOPAMINE].to(device=like.device, dtype=like.dtype)
        if dop.numel() == 0:
            return cast(Tensor, torch.zeros_like(like))
        return cast(Tensor, dop.mean().reshape_as(like))
    return cast(Tensor, torch.zeros_like(like))


def _as_scalar_tensor(value: Any, *, like: Tensor) -> Tensor:
    torch = require_torch()
    if hasattr(value, "to"):
        cast_value = cast(Tensor, value).to(device=like.device, dtype=like.dtype)
        if cast_value.numel() == 0:
            return cast(Tensor, torch.zeros_like(like))
        return cast(Tensor, cast_value.mean().reshape_as(like))
    if isinstance(value, bool):
        return cast(Tensor, torch.full_like(like, 1.0 if value else 0.0))
    if isinstance(value, (int, float)):
        return cast(Tensor, torch.full_like(like, float(value)))
    return cast(Tensor, torch.zeros_like(like))


__all__ = [
    "MetaplasticProjectionParams",
    "MetaplasticProjectionRule",
    "MetaplasticProjectionState",
]
