"""Wrapper learning rule that modulates dW via additional neuromodulators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch


@dataclass(frozen=True, slots=True)
class ModulatedRuleWrapperParams:
    ach_lr_gain: float = 0.0
    ne_lr_gain: float = 0.0
    ht_lr_gain: float = 0.0
    ht_extra_weight_decay: float = 0.0
    lr_clip_min: float = 0.1
    lr_clip_max: float = 10.0
    dopamine_baseline: float = 0.0
    ach_baseline: float = 0.0
    ne_baseline: float = 0.0
    ht_baseline: float = 0.0
    combine_mode: Literal["exp", "linear"] = "exp"
    missing_modulators_policy: Literal["zero"] = "zero"


@dataclass(slots=True)
class ModulatedRuleWrapperState:
    inner_state: Any


class ModulatedRuleWrapper(ILearningRule):
    """Wrap any base learning rule and modulate edge-local weight updates."""

    name = "modulated_wrapper"
    supports_sparse = True

    def __init__(
        self,
        inner: ILearningRule,
        params: ModulatedRuleWrapperParams | None = None,
    ) -> None:
        self.inner = inner
        self.params = params or ModulatedRuleWrapperParams()
        self.name = f"{getattr(inner, 'name', 'learning_rule')}_modulated"
        self.supports_sparse = bool(getattr(inner, "supports_sparse", False))

    def init_state(self, e: int, *, ctx: Any) -> ModulatedRuleWrapperState:
        return ModulatedRuleWrapperState(inner_state=self.inner.init_state(e, ctx=ctx))

    def step(
        self,
        state: ModulatedRuleWrapperState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: Any,
    ) -> tuple[ModulatedRuleWrapperState, LearningStepResult]:
        torch = require_torch()
        state.inner_state, inner_result = self.inner.step(
            state.inner_state,
            batch,
            dt=dt,
            t=t,
            ctx=ctx,
        )

        weights = batch.weights
        dw = inner_result.d_weights
        if dw.device != weights.device:
            dw = dw.to(device=weights.device)
        if dw.dtype != weights.dtype:
            dw = dw.to(dtype=weights.dtype)

        ach = _resolve_modulator(
            batch=batch,
            kind=ModulatorKind.ACETYLCHOLINE,
            like=weights,
            missing_policy=self.params.missing_modulators_policy,
        )
        ne = _resolve_modulator(
            batch=batch,
            kind=ModulatorKind.NORADRENALINE,
            like=weights,
            missing_policy=self.params.missing_modulators_policy,
        )
        ht = _resolve_modulator(
            batch=batch,
            kind=ModulatorKind.SEROTONIN,
            like=weights,
            missing_policy=self.params.missing_modulators_policy,
        )
        dopamine = _resolve_modulator(
            batch=batch,
            kind=ModulatorKind.DOPAMINE,
            like=weights,
            missing_policy=self.params.missing_modulators_policy,
        )

        centered_ach = ach - float(self.params.ach_baseline)
        centered_ne = ne - float(self.params.ne_baseline)
        centered_ht = ht - float(self.params.ht_baseline)
        mod_arg = (
            float(self.params.ach_lr_gain) * centered_ach
            + float(self.params.ne_lr_gain) * centered_ne
            + float(self.params.ht_lr_gain) * centered_ht
        )

        combine_mode = str(self.params.combine_mode).strip().lower()
        lr_scale = 1.0 + mod_arg if combine_mode == "linear" else mod_arg.exp()
        lr_scale = lr_scale.clamp(min=float(self.params.lr_clip_min), max=float(self.params.lr_clip_max))
        dw = dw * lr_scale

        extra_decay = float(self.params.ht_extra_weight_decay)
        if extra_decay != 0.0:
            ht_drive = (ht - float(self.params.ht_baseline)).relu()
            decay_term = ht_drive * extra_decay
            # Sign-correct extra decay: always pushes weights toward zero.
            dw.addcmul_(weights, decay_term, value=-1.0)

        extras: dict[str, Tensor] = dict(inner_result.extras) if inner_result.extras is not None else {}
        extras.update(
            {
                "mean_lr_scale": lr_scale.mean() if lr_scale.numel() else torch.zeros((), device=dw.device, dtype=dw.dtype),
                "mean_ach": ach.mean() if ach.numel() else torch.zeros((), device=dw.device, dtype=dw.dtype),
                "mean_ne": ne.mean() if ne.numel() else torch.zeros((), device=dw.device, dtype=dw.dtype),
                "mean_ht": ht.mean() if ht.numel() else torch.zeros((), device=dw.device, dtype=dw.dtype),
                "mean_dopamine": dopamine.mean()
                if dopamine.numel()
                else torch.zeros((), device=dw.device, dtype=dw.dtype),
            }
        )
        return state, LearningStepResult(d_weights=dw, extras=extras)

    def state_tensors(self, state: ModulatedRuleWrapperState) -> Mapping[str, Tensor]:
        return self.inner.state_tensors(state.inner_state)


def _resolve_modulator(
    *,
    batch: LearningBatch,
    kind: ModulatorKind,
    like: Tensor,
    missing_policy: str,
) -> Tensor:
    torch = require_torch()
    if batch.modulators is None or kind not in batch.modulators:
        if missing_policy == "zero":
            return cast(Tensor, torch.zeros_like(like))
        raise ValueError(f"Unsupported missing_modulators_policy={missing_policy!r}")

    value = batch.modulators[kind]
    if value.device != like.device or value.dtype != like.dtype:
        value = value.to(device=like.device, dtype=like.dtype)
    if value.numel() == like.numel():
        return cast(Tensor, value)
    if value.numel() == 1:
        return cast(Tensor, value.reshape(()).expand_as(like))
    raise ValueError(
        f"Modulator {kind.value} must be scalar or edge-local sized; "
        f"got {tuple(value.shape)} for expected {tuple(like.shape)}"
    )


__all__ = [
    "ModulatedRuleWrapper",
    "ModulatedRuleWrapperParams",
    "ModulatedRuleWrapperState",
]
