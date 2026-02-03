"""Generalized LIF (GLIF) neuron model (batched, torch-friendly)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

from biosnn.biophysics.models._torch_utils import require_torch, resolve_device_dtype
from biosnn.contracts.neurons import (
    Compartment,
    GLIFParams,
    GLIFState,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
    Tensor,
)


@dataclass(frozen=True)
class _GLIFConsts:
    dt: Tensor
    v_reset: Tensor
    refrac_period: Tensor
    spike_hold_time: Tensor


@dataclass(frozen=True)
class _GLIFParamsTensors:
    v_rest: Tensor
    v_thresh0: Tensor
    tau_m: Tensor
    r_m: Tensor
    theta_tau: Tensor
    theta_increment: Tensor
    theta_min: Tensor
    theta_max: Tensor


class GLIFModel(INeuronModel):
    """Population-level GLIF model."""

    name = "glif"
    compartments = frozenset({Compartment.SOMA})

    def __init__(self, params: GLIFParams | None = None) -> None:
        self.params = params or GLIFParams()
        self._const_cache: OrderedDict[tuple[object, object, float], _GLIFConsts] = OrderedDict()
        self._param_cache: dict[tuple[object, object], _GLIFParamsTensors] = {}
        self._const_cache_max = 32

    def init_state(self, n: int, *, ctx: StepContext) -> GLIFState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        v_soma = torch.full((n,), self.params.v_rest, device=device, dtype=dtype)
        refrac_left = torch.zeros((n,), device=device, dtype=dtype)
        spike_hold_left = torch.zeros((n,), device=device, dtype=dtype)
        theta = torch.zeros((n,), device=device, dtype=dtype)
        return GLIFState(
            v_soma=v_soma,
            refrac_left=refrac_left,
            spike_hold_left=spike_hold_left,
            theta=theta,
        )

    def reset_state(
        self,
        state: GLIFState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> GLIFState:
        _ = ctx
        if indices is None:
            state.v_soma.fill_(self.params.v_rest)
            state.refrac_left.zero_()
            state.spike_hold_left.zero_()
            state.theta.zero_()
            return state
        state.v_soma[indices] = self.params.v_rest
        state.refrac_left[indices] = 0.0
        state.spike_hold_left[indices] = 0.0
        state.theta[indices] = 0.0
        return state

    def step(
        self,
        state: GLIFState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[GLIFState, NeuronStepResult]:
        torch = require_torch()
        p = self.params
        use_no_grad = _use_no_grad(ctx)
        with torch.no_grad() if use_no_grad else nullcontext():
            v_soma = state.v_soma
            consts = self._consts(v_soma, dt, ctx)
            pt = self._params(v_soma)
            validate_shapes = _validate_shapes(ctx)

            drive = _as_like(
                inputs.drive.get(Compartment.SOMA),
                v_soma,
                name="drive_soma",
                validate_shapes=validate_shapes,
            )
            refrac_active = state.refrac_left > 0

            dv = (-(v_soma - pt.v_rest) + pt.r_m * drive) / pt.tau_m
            v_next = v_soma + consts.dt * dv
            v_next = torch.where(refrac_active, consts.v_reset, v_next)

            theta_next = state.theta
            if p.enable_threshold_adaptation:
                if p.theta_tau > 0:
                    theta_next = theta_next - (consts.dt / pt.theta_tau) * theta_next
                theta_next = theta_next.clamp(min=pt.theta_min, max=pt.theta_max)
                v_thresh = pt.v_thresh0 + theta_next
                reset_theta = False
            else:
                v_thresh = pt.v_thresh0
                reset_theta = True
            spike_fired = (~refrac_active) & (v_next >= v_thresh)

            if p.enable_threshold_adaptation and p.theta_increment != 0.0:
                theta_next = torch.where(
                    spike_fired,
                    theta_next + pt.theta_increment,
                    theta_next,
                )
                theta_next = theta_next.clamp(min=pt.theta_min, max=pt.theta_max)

            v_next = torch.where(spike_fired, consts.v_reset, v_next)

            refrac_left = (state.refrac_left - consts.dt).clamp(min=0.0)
            refrac_left = torch.where(
                spike_fired,
                consts.refrac_period,
                refrac_left,
            )

            spike_hold_left = (state.spike_hold_left - consts.dt).clamp(min=0.0)
            if p.spike_hold_time >= 0.0:
                spike_hold_left = torch.where(
                    spike_fired,
                    consts.spike_hold_time,
                    spike_hold_left,
                )
            spikes = spike_fired | (spike_hold_left > 0)

            if _assert_finite(ctx):
                _assert_all_finite(v_next, "v_soma", t)
                _assert_all_finite(refrac_left, "refrac_left", t)
                _assert_all_finite(theta_next, "theta", t)

            state.v_soma.copy_(v_next)
            state.refrac_left.copy_(refrac_left)
            state.spike_hold_left.copy_(spike_hold_left)
            if reset_theta:
                state.theta.zero_()
            else:
                state.theta.copy_(theta_next)

        return state, NeuronStepResult(spikes=spikes, membrane={Compartment.SOMA: v_next})

    def state_tensors(self, state: GLIFState) -> Mapping[str, Tensor]:
        return {
            "v_soma": state.v_soma,
            "refrac_left": state.refrac_left,
            "spike_hold_left": state.spike_hold_left,
            "theta": state.theta,
        }

    def _consts(self, like: Tensor, dt: float, ctx: StepContext) -> _GLIFConsts:
        key = (like.device, like.dtype, float(dt))
        cached = self._const_cache.get(key)
        if cached is not None:
            self._const_cache.move_to_end(key)
            return cached
        torch = require_torch()
        consts = _GLIFConsts(
            dt=torch.tensor(dt, device=like.device, dtype=like.dtype),
            v_reset=torch.tensor(self.params.v_reset, device=like.device, dtype=like.dtype),
            refrac_period=torch.tensor(
                self.params.refrac_period, device=like.device, dtype=like.dtype
            ),
            spike_hold_time=torch.tensor(
                self.params.spike_hold_time, device=like.device, dtype=like.dtype
            ),
        )
        max_entries = _const_cache_max(ctx, self._const_cache_max)
        if max_entries is None:
            return consts
        if max_entries > 0:
            while len(self._const_cache) >= max_entries:
                self._const_cache.popitem(last=False)
            self._const_cache[key] = consts
        return consts

    def _params(self, like: Tensor) -> _GLIFParamsTensors:
        key = (like.device, like.dtype)
        cached = self._param_cache.get(key)
        if cached is not None:
            return cached
        torch = require_torch()
        params = _GLIFParamsTensors(
            v_rest=torch.tensor(self.params.v_rest, device=like.device, dtype=like.dtype),
            v_thresh0=torch.tensor(self.params.v_thresh0, device=like.device, dtype=like.dtype),
            tau_m=torch.tensor(self.params.tau_m, device=like.device, dtype=like.dtype),
            r_m=torch.tensor(self.params.r_m, device=like.device, dtype=like.dtype),
            theta_tau=torch.tensor(self.params.theta_tau, device=like.device, dtype=like.dtype),
            theta_increment=torch.tensor(
                self.params.theta_increment, device=like.device, dtype=like.dtype
            ),
            theta_min=torch.tensor(self.params.theta_min, device=like.device, dtype=like.dtype),
            theta_max=torch.tensor(self.params.theta_max, device=like.device, dtype=like.dtype),
        )
        self._param_cache[key] = params
        return params


def _as_like(
    tensor: Tensor | None,
    like: Tensor,
    *,
    name: str,
    validate_shapes: bool,
) -> Tensor:
    if tensor is None:
        out = like.new_zeros(like.shape)
    elif tensor.device != like.device or tensor.dtype != like.dtype:
        out = tensor.to(device=like.device, dtype=like.dtype)
    else:
        out = tensor
    if validate_shapes:
        _ensure_1d_len(out, like.shape[0], name)
    return out


def _ensure_1d_len(tensor: Tensor, n: int, name: str) -> None:
    if tensor.dim() != 1 or tensor.shape[0] != n:
        raise ValueError(f"{name} must have shape [{n}], got {tuple(tensor.shape)}")


def _use_no_grad(ctx: StepContext) -> bool:
    if ctx.extras and "no_grad" in ctx.extras:
        return bool(ctx.extras["no_grad"])
    return not ctx.is_training


def _validate_shapes(ctx: StepContext) -> bool:
    if ctx.extras and "validate_shapes" in ctx.extras:
        return bool(ctx.extras["validate_shapes"])
    return True


def _assert_finite(ctx: StepContext) -> bool:
    if ctx.extras and "assert_finite" in ctx.extras:
        return bool(ctx.extras["assert_finite"])
    return False


def _assert_all_finite(tensor: Tensor, name: str, t: float) -> None:
    torch = require_torch()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite values in {name} at t={t}")


def _const_cache_max(ctx: StepContext, default: int) -> int | None:
    if ctx.extras and "const_cache_max" in ctx.extras:
        try:
            value = int(ctx.extras["const_cache_max"])
        except (TypeError, ValueError):
            return default
        return value if value > 0 else None
    return default
