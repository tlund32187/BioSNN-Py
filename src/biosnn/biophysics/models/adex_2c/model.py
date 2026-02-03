"""Adaptive Exponential Integrate-and-Fire (AdEx) 2-compartment model."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

from biosnn.biophysics.models._torch_utils import require_torch, resolve_device_dtype
from biosnn.contracts.neurons import (
    AdEx2CompParams,
    AdEx2CompState,
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
    Tensor,
)


@dataclass(frozen=True)
class _AdExConsts:
    dt: Tensor
    v_reset: Tensor
    refrac_period: Tensor
    spike_hold_time: Tensor


@dataclass(frozen=True)
class _AdExParamsTensors:
    c_s: Tensor
    g_l_s: Tensor
    e_l_s: Tensor
    c_d: Tensor
    g_l_d: Tensor
    e_l_d: Tensor
    g_c: Tensor
    v_t: Tensor
    delta_t: Tensor
    v_spike: Tensor
    a: Tensor
    b: Tensor
    tau_w: Tensor


class AdEx2CompModel(INeuronModel):
    """Population-level AdEx 2-compartment model (soma + dendrite)."""

    name = "adex_2c"
    compartments = frozenset({Compartment.SOMA, Compartment.DENDRITE})

    def __init__(self, params: AdEx2CompParams | None = None) -> None:
        self.params = params or AdEx2CompParams()
        self._const_cache: OrderedDict[tuple[object, object, float], _AdExConsts] = OrderedDict()
        self._param_cache: dict[tuple[object, object], _AdExParamsTensors] = {}
        self._const_cache_max = 32

    def init_state(self, n: int, *, ctx: StepContext) -> AdEx2CompState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        v_soma = torch.full((n,), self.params.e_l_s, device=device, dtype=dtype)
        v_dend = torch.full((n,), self.params.e_l_d, device=device, dtype=dtype)
        w = torch.zeros((n,), device=device, dtype=dtype)
        refrac_left = torch.zeros((n,), device=device, dtype=dtype)
        spike_hold_left = torch.zeros((n,), device=device, dtype=dtype)
        return AdEx2CompState(
            v_soma=v_soma,
            v_dend=v_dend,
            w=w,
            refrac_left=refrac_left,
            spike_hold_left=spike_hold_left,
        )

    def reset_state(
        self,
        state: AdEx2CompState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> AdEx2CompState:
        _ = ctx
        if indices is None:
            state.v_soma.fill_(self.params.e_l_s)
            state.v_dend.fill_(self.params.e_l_d)
            state.w.zero_()
            state.refrac_left.zero_()
            state.spike_hold_left.zero_()
            return state
        state.v_soma[indices] = self.params.e_l_s
        state.v_dend[indices] = self.params.e_l_d
        state.w[indices] = 0.0
        state.refrac_left[indices] = 0.0
        state.spike_hold_left[indices] = 0.0
        return state

    def step(
        self,
        state: AdEx2CompState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[AdEx2CompState, NeuronStepResult]:
        torch = require_torch()
        p = self.params
        use_no_grad = _use_no_grad(ctx)
        with torch.no_grad() if use_no_grad else nullcontext():
            v_soma = state.v_soma
            v_dend = state.v_dend
            w = state.w
            consts = self._consts(v_soma, dt, ctx)
            pt = self._params(v_soma)
            validate_shapes = _validate_shapes(ctx)
            require_inputs_on_device = _require_inputs_on_device(ctx)

            drive_s = _as_like(
                inputs.drive.get(Compartment.SOMA),
                v_soma,
                name="drive_soma",
                validate_shapes=validate_shapes,
                require_on_device=require_inputs_on_device,
            )
            drive_d = _as_like(
                inputs.drive.get(Compartment.DENDRITE),
                v_dend,
                name="drive_dend",
                validate_shapes=validate_shapes,
                require_on_device=require_inputs_on_device,
            )

            refrac_active = state.refrac_left > 0

            if p.delta_t > 0:
                exp_arg = (v_soma - pt.v_t) / pt.delta_t
                exp_arg = torch.clamp(exp_arg, max=50.0)
                exp_term = pt.g_l_s * pt.delta_t * torch.exp(exp_arg)
            else:
                exp_term = v_soma.new_zeros(v_soma.shape)

            i_leak_s = -pt.g_l_s * (v_soma - pt.e_l_s)
            i_couple_s = pt.g_c * (v_dend - v_soma)
            i_total_s = i_leak_s + i_couple_s - w + exp_term + drive_s
            dv_s = i_total_s / pt.c_s
            v_soma_next = v_soma + consts.dt * dv_s
            v_soma_next = torch.where(
                refrac_active,
                consts.v_reset,
                v_soma_next,
            )

            i_leak_d = -pt.g_l_d * (v_dend - pt.e_l_d)
            i_couple_d = pt.g_c * (v_soma - v_dend)
            i_total_d = i_leak_d + i_couple_d + drive_d
            dv_d = i_total_d / pt.c_d
            v_dend_next = v_dend + consts.dt * dv_d

            if p.tau_w > 0:
                # Use v_soma_next for semi-implicit coupling; keep consistent with regression tests.
                dw = (pt.a * (v_soma_next - pt.e_l_s) - w) / pt.tau_w
                w_next = w + consts.dt * dw
            else:
                w_next = w

            spike_fired = (~refrac_active) & (v_soma_next >= pt.v_spike)

            v_soma_next = torch.where(
                spike_fired,
                consts.v_reset,
                v_soma_next,
            )
            w_next = torch.where(spike_fired, w_next + pt.b, w_next)

            refrac_left = (state.refrac_left - consts.dt).clamp(min=0.0)
            refrac_left = torch.where(
                spike_fired,
                consts.refrac_period,
                refrac_left,
            )

            spike_hold_left = (state.spike_hold_left - consts.dt).clamp(min=0.0)
            spike_hold_left = torch.where(
                spike_fired,
                consts.spike_hold_time,
                spike_hold_left,
            )
            spikes = spike_fired | (spike_hold_left > 0)

            if _assert_finite(ctx):
                _assert_all_finite(v_soma_next, "v_soma", t)
                _assert_all_finite(v_dend_next, "v_dend", t)
                _assert_all_finite(w_next, "w", t)

            state.v_soma.copy_(v_soma_next)
            state.v_dend.copy_(v_dend_next)
            state.w.copy_(w_next)
            state.refrac_left.copy_(refrac_left)
            state.spike_hold_left.copy_(spike_hold_left)

            membrane = {Compartment.SOMA: v_soma_next, Compartment.DENDRITE: v_dend_next}
            return state, NeuronStepResult(spikes=spikes, membrane=membrane)

    def state_tensors(self, state: AdEx2CompState) -> Mapping[str, Tensor]:
        return {
            "v_soma": state.v_soma,
            "v_dend": state.v_dend,
            "w": state.w,
            "refrac_left": state.refrac_left,
            "spike_hold_left": state.spike_hold_left,
        }

    def _consts(self, like: Tensor, dt: float, ctx: StepContext) -> _AdExConsts:
        key = (like.device, like.dtype, float(dt))
        cached = self._const_cache.get(key)
        if cached is not None:
            self._const_cache.move_to_end(key)
            return cached
        torch = require_torch()
        consts = _AdExConsts(
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

    def _params(self, like: Tensor) -> _AdExParamsTensors:
        key = (like.device, like.dtype)
        cached = self._param_cache.get(key)
        if cached is not None:
            return cached
        torch = require_torch()
        params = _AdExParamsTensors(
            c_s=torch.tensor(self.params.c_s, device=like.device, dtype=like.dtype),
            g_l_s=torch.tensor(self.params.g_l_s, device=like.device, dtype=like.dtype),
            e_l_s=torch.tensor(self.params.e_l_s, device=like.device, dtype=like.dtype),
            c_d=torch.tensor(self.params.c_d, device=like.device, dtype=like.dtype),
            g_l_d=torch.tensor(self.params.g_l_d, device=like.device, dtype=like.dtype),
            e_l_d=torch.tensor(self.params.e_l_d, device=like.device, dtype=like.dtype),
            g_c=torch.tensor(self.params.g_c, device=like.device, dtype=like.dtype),
            v_t=torch.tensor(self.params.v_t, device=like.device, dtype=like.dtype),
            delta_t=torch.tensor(self.params.delta_t, device=like.device, dtype=like.dtype),
            v_spike=torch.tensor(self.params.v_spike, device=like.device, dtype=like.dtype),
            a=torch.tensor(self.params.a, device=like.device, dtype=like.dtype),
            b=torch.tensor(self.params.b, device=like.device, dtype=like.dtype),
            tau_w=torch.tensor(self.params.tau_w, device=like.device, dtype=like.dtype),
        )
        self._param_cache[key] = params
        return params


def _as_like(
    tensor: Tensor | None,
    like: Tensor,
    *,
    name: str,
    validate_shapes: bool,
    require_on_device: bool,
) -> Tensor:
    if tensor is None:
        out = like.new_zeros(like.shape)
    elif tensor.device != like.device or tensor.dtype != like.dtype:
        if require_on_device and tensor.device != like.device:
            raise ValueError(f"{name} must be on device {like.device}, got {tensor.device}")
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


def _require_inputs_on_device(ctx: StepContext) -> bool:
    if ctx.extras and "require_inputs_on_device" in ctx.extras:
        return bool(ctx.extras["require_inputs_on_device"])
    return False


def _const_cache_max(ctx: StepContext, default: int) -> int | None:
    if ctx.extras and "const_cache_max" in ctx.extras:
        try:
            value = int(ctx.extras["const_cache_max"])
        except (TypeError, ValueError):
            return default
        return value if value > 0 else None
    return default


def _assert_finite(ctx: StepContext) -> bool:
    if ctx.extras and "assert_finite" in ctx.extras:
        return bool(ctx.extras["assert_finite"])
    return False


def _assert_all_finite(tensor: Tensor, name: str, t: float) -> None:
    torch = require_torch()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite values in {name} at t={t}")
