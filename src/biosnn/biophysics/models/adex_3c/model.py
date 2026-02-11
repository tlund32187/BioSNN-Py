"""Adaptive Exponential Integrate-and-Fire (AdEx) 3-compartment model."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast

from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
    Tensor,
)
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(frozen=True, slots=True)
class AdEx3CompParams:
    """Parameters for a 3-compartment AdEx neuron (soma+dendrite+axon)."""

    # Soma membrane parameters
    c_s: float = 200e-12
    g_l_s: float = 10e-9
    e_l_s: float = -0.070

    # Dendrite membrane parameters
    c_d: float = 200e-12
    g_l_d: float = 10e-9
    e_l_d: float = -0.070

    # Axon/AIS membrane parameters
    c_a: float = 120e-12
    g_l_a: float = 10e-9
    e_l_a: float = -0.068

    # Axial couplings (dend<->soma and soma<->axon)
    g_c_ds: float = 5e-9
    g_c_sa: float = 7e-9

    # AdEx exponential terms
    v_t_s: float = -0.050
    delta_t_s: float = 0.002
    v_t_a: float = -0.052
    delta_t_a: float = 0.001
    exp_arg_cap: float = 50.0

    # Spike/reset
    v_reset: float = -0.065
    v_spike: float = 0.020
    spike_source: Compartment = Compartment.SOMA  # SOMA or AXON/AIS

    # Adaptation currents
    a_s: float = 2e-9
    b_s: float = 0.05e-9
    tau_w_s: float = 0.200
    a_a: float = 0.0
    b_a: float = 0.0
    tau_w_a: float = 0.100

    # Temporal spike shaping
    refrac_period: float = 0.002
    spike_hold_time: float = 0.001

    # Reset policy
    reset_axon_on_spike: bool = True
    reset_dend_on_spike: bool = False


@dataclass(slots=True)
class AdEx3CompState:
    """State tensors for the 3-compartment AdEx model."""

    v_soma: Tensor
    v_dend: Tensor
    v_axon: Tensor
    w_soma: Tensor
    w_axon: Tensor
    refrac_left: Tensor
    spike_hold_left: Tensor
    last_drive_soma: Tensor
    last_drive_dend: Tensor
    last_drive_axon: Tensor


@dataclass(frozen=True)
class _AdEx3Consts:
    dt: Tensor
    v_reset: Tensor
    refrac_period: Tensor
    spike_hold_time: Tensor


@dataclass(frozen=True)
class _AdEx3ParamsTensors:
    c_s: Tensor
    g_l_s: Tensor
    e_l_s: Tensor
    c_d: Tensor
    g_l_d: Tensor
    e_l_d: Tensor
    c_a: Tensor
    g_l_a: Tensor
    e_l_a: Tensor
    g_c_ds: Tensor
    g_c_sa: Tensor
    v_t_s: Tensor
    delta_t_s: Tensor
    v_t_a: Tensor
    delta_t_a: Tensor
    v_spike: Tensor
    a_s: Tensor
    b_s: Tensor
    tau_w_s: Tensor
    a_a: Tensor
    b_a: Tensor
    tau_w_a: Tensor


class AdEx3CompModel(INeuronModel):
    """Population-level AdEx 3-compartment model (soma + dendrite + axon)."""

    name = "adex_3c"
    compartments = frozenset({Compartment.SOMA, Compartment.DENDRITE, Compartment.AXON})

    def __init__(self, params: AdEx3CompParams | None = None) -> None:
        self.params = params or AdEx3CompParams()
        self._validate_params(self.params)
        self._const_cache: OrderedDict[tuple[object, object, float], _AdEx3Consts] = OrderedDict()
        self._param_cache: dict[tuple[object, object], _AdEx3ParamsTensors] = {}
        self._const_cache_max = 32

    def init_state(self, n: int, *, ctx: StepContext) -> AdEx3CompState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        v_soma = torch.full((n,), self.params.e_l_s, device=device, dtype=dtype)
        v_dend = torch.full((n,), self.params.e_l_d, device=device, dtype=dtype)
        v_axon = torch.full((n,), self.params.e_l_a, device=device, dtype=dtype)
        w_soma = torch.zeros((n,), device=device, dtype=dtype)
        w_axon = torch.zeros((n,), device=device, dtype=dtype)
        refrac_left = torch.zeros((n,), device=device, dtype=dtype)
        spike_hold_left = torch.zeros((n,), device=device, dtype=dtype)
        last_drive_soma = torch.zeros((n,), device=device, dtype=dtype)
        last_drive_dend = torch.zeros((n,), device=device, dtype=dtype)
        last_drive_axon = torch.zeros((n,), device=device, dtype=dtype)
        return AdEx3CompState(
            v_soma=v_soma,
            v_dend=v_dend,
            v_axon=v_axon,
            w_soma=w_soma,
            w_axon=w_axon,
            refrac_left=refrac_left,
            spike_hold_left=spike_hold_left,
            last_drive_soma=last_drive_soma,
            last_drive_dend=last_drive_dend,
            last_drive_axon=last_drive_axon,
        )

    def reset_state(
        self,
        state: AdEx3CompState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> AdEx3CompState:
        _ = ctx
        if indices is None:
            state.v_soma.fill_(self.params.e_l_s)
            state.v_dend.fill_(self.params.e_l_d)
            state.v_axon.fill_(self.params.e_l_a)
            state.w_soma.zero_()
            state.w_axon.zero_()
            state.refrac_left.zero_()
            state.spike_hold_left.zero_()
            state.last_drive_soma.zero_()
            state.last_drive_dend.zero_()
            state.last_drive_axon.zero_()
            return state
        state.v_soma[indices] = self.params.e_l_s
        state.v_dend[indices] = self.params.e_l_d
        state.v_axon[indices] = self.params.e_l_a
        state.w_soma[indices] = 0.0
        state.w_axon[indices] = 0.0
        state.refrac_left[indices] = 0.0
        state.spike_hold_left[indices] = 0.0
        state.last_drive_soma[indices] = 0.0
        state.last_drive_dend[indices] = 0.0
        state.last_drive_axon[indices] = 0.0
        return state

    def step(
        self,
        state: AdEx3CompState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[AdEx3CompState, NeuronStepResult]:
        torch = require_torch()
        p = self.params
        use_no_grad = _use_no_grad(ctx)
        with torch.no_grad() if use_no_grad else nullcontext():
            v_soma = state.v_soma
            v_dend = state.v_dend
            v_axon = state.v_axon
            w_soma = state.w_soma
            w_axon = state.w_axon
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
            drive_a = _as_like(
                inputs.drive.get(Compartment.AXON),
                v_axon,
                name="drive_axon",
                validate_shapes=validate_shapes,
                require_on_device=require_inputs_on_device,
            )

            state.last_drive_soma.copy_(drive_s)
            state.last_drive_dend.copy_(drive_d)
            state.last_drive_axon.copy_(drive_a)

            refrac_active = state.refrac_left > 0

            exp_term_s = _adex_exp_term(
                v=v_soma,
                g_l=pt.g_l_s,
                v_t=pt.v_t_s,
                delta_t=pt.delta_t_s,
                enabled=self.params.delta_t_s > 0.0,
                exp_arg_cap=self.params.exp_arg_cap,
            )
            exp_term_a = _adex_exp_term(
                v=v_axon,
                g_l=pt.g_l_a,
                v_t=pt.v_t_a,
                delta_t=pt.delta_t_a,
                enabled=self.params.delta_t_a > 0.0,
                exp_arg_cap=self.params.exp_arg_cap,
            )

            i_leak_s = -pt.g_l_s * (v_soma - pt.e_l_s)
            i_couple_s = pt.g_c_ds * (v_dend - v_soma) + pt.g_c_sa * (v_axon - v_soma)
            i_total_s = i_leak_s + i_couple_s - w_soma + exp_term_s + drive_s
            dv_s = i_total_s / pt.c_s
            v_soma_next = v_soma + consts.dt * dv_s
            v_soma_next = torch.where(refrac_active, consts.v_reset, v_soma_next)

            i_leak_d = -pt.g_l_d * (v_dend - pt.e_l_d)
            i_couple_d = pt.g_c_ds * (v_soma - v_dend)
            i_total_d = i_leak_d + i_couple_d + drive_d
            dv_d = i_total_d / pt.c_d
            v_dend_next = v_dend + consts.dt * dv_d

            i_leak_a = -pt.g_l_a * (v_axon - pt.e_l_a)
            i_couple_a = pt.g_c_sa * (v_soma - v_axon)
            i_total_a = i_leak_a + i_couple_a - w_axon + exp_term_a + drive_a
            dv_a = i_total_a / pt.c_a
            v_axon_next = v_axon + consts.dt * dv_a

            if p.tau_w_s > 0.0:
                dw_s = (pt.a_s * (v_soma_next - pt.e_l_s) - w_soma) / pt.tau_w_s
                w_soma_next = w_soma + consts.dt * dw_s
            else:
                w_soma_next = w_soma

            if p.tau_w_a > 0.0:
                dw_a = (pt.a_a * (v_axon_next - pt.e_l_a) - w_axon) / pt.tau_w_a
                w_axon_next = w_axon + consts.dt * dw_a
            else:
                w_axon_next = w_axon

            spike_signal = _select_spike_signal(
                spike_source=p.spike_source,
                v_soma=v_soma_next,
                v_axon=v_axon_next,
            )
            spike_fired = (~refrac_active) & (spike_signal >= pt.v_spike)

            v_soma_next = torch.where(spike_fired, consts.v_reset, v_soma_next)
            if p.reset_axon_on_spike:
                v_axon_next = torch.where(spike_fired, consts.v_reset, v_axon_next)
            if p.reset_dend_on_spike:
                v_dend_next = torch.where(spike_fired, consts.v_reset, v_dend_next)

            w_soma_next = torch.where(spike_fired, w_soma_next + pt.b_s, w_soma_next)
            w_axon_next = torch.where(spike_fired, w_axon_next + pt.b_a, w_axon_next)

            refrac_left = (state.refrac_left - consts.dt).clamp(min=0.0)
            refrac_left = torch.where(spike_fired, consts.refrac_period, refrac_left)

            spike_hold_left = (state.spike_hold_left - consts.dt).clamp(min=0.0)
            spike_hold_left = torch.where(spike_fired, consts.spike_hold_time, spike_hold_left)
            spikes = spike_fired | (spike_hold_left > 0)

            if _assert_finite(ctx):
                _assert_all_finite(v_soma_next, "v_soma", t)
                _assert_all_finite(v_dend_next, "v_dend", t)
                _assert_all_finite(v_axon_next, "v_axon", t)
                _assert_all_finite(w_soma_next, "w_soma", t)
                _assert_all_finite(w_axon_next, "w_axon", t)

            state.v_soma.copy_(v_soma_next)
            state.v_dend.copy_(v_dend_next)
            state.v_axon.copy_(v_axon_next)
            state.w_soma.copy_(w_soma_next)
            state.w_axon.copy_(w_axon_next)
            state.refrac_left.copy_(refrac_left)
            state.spike_hold_left.copy_(spike_hold_left)

            membrane = {
                Compartment.SOMA: v_soma_next,
                Compartment.DENDRITE: v_dend_next,
                Compartment.AXON: v_axon_next,
            }
            return state, NeuronStepResult(spikes=spikes, membrane=membrane)

    def state_tensors(self, state: AdEx3CompState) -> Mapping[str, Tensor]:
        return {
            "v_soma": state.v_soma,
            "v_dend": state.v_dend,
            "v_axon": state.v_axon,
            "w": state.w_soma,
            "w_soma": state.w_soma,
            "w_axon": state.w_axon,
            "refrac_left": state.refrac_left,
            "spike_hold_left": state.spike_hold_left,
            "last_drive_soma": state.last_drive_soma,
            "last_drive_dend": state.last_drive_dend,
            "last_drive_axon": state.last_drive_axon,
        }

    def _consts(self, like: Tensor, dt: float, ctx: StepContext) -> _AdEx3Consts:
        key = (like.device, like.dtype, float(dt))
        cached = self._const_cache.get(key)
        if cached is not None:
            self._const_cache.move_to_end(key)
            return cached
        torch = require_torch()
        consts = _AdEx3Consts(
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

    def _params(self, like: Tensor) -> _AdEx3ParamsTensors:
        key = (like.device, like.dtype)
        cached = self._param_cache.get(key)
        if cached is not None:
            return cached
        torch = require_torch()
        params = _AdEx3ParamsTensors(
            c_s=torch.tensor(self.params.c_s, device=like.device, dtype=like.dtype),
            g_l_s=torch.tensor(self.params.g_l_s, device=like.device, dtype=like.dtype),
            e_l_s=torch.tensor(self.params.e_l_s, device=like.device, dtype=like.dtype),
            c_d=torch.tensor(self.params.c_d, device=like.device, dtype=like.dtype),
            g_l_d=torch.tensor(self.params.g_l_d, device=like.device, dtype=like.dtype),
            e_l_d=torch.tensor(self.params.e_l_d, device=like.device, dtype=like.dtype),
            c_a=torch.tensor(self.params.c_a, device=like.device, dtype=like.dtype),
            g_l_a=torch.tensor(self.params.g_l_a, device=like.device, dtype=like.dtype),
            e_l_a=torch.tensor(self.params.e_l_a, device=like.device, dtype=like.dtype),
            g_c_ds=torch.tensor(self.params.g_c_ds, device=like.device, dtype=like.dtype),
            g_c_sa=torch.tensor(self.params.g_c_sa, device=like.device, dtype=like.dtype),
            v_t_s=torch.tensor(self.params.v_t_s, device=like.device, dtype=like.dtype),
            delta_t_s=torch.tensor(self.params.delta_t_s, device=like.device, dtype=like.dtype),
            v_t_a=torch.tensor(self.params.v_t_a, device=like.device, dtype=like.dtype),
            delta_t_a=torch.tensor(self.params.delta_t_a, device=like.device, dtype=like.dtype),
            v_spike=torch.tensor(self.params.v_spike, device=like.device, dtype=like.dtype),
            a_s=torch.tensor(self.params.a_s, device=like.device, dtype=like.dtype),
            b_s=torch.tensor(self.params.b_s, device=like.device, dtype=like.dtype),
            tau_w_s=torch.tensor(self.params.tau_w_s, device=like.device, dtype=like.dtype),
            a_a=torch.tensor(self.params.a_a, device=like.device, dtype=like.dtype),
            b_a=torch.tensor(self.params.b_a, device=like.device, dtype=like.dtype),
            tau_w_a=torch.tensor(self.params.tau_w_a, device=like.device, dtype=like.dtype),
        )
        self._param_cache[key] = params
        return params

    @staticmethod
    def _validate_params(params: AdEx3CompParams) -> None:
        for name, value in (
            ("c_s", params.c_s),
            ("g_l_s", params.g_l_s),
            ("c_d", params.c_d),
            ("g_l_d", params.g_l_d),
            ("c_a", params.c_a),
            ("g_l_a", params.g_l_a),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        for name, value in (
            ("g_c_ds", params.g_c_ds),
            ("g_c_sa", params.g_c_sa),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
        for name, value in (
            ("tau_w_s", params.tau_w_s),
            ("tau_w_a", params.tau_w_a),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if params.refrac_period < 0.0:
            raise ValueError("refrac_period must be >= 0")
        if params.spike_hold_time < 0.0:
            raise ValueError("spike_hold_time must be >= 0")
        if params.exp_arg_cap <= 0.0:
            raise ValueError("exp_arg_cap must be > 0")
        if params.spike_source not in {
            Compartment.SOMA,
            Compartment.AXON,
            Compartment.AIS,
        }:
            raise ValueError("spike_source must be SOMA, AXON, or AIS")


def _adex_exp_term(
    *,
    v: Tensor,
    g_l: Tensor,
    v_t: Tensor,
    delta_t: Tensor,
    enabled: bool,
    exp_arg_cap: float,
) -> Tensor:
    torch = require_torch()
    if not enabled:
        return cast(Tensor, v.new_zeros(v.shape))
    exp_arg = (v - v_t) / delta_t
    exp_arg = torch.clamp(exp_arg, max=exp_arg_cap)
    return cast(Tensor, g_l * delta_t * torch.exp(exp_arg))


def _select_spike_signal(*, spike_source: Compartment, v_soma: Tensor, v_axon: Tensor) -> Tensor:
    if spike_source in {Compartment.AXON, Compartment.AIS}:
        return v_axon
    return v_soma


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


__all__ = ["AdEx3CompModel", "AdEx3CompParams", "AdEx3CompState"]
