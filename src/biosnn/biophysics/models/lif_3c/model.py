"""Three-compartment LIF neuron model (soma + dendrite + axon/AIS)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

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
class LIF3CompParams:
    """Parameters for a simple 3-compartment LIF neuron."""

    v_rest: float = -0.065
    v_reset: float = -0.070
    v_thresh: float = -0.050
    tau_m_soma: float = 0.020
    tau_m_dend: float = 0.020
    tau_m_axon: float = 0.010
    axial_g_dend_soma: float = 1.5
    axial_g_soma_axon: float = 1.0
    refrac_period: float = 0.002
    spike_hold_time: float = 0.001
    reset_axon_on_spike: bool = True


@dataclass(slots=True)
class LIF3CompState:
    """State tensors for the 3-compartment LIF model."""

    v: Tensor  # [N, 3] where columns are [soma, dendrite, axon]
    refrac_left: Tensor  # [N]
    spike_hold_left: Tensor  # [N]
    last_drive_soma: Tensor  # [N]
    last_drive_dend: Tensor  # [N]
    last_drive_axon: Tensor  # [N]


@dataclass(frozen=True)
class _LIF3Consts:
    dt: Tensor
    v_reset: Tensor
    refrac_period: Tensor
    spike_hold_time: Tensor


@dataclass(frozen=True)
class _LIF3ParamsTensors:
    v_rest: Tensor
    v_thresh: Tensor
    tau_soma: Tensor
    tau_dend: Tensor
    tau_axon: Tensor
    g_ds: Tensor
    g_sa: Tensor


class LIF3CompModel(INeuronModel):
    """Vectorized 3-compartment LIF model."""

    name = "lif_3c"
    compartments = frozenset({Compartment.SOMA, Compartment.DENDRITE, Compartment.AXON})

    def __init__(self, params: LIF3CompParams | None = None) -> None:
        self.params = params or LIF3CompParams()
        self._validate_params(self.params)
        self._const_cache: OrderedDict[tuple[object, object, float], _LIF3Consts] = OrderedDict()
        self._param_cache: dict[tuple[object, object], _LIF3ParamsTensors] = {}
        self._const_cache_max = 32

    def init_state(self, n: int, *, ctx: StepContext) -> LIF3CompState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        v = torch.full((n, 3), self.params.v_rest, device=device, dtype=dtype)
        refrac_left = torch.zeros((n,), device=device, dtype=dtype)
        spike_hold_left = torch.zeros((n,), device=device, dtype=dtype)
        last_drive_soma = torch.zeros((n,), device=device, dtype=dtype)
        last_drive_dend = torch.zeros((n,), device=device, dtype=dtype)
        last_drive_axon = torch.zeros((n,), device=device, dtype=dtype)
        return LIF3CompState(
            v=v,
            refrac_left=refrac_left,
            spike_hold_left=spike_hold_left,
            last_drive_soma=last_drive_soma,
            last_drive_dend=last_drive_dend,
            last_drive_axon=last_drive_axon,
        )

    def reset_state(
        self,
        state: LIF3CompState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> LIF3CompState:
        _ = ctx
        if indices is None:
            state.v.fill_(self.params.v_rest)
            state.refrac_left.zero_()
            state.spike_hold_left.zero_()
            state.last_drive_soma.zero_()
            state.last_drive_dend.zero_()
            state.last_drive_axon.zero_()
            return state
        state.v[indices] = self.params.v_rest
        state.refrac_left[indices] = 0.0
        state.spike_hold_left[indices] = 0.0
        state.last_drive_soma[indices] = 0.0
        state.last_drive_dend[indices] = 0.0
        state.last_drive_axon[indices] = 0.0
        return state

    def step(
        self,
        state: LIF3CompState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[LIF3CompState, NeuronStepResult]:
        torch = require_torch()
        use_no_grad = _use_no_grad(ctx)
        with torch.no_grad() if use_no_grad else nullcontext():
            v = state.v
            v_soma = v[:, 0]
            v_dend = v[:, 1]
            v_axon = v[:, 2]
            consts = self._consts(v_soma, dt, ctx)
            pt = self._params(v_soma)
            validate_shapes = _validate_shapes(ctx)
            require_inputs_on_device = _require_inputs_on_device(ctx)

            drive_soma = _as_like(
                inputs.drive.get(Compartment.SOMA),
                v_soma,
                name="drive_soma",
                validate_shapes=validate_shapes,
                require_on_device=require_inputs_on_device,
            )
            drive_dend = _as_like(
                inputs.drive.get(Compartment.DENDRITE),
                v_dend,
                name="drive_dend",
                validate_shapes=validate_shapes,
                require_on_device=require_inputs_on_device,
            )
            drive_axon = _as_like(
                inputs.drive.get(Compartment.AXON),
                v_axon,
                name="drive_axon",
                validate_shapes=validate_shapes,
                require_on_device=require_inputs_on_device,
            )

            state.last_drive_soma.copy_(drive_soma)
            state.last_drive_dend.copy_(drive_dend)
            state.last_drive_axon.copy_(drive_axon)

            refrac_active = state.refrac_left > 0

            i_cpl_soma = pt.g_ds * (v_dend - v_soma) + pt.g_sa * (v_axon - v_soma)
            i_cpl_dend = pt.g_ds * (v_soma - v_dend)
            i_cpl_axon = pt.g_sa * (v_soma - v_axon)

            dv_soma = (-(v_soma - pt.v_rest) + drive_soma + i_cpl_soma) / pt.tau_soma
            dv_dend = (-(v_dend - pt.v_rest) + drive_dend + i_cpl_dend) / pt.tau_dend
            dv_axon = (-(v_axon - pt.v_rest) + drive_axon + i_cpl_axon) / pt.tau_axon

            v_soma_next = v_soma + consts.dt * dv_soma
            v_dend_next = v_dend + consts.dt * dv_dend
            v_axon_next = v_axon + consts.dt * dv_axon

            v_soma_next = torch.where(refrac_active, consts.v_reset, v_soma_next)
            spike_fired = (~refrac_active) & (v_soma_next >= pt.v_thresh)
            v_soma_next = torch.where(spike_fired, consts.v_reset, v_soma_next)
            if self.params.reset_axon_on_spike:
                v_axon_next = torch.where(spike_fired, consts.v_reset, v_axon_next)

            refrac_left = (state.refrac_left - consts.dt).clamp(min=0.0)
            refrac_left = torch.where(spike_fired, consts.refrac_period, refrac_left)

            spike_hold_left = (state.spike_hold_left - consts.dt).clamp(min=0.0)
            spike_hold_left = torch.where(spike_fired, consts.spike_hold_time, spike_hold_left)
            spikes = spike_fired | (spike_hold_left > 0)

            if _assert_finite(ctx):
                _assert_all_finite(v_soma_next, "v_soma", t)
                _assert_all_finite(v_dend_next, "v_dend", t)
                _assert_all_finite(v_axon_next, "v_axon", t)

            state.v[:, 0].copy_(v_soma_next)
            state.v[:, 1].copy_(v_dend_next)
            state.v[:, 2].copy_(v_axon_next)
            state.refrac_left.copy_(refrac_left)
            state.spike_hold_left.copy_(spike_hold_left)

            membrane = {
                Compartment.SOMA: v_soma_next,
                Compartment.DENDRITE: v_dend_next,
                Compartment.AXON: v_axon_next,
            }
            return state, NeuronStepResult(spikes=spikes, membrane=membrane)

    def state_tensors(self, state: LIF3CompState) -> Mapping[str, Tensor]:
        return {
            "v": state.v,
            "v_soma": state.v[:, 0],
            "v_dend": state.v[:, 1],
            "v_axon": state.v[:, 2],
            "refrac_left": state.refrac_left,
            "spike_hold_left": state.spike_hold_left,
            "last_drive_soma": state.last_drive_soma,
            "last_drive_dend": state.last_drive_dend,
            "last_drive_axon": state.last_drive_axon,
        }

    def _consts(self, like: Tensor, dt: float, ctx: StepContext) -> _LIF3Consts:
        key = (like.device, like.dtype, float(dt))
        cached = self._const_cache.get(key)
        if cached is not None:
            self._const_cache.move_to_end(key)
            return cached
        torch = require_torch()
        consts = _LIF3Consts(
            dt=torch.tensor(dt, device=like.device, dtype=like.dtype),
            v_reset=torch.tensor(self.params.v_reset, device=like.device, dtype=like.dtype),
            refrac_period=torch.tensor(self.params.refrac_period, device=like.device, dtype=like.dtype),
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

    def _params(self, like: Tensor) -> _LIF3ParamsTensors:
        key = (like.device, like.dtype)
        cached = self._param_cache.get(key)
        if cached is not None:
            return cached
        torch = require_torch()
        params = _LIF3ParamsTensors(
            v_rest=torch.tensor(self.params.v_rest, device=like.device, dtype=like.dtype),
            v_thresh=torch.tensor(self.params.v_thresh, device=like.device, dtype=like.dtype),
            tau_soma=torch.tensor(self.params.tau_m_soma, device=like.device, dtype=like.dtype),
            tau_dend=torch.tensor(self.params.tau_m_dend, device=like.device, dtype=like.dtype),
            tau_axon=torch.tensor(self.params.tau_m_axon, device=like.device, dtype=like.dtype),
            g_ds=torch.tensor(self.params.axial_g_dend_soma, device=like.device, dtype=like.dtype),
            g_sa=torch.tensor(self.params.axial_g_soma_axon, device=like.device, dtype=like.dtype),
        )
        self._param_cache[key] = params
        return params

    @staticmethod
    def _validate_params(params: LIF3CompParams) -> None:
        if params.tau_m_soma <= 0.0 or params.tau_m_dend <= 0.0 or params.tau_m_axon <= 0.0:
            raise ValueError("All tau_m values must be > 0")
        if params.refrac_period < 0.0:
            raise ValueError("refrac_period must be >= 0")
        if params.spike_hold_time < 0.0:
            raise ValueError("spike_hold_time must be >= 0")


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


__all__ = ["LIF3CompModel", "LIF3CompParams", "LIF3CompState"]

