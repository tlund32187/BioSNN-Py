from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(slots=True)
class SpikeInputState:
    v_soma: Tensor
    last_drive_soma: Tensor


class SpikeInputModel(INeuronModel):
    name = "spike_input"
    compartments = frozenset({Compartment.SOMA})

    def init_state(self, n: int, *, ctx: StepContext) -> SpikeInputState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        zeros = torch.zeros((n,), device=device, dtype=dtype)
        return SpikeInputState(
            v_soma=zeros.clone(),
            last_drive_soma=zeros,
        )

    def reset_state(
        self,
        state: SpikeInputState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> SpikeInputState:
        _ = ctx
        if indices is None:
            state.v_soma.zero_()
            state.last_drive_soma.zero_()
        else:
            state.v_soma[indices] = 0.0
            state.last_drive_soma[indices] = 0.0
        return state

    def step(
        self,
        state: SpikeInputState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[SpikeInputState, NeuronStepResult]:
        _ = (dt, t, ctx)
        drive = _as_like(inputs.drive.get(Compartment.SOMA), state.v_soma)
        state.last_drive_soma.copy_(drive)
        state.v_soma.copy_(drive)
        spikes = (drive > 0).bool()
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: SpikeInputState) -> Mapping[str, Tensor]:
        return {
            "v_soma": state.v_soma,
            "last_drive_soma": state.last_drive_soma,
        }


@dataclass(slots=True)
class DeterministicLIFState:
    v_soma: Tensor
    v_soma_raw: Tensor
    last_drive_dendrite: Tensor


class DeterministicLIFModel(INeuronModel):
    name = "deterministic_lif"
    compartments = frozenset({Compartment.DENDRITE})

    def __init__(
        self,
        *,
        leak: float,
        gain: float,
        thresh: float,
        reset: float,
    ) -> None:
        self.leak = float(leak)
        self.gain = float(gain)
        self.thresh = float(thresh)
        self.reset = float(reset)

    def init_state(self, n: int, *, ctx: StepContext) -> DeterministicLIFState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        zeros = torch.zeros((n,), device=device, dtype=dtype)
        return DeterministicLIFState(
            v_soma=zeros.clone(),
            v_soma_raw=zeros.clone(),
            last_drive_dendrite=zeros,
        )

    def reset_state(
        self,
        state: DeterministicLIFState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> DeterministicLIFState:
        _ = ctx
        if indices is None:
            state.v_soma.zero_()
            state.v_soma_raw.zero_()
            state.last_drive_dendrite.zero_()
        else:
            state.v_soma[indices] = 0.0
            state.v_soma_raw[indices] = 0.0
            state.last_drive_dendrite[indices] = 0.0
        return state

    def step(
        self,
        state: DeterministicLIFState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[DeterministicLIFState, NeuronStepResult]:
        torch = require_torch()
        _ = (dt, t, ctx)
        drive = _as_like(inputs.drive.get(Compartment.DENDRITE), state.v_soma)
        state.last_drive_dendrite.copy_(drive)

        v_raw = state.v_soma * (1.0 - self.leak) + (drive * self.gain)
        spikes = v_raw >= self.thresh
        reset_value = v_raw.new_full((), self.reset)
        v_next = torch.where(spikes, reset_value, v_raw)

        state.v_soma_raw.copy_(v_raw)
        state.v_soma.copy_(v_next)
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: DeterministicLIFState) -> Mapping[str, Tensor]:
        return {
            "v_soma": state.v_soma,
            "v_soma_raw": state.v_soma_raw,
            "last_drive_dendrite": state.last_drive_dendrite,
        }


def _as_like(value: Tensor | None, ref: Tensor) -> Tensor:
    if value is None:
        torch = require_torch()
        return cast(Tensor, torch.zeros_like(ref))
    if value.device != ref.device or value.dtype != ref.dtype:
        return cast(Tensor, value.to(device=ref.device, dtype=ref.dtype))
    return cast(Tensor, value)
