"""Template neuron model for custom implementations."""

from __future__ import annotations

from dataclasses import dataclass

from biosnn.biophysics.models.base import NeuronModelBase, StateTensorSpec
from biosnn.contracts.neurons import Compartment, NeuronStepResult, StepContext
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch


@dataclass(frozen=True, slots=True)
class TemplateLIFParams:
    v_rest: float = -0.065
    v_reset: float = -0.070
    v_thresh: float = -0.050
    tau_m: float = 0.020
    refrac_period: float = 0.002


@dataclass(slots=True)
class TemplateLIFState:
    v_soma: Tensor
    refrac_left: Tensor


class TemplateNeuronModel(NeuronModelBase):
    """Minimal LIF-like example extending NeuronModelBase."""

    name = "template_lif"

    def __init__(self, params: TemplateLIFParams | None = None) -> None:
        super().__init__()
        self.params = params or TemplateLIFParams()

    def state_tensors_spec(self):
        return {
            "v_soma": StateTensorSpec(shape=(None,), dtype="float"),
            "refrac_left": StateTensorSpec(shape=(None,), dtype="float"),
        }

    def init_state_tensors(self, n: int, *, device, dtype) -> TemplateLIFState:
        torch = require_torch()
        v_soma = torch.full((n,), self.params.v_rest, device=device, dtype=dtype)
        refrac_left = torch.zeros((n,), device=device, dtype=dtype)
        return TemplateLIFState(v_soma=v_soma, refrac_left=refrac_left)

    def state_tensors(self, state: TemplateLIFState):
        return {"v_soma": state.v_soma, "refrac_left": state.refrac_left}

    def step_state(
        self,
        state: TemplateLIFState,
        drive,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ):
        _ = t, ctx
        torch = require_torch()
        v_soma = state.v_soma
        refrac_left = state.refrac_left

        drive_soma = drive.get(Compartment.SOMA)
        if drive_soma is None:
            drive_soma = torch.zeros_like(v_soma)

        not_refrac = refrac_left <= 0
        dv = (-(v_soma - self.params.v_rest) + drive_soma) * (dt / self.params.tau_m)
        v_next = torch.where(not_refrac, v_soma + dv, torch.full_like(v_soma, self.params.v_reset))

        spike_fired = (v_next >= self.params.v_thresh) & not_refrac
        v_next = torch.where(spike_fired, torch.full_like(v_next, self.params.v_reset), v_next)

        refrac_next = torch.where(
            spike_fired,
            torch.full_like(refrac_left, self.params.refrac_period),
            torch.clamp(refrac_left - dt, min=0.0),
        )

        state.v_soma.copy_(v_next)
        state.refrac_left.copy_(refrac_next)

        return state, NeuronStepResult(spikes=spike_fired, membrane={Compartment.SOMA: v_next})


__all__ = ["TemplateNeuronModel", "TemplateLIFParams", "TemplateLIFState"]
