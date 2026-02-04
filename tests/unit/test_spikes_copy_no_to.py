from __future__ import annotations

from dataclasses import dataclass

import pytest

from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.simulation import SimulationConfig
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    pass


class FixedSpikesModel(INeuronModel):
    name = "fixed_spikes"
    compartments = frozenset({Compartment.SOMA})

    def init_state(self, n: int, *, ctx: StepContext) -> DummyState:
        _ = n, ctx
        return DummyState()

    def reset_state(
        self,
        state: DummyState,
        *,
        ctx: StepContext,
        indices=None,
    ) -> DummyState:
        _ = ctx, indices
        return state

    def step(
        self,
        state: DummyState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[DummyState, NeuronStepResult]:
        _ = state, inputs, dt, t
        spikes = torch.tensor([True, False, True], device=ctx.device, dtype=torch.bool)
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: DummyState):
        _ = state
        return {}


@pytest.mark.parametrize("compiled_mode", [False, True])
def test_spike_copy_no_to(monkeypatch, compiled_mode: bool) -> None:
    pop = PopulationSpec(name="A", model=FixedSpikesModel(), n=3)
    engine = TorchNetworkEngine(populations=[pop], projections=[], compiled_mode=compiled_mode)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))

    def _fail(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("Tensor.to should not be called in spike copy path")

    monkeypatch.setattr(torch.Tensor, "to", _fail, raising=False)
    engine.step()
    spikes = engine._pop_states["A"].spikes
    assert spikes.dtype == torch.bool
    assert spikes.tolist() == [True, False, True]
