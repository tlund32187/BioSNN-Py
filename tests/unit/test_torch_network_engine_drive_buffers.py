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
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import resolve_device_dtype
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    v: Tensor


class DriveNeuronModel(INeuronModel):
    name = "drive"
    compartments = frozenset({Compartment.SOMA})

    def init_state(self, n: int, *, ctx: StepContext) -> DummyState:
        device, dtype = resolve_device_dtype(ctx)
        return DummyState(v=torch.zeros((n,), device=device, dtype=dtype))

    def reset_state(
        self,
        state: DummyState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> DummyState:
        if indices is None:
            state.v.zero_()
        else:
            state.v[indices] = 0.0
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
        spikes = torch.zeros_like(state.v)
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: DummyState):
        return {"v": state.v}


def _snapshot_drive_buffers(engine: TorchNetworkEngine):
    return {
        pop_name: (
            id(comp_map),
            {comp: id(tensor) for comp, tensor in comp_map.items()},
        )
        for pop_name, comp_map in engine._drive_buffers.items()
    }


def test_drive_buffers_reused_across_steps():
    n = 4

    def external_drive_fn(t, step, pop_name, ctx):
        device, dtype = resolve_device_dtype(ctx)
        return {Compartment.SOMA: torch.ones((n,), device=device, dtype=dtype)}

    pop = PopulationSpec(name="A", model=DriveNeuronModel(), n=n)
    engine = TorchNetworkEngine(
        populations=[pop],
        projections=[],
        external_drive_fn=external_drive_fn,
    )
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))

    before = _snapshot_drive_buffers(engine)
    assert before

    engine.step()
    after_first = _snapshot_drive_buffers(engine)
    engine.step()
    after_second = _snapshot_drive_buffers(engine)

    assert before == after_first
    assert before == after_second
