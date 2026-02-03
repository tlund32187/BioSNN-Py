from __future__ import annotations

from dataclasses import dataclass

import pytest

from biosnn.biophysics.models._torch_utils import resolve_device_dtype
from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    dummy: Tensor


class SilentNeuronModel(INeuronModel):
    name = "silent"
    compartments = frozenset({Compartment.DENDRITE})

    def init_state(self, n: int, *, ctx: StepContext) -> DummyState:
        device, dtype = resolve_device_dtype(ctx)
        return DummyState(dummy=torch.zeros((n,), device=device, dtype=dtype))

    def reset_state(
        self,
        state: DummyState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> DummyState:
        if indices is None:
            state.dummy.zero_()
        else:
            state.dummy[indices] = 0.0
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
        spikes = torch.zeros_like(state.dummy)
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: DummyState):
        return {}


def test_network_engine_delay_routing():
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    delay_steps = torch.tensor([2], dtype=torch.int32)
    weights = torch.tensor([1.0], dtype=torch.float32)

    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        weights=weights,
    )

    pop_a = PopulationSpec(name="A", model=SilentNeuronModel(), n=1)
    pop_b = PopulationSpec(name="B", model=SilentNeuronModel(), n=1)

    proj = ProjectionSpec(
        name="A_to_B",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1.0)),
        topology=topology,
        pre="A",
        post="B",
    )

    engine = TorchNetworkEngine(populations=[pop_a, pop_b], projections=[proj])
    config = SimulationConfig(dt=1e-3, meta={"initial_spikes_by_pop": {"A": [0]}})
    engine.reset(config=config)

    for step in range(3):
        engine.step()
        drive = engine.last_projection_drive["A_to_B"][Compartment.DENDRITE]
        value = float(drive[0].item())
        if step < 2:
            assert value == pytest.approx(0.0)
        else:
            assert value > 0.0
