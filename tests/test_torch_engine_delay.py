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
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import resolve_device_dtype
from biosnn.simulation.engine import TorchSimulationEngine
from biosnn.synapses.dynamics.delayed_current import (
    DelayedCurrentParams,
    DelayedCurrentSynapse,
)

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    dummy: Tensor


class DummyNeuronModel(INeuronModel):
    name = "dummy"
    compartments = frozenset({Compartment.SOMA})

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


def test_engine_delay_semantics():
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([1], dtype=torch.long)
    delay_steps = torch.tensor([3], dtype=torch.int32)
    weights = torch.tensor([1.0], dtype=torch.float32)

    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        weights=weights,
    )

    synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1.0))
    engine = TorchSimulationEngine(
        neuron_model=DummyNeuronModel(),
        synapse_model=synapse,
        topology=topology,
        n=2,
    )

    config = SimulationConfig(dt=1e-3, meta={"initial_spike_indices": [0]})
    engine.reset(config=config)

    for step in range(4):
        engine.step()
        post_drive = engine.last_post_drive
        assert post_drive is not None
        drive = post_drive[Compartment.DENDRITE]
        value = float(drive[1].item())
        if step < 3:
            assert value == pytest.approx(0.0)
        else:
            assert value > 0.0
