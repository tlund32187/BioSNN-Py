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
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    pass


class DummyNeuron(INeuronModel):
    name = "dummy"
    compartments = frozenset({Compartment.SOMA})

    def init_state(self, n: int, *, ctx: StepContext) -> DummyState:
        _ = n, ctx
        return DummyState()

    def reset_state(
        self,
        state: DummyState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
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
        _ = inputs, dt, t, ctx
        spikes = torch.zeros((1,), dtype=torch.bool)
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: DummyState):
        _ = state
        return {}


def _make_topology(tag: str) -> SynapseTopology:
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    delay_steps = torch.zeros_like(pre_idx)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        meta={"tag": tag},
    )


def test_reset_compilation_parallel_preserves_order(monkeypatch):
    import biosnn.simulation.engine.torch_network_engine as tne

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class DummyExecutor:
        instances: list[DummyExecutor] = []

        def __init__(self, max_workers=None):
            self.max_workers = max_workers
            self.submitted: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
            DummyExecutor.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            self.submitted.append((fn, args, kwargs))
            return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr(tne, "ThreadPoolExecutor", DummyExecutor)

    neuron = DummyNeuron()
    pop_a = PopulationSpec(name="A", model=neuron, n=1)
    pop_b = PopulationSpec(name="B", model=neuron, n=1)
    pop_c = PopulationSpec(name="C", model=neuron, n=1)

    synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.0))
    proj_ab = ProjectionSpec(
        name="AB",
        synapse=synapse,
        topology=_make_topology("AB"),
        pre="A",
        post="B",
    )
    proj_bc = ProjectionSpec(
        name="BC",
        synapse=synapse,
        topology=_make_topology("BC"),
        pre="B",
        post="C",
    )
    proj_ac = ProjectionSpec(
        name="AC",
        synapse=synapse,
        topology=_make_topology("AC"),
        pre="A",
        post="C",
    )

    engine = TorchNetworkEngine(
        populations=[pop_a, pop_b, pop_c],
        projections=[proj_ab, proj_bc, proj_ac],
    )
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))

    assert [proj.name for proj in engine._proj_specs] == ["AB", "BC", "AC"]
    tags = [proj.topology.meta["tag"] for proj in engine._proj_specs if proj.topology.meta]
    assert tags == ["AB", "BC", "AC"]

    assert DummyExecutor.instances
    assert DummyExecutor.instances[0].submitted
    assert len(DummyExecutor.instances[0].submitted) == 3
