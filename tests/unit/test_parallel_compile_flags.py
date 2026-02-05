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


def _make_engine(*, parallel_compile: str, workers: int | None) -> TorchNetworkEngine:
    pop_a = PopulationSpec(name="A", model=DummyNeuron(), n=1)
    pop_b = PopulationSpec(name="B", model=DummyNeuron(), n=1)
    topology = SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long),
        post_idx=torch.tensor([0], dtype=torch.long),
        delay_steps=torch.tensor([0], dtype=torch.int32),
        target_compartment=Compartment.SOMA,
    )
    syn = DelayedCurrentSynapse(DelayedCurrentParams(event_driven=False))
    proj_ab = ProjectionSpec(name="AB", synapse=syn, topology=topology, pre="A", post="B")
    proj_ba = ProjectionSpec(name="BA", synapse=syn, topology=topology, pre="B", post="A")
    return TorchNetworkEngine(
        populations=[pop_a, pop_b],
        projections=[proj_ab, proj_ba],
        parallel_compile=parallel_compile,
        parallel_compile_workers=workers,
        parallel_compile_torch_threads=1,
    )


def test_parallel_compile_on_uses_executor(monkeypatch):
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
            DummyExecutor.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr(tne, "ThreadPoolExecutor", DummyExecutor)

    engine = _make_engine(parallel_compile="on", workers=2)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))

    assert DummyExecutor.instances
    assert DummyExecutor.instances[0].max_workers == 2
    assert [proj.name for proj in engine._proj_specs] == ["AB", "BA"]


def test_parallel_compile_off_is_sequential(monkeypatch):
    import biosnn.simulation.engine.torch_network_engine as tne

    class DummyExecutor:
        instances: list[DummyExecutor] = []

        def __init__(self, max_workers=None):
            DummyExecutor.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            raise RuntimeError("should not be called when parallel compile is off")

    monkeypatch.setattr(tne, "ThreadPoolExecutor", DummyExecutor)

    engine = _make_engine(parallel_compile="off", workers=None)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))

    assert DummyExecutor.instances == []
