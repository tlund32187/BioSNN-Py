from __future__ import annotations

from typing import cast

import pytest

from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
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
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


class _DummyNeuron(INeuronModel):
    name = "dummy"
    compartments = frozenset({Compartment.SOMA})

    def init_state(self, n: int, *, ctx: StepContext) -> Tensor:
        _ = ctx
        return cast(Tensor, torch.zeros((n,), dtype=torch.float32))

    def reset_state(self, state: Tensor, *, ctx: StepContext, indices: Tensor | None = None) -> Tensor:
        _ = ctx, indices
        state.zero_()
        return state

    def step(
        self,
        state: Tensor,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[Tensor, NeuronStepResult]:
        _ = inputs, dt, t, ctx
        spikes = torch.zeros_like(state, dtype=torch.bool)
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: Tensor) -> dict[str, Tensor]:
        return {"v_soma": state}


class _SpikesOnlyMonitor(IMonitor):
    name = "spikes_only"

    def __init__(self) -> None:
        self.last_event: StepEvent | None = None

    def requirements(self) -> MonitorRequirements:
        return MonitorRequirements(needs_spikes=True)

    def on_step(self, event: StepEvent) -> None:
        self.last_event = event

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _WeightsOnlyMonitor(IMonitor):
    name = "weights_only"

    def __init__(self) -> None:
        self.last_event: StepEvent | None = None

    def requirements(self) -> MonitorRequirements:
        return MonitorRequirements(needs_projection_weights=True, needs_scalars=True)

    def on_step(self, event: StepEvent) -> None:
        self.last_event = event

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _build_engine(monitors: list[IMonitor]) -> TorchNetworkEngine:
    pop = PopulationSpec(name="Pop", model=_DummyNeuron(), n=4)
    topology = SynapseTopology(
        pre_idx=torch.tensor([0, 1, 2], dtype=torch.long),
        post_idx=torch.tensor([1, 2, 3], dtype=torch.long),
        delay_steps=torch.tensor([0, 1, 2], dtype=torch.int32),
        target_compartment=Compartment.SOMA,
    )
    proj = ProjectionSpec(
        name="P",
        synapse=DelayedSparseMatmulSynapse(DelayedSparseMatmulParams()),
        topology=topology,
        pre="Pop",
        post="Pop",
    )
    engine = TorchNetworkEngine(populations=[pop], projections=[proj])
    engine.attach_monitors(monitors)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    return engine


def test_monitor_requirements_spikes_only_event_excludes_weights() -> None:
    monitor = _SpikesOnlyMonitor()
    engine = _build_engine([monitor])
    engine.step()

    event = monitor.last_event
    assert event is not None
    assert event.spikes is not None
    assert event.tensors is None or "proj/P/weights" not in event.tensors


def test_monitor_requirements_weight_monitor_populates_projection_weights() -> None:
    monitor = _WeightsOnlyMonitor()
    engine = _build_engine([monitor])
    engine.step()

    event = monitor.last_event
    assert event is not None
    assert event.tensors is not None
    assert "proj/P/weights" in event.tensors
