from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import build_erdos_renyi_topology
from biosnn.contracts.homeostasis import HomeostasisPopulation
from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.neurons import StepContext, Tensor
from biosnn.contracts.simulation import SimulationConfig
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


class _MarkerHomeostasis:
    name = "marker_homeostasis"

    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def init(
        self,
        populations: Sequence[HomeostasisPopulation],
        *,
        device: Any,
        dtype: Any,
        ctx: StepContext,
    ) -> None:
        _ = (populations, device, dtype, ctx)

    def step(
        self,
        spikes_by_pop: Mapping[str, Tensor],
        *,
        dt: float,
        ctx: StepContext,
    ) -> Mapping[str, Tensor]:
        _ = (spikes_by_pop, dt, ctx)
        self._trace.append("homeostasis")
        return {}

    def state_tensors(self) -> Mapping[str, Tensor]:
        return {}


@dataclass(slots=True)
class _MarkerLearningState:
    count: int = 0


class _MarkerLearningRule(ILearningRule):
    name = "marker_learning"
    supports_sparse = False

    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def init_state(self, e: int, *, ctx: StepContext) -> _MarkerLearningState:
        _ = (e, ctx)
        return _MarkerLearningState()

    def step(
        self,
        state: _MarkerLearningState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[_MarkerLearningState, LearningStepResult]:
        _ = (dt, t, ctx)
        self._trace.append("learning")
        state.count += 1
        return state, LearningStepResult(d_weights=torch.zeros_like(batch.weights))

    def state_tensors(self, state: _MarkerLearningState) -> dict[str, Tensor]:
        _ = state
        return {}


def test_homeostasis_runs_before_learning() -> None:
    trace: list[str] = []
    pop = PopulationSpec(name="Pop", model=GLIFModel(), n=16)
    topology = build_erdos_renyi_topology(n=16, p=0.5, allow_self=False, dt=1e-3)
    synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.05))
    learning = _MarkerLearningRule(trace)
    proj = ProjectionSpec(
        name="Pop->Pop",
        synapse=synapse,
        topology=topology,
        pre="Pop",
        post="Pop",
        learning=learning,
    )

    engine = TorchNetworkEngine(
        populations=[pop],
        projections=[proj],
        homeostasis=_MarkerHomeostasis(trace),
    )
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu", dtype="float32", seed=7))
    engine.step()

    assert "homeostasis" in trace
    assert "learning" in trace
    assert trace.index("homeostasis") < trace.index("learning")
