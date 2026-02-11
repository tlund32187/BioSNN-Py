from __future__ import annotations

from dataclasses import dataclass

import pytest

from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
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

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class _DummyState:
    pass


class _FixedSpikesModel(INeuronModel):
    name = "fixed_spikes"
    compartments = frozenset({Compartment.SOMA})

    def __init__(self, spikes: Tensor) -> None:
        self._spikes = spikes

    def init_state(self, n: int, *, ctx: StepContext) -> _DummyState:
        _ = n, ctx
        return _DummyState()

    def reset_state(
        self,
        state: _DummyState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> _DummyState:
        _ = ctx, indices
        return state

    def step(
        self,
        state: _DummyState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[_DummyState, NeuronStepResult]:
        _ = inputs, dt, t, ctx
        return state, NeuronStepResult(spikes=self._spikes)

    def state_tensors(self, state: _DummyState):
        _ = state
        return {}


@dataclass(slots=True)
class _DtCaptureState:
    dts: list[float]


class _DtCaptureRule(ILearningRule):
    name = "dt_capture"
    supports_sparse = False

    def init_state(self, e: int, *, ctx: StepContext) -> _DtCaptureState:
        _ = e, ctx
        return _DtCaptureState(dts=[])

    def step(
        self,
        state: _DtCaptureState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[_DtCaptureState, LearningStepResult]:
        _ = t, ctx
        state.dts.append(float(dt))
        return state, LearningStepResult(d_weights=torch.zeros_like(batch.weights))

    def state_tensors(self, state: _DtCaptureState):
        _ = state
        return {}


def _build_engine(*, compiled_mode: bool, learn_every: int) -> TorchNetworkEngine:
    pre_spikes = torch.tensor([True], dtype=torch.bool)
    post_spikes = torch.tensor([True], dtype=torch.bool)
    pop_pre = PopulationSpec(name="Pre", model=_FixedSpikesModel(pre_spikes), n=1)
    pop_post = PopulationSpec(name="Post", model=_FixedSpikesModel(post_spikes), n=1)
    topology = SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long),
        post_idx=torch.tensor([0], dtype=torch.long),
        weights=torch.zeros((1,), dtype=torch.float32),
    )
    projection = ProjectionSpec(
        name="P",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.0)),
        topology=topology,
        pre="Pre",
        post="Post",
        learning=_DtCaptureRule(),
        sparse_learning=False,
        learn_every=learn_every,
    )
    return TorchNetworkEngine(
        populations=[pop_pre, pop_post],
        projections=[projection],
        compiled_mode=compiled_mode,
    )


@pytest.mark.parametrize("compiled_mode", [False, True])
def test_learning_dt_scales_by_learn_every(compiled_mode: bool) -> None:
    sim_dt = 1e-3
    learn_every = 4
    engine = _build_engine(compiled_mode=compiled_mode, learn_every=learn_every)
    engine.reset(config=SimulationConfig(dt=sim_dt, device="cpu"))

    for _ in range(10):
        engine.step()

    state = engine._proj_states["P"].learning_state
    assert isinstance(state, _DtCaptureState)
    assert len(state.dts) == 3
    for observed in state.dts:
        assert observed == pytest.approx(sim_dt * learn_every)

