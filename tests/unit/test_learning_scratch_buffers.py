from __future__ import annotations

from dataclasses import dataclass

import pytest
pytestmark = pytest.mark.unit


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

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    pass


class FixedSpikesModel(INeuronModel):
    name = "fixed_spikes"
    compartments = frozenset({Compartment.SOMA})

    def __init__(self, spikes: Tensor) -> None:
        self._spikes = spikes

    def init_state(self, n: int, *, ctx: StepContext) -> DummyState:
        _ = n, ctx
        return DummyState()

    def reset_state(self, state: DummyState, *, ctx: StepContext, indices: Tensor | None = None) -> DummyState:
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
        _ = state, inputs, dt, t, ctx
        return state, NeuronStepResult(spikes=self._spikes)

    def state_tensors(self, state: DummyState):
        _ = state
        return {}


class SparseOnesRule(ILearningRule):
    name = "sparse_ones"
    supports_sparse = True

    @dataclass(slots=True)
    class _State:
        pass

    def init_state(self, e: int, *, ctx: StepContext):
        _ = e, ctx
        return self._State()

    def step(
        self,
        state: _State,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[_State, LearningStepResult]:
        _ = dt, t, ctx
        dw = torch.ones_like(batch.weights) * 0.5
        return state, LearningStepResult(d_weights=dw)

    def state_tensors(self, state: _State):
        _ = state
        return {}


def _build_engine(*, use_scratch: bool) -> TorchNetworkEngine:
    pre_spikes = torch.tensor([True, False, False], dtype=torch.bool)
    post_spikes = torch.tensor([False, False], dtype=torch.bool)
    pop_pre = PopulationSpec(name="Pre", model=FixedSpikesModel(pre_spikes), n=3)
    pop_post = PopulationSpec(name="Post", model=FixedSpikesModel(post_spikes), n=2)

    pre_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    topology = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, target_compartment=Compartment.SOMA)

    syn = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.0))
    proj = ProjectionSpec(
        name="P",
        synapse=syn,
        topology=topology,
        pre="Pre",
        post="Post",
        learning=SparseOnesRule(),
        sparse_learning=True,
    )
    return TorchNetworkEngine(
        populations=[pop_pre, pop_post],
        projections=[proj],
        learning_use_scratch=use_scratch,
    )


def test_learning_scratch_matches_no_scratch():
    engine_ref = _build_engine(use_scratch=False)
    engine_opt = _build_engine(use_scratch=True)
    config = SimulationConfig(dt=1e-3, device="cpu")
    engine_ref.reset(config=config)
    engine_opt.reset(config=config)
    engine_ref.step()
    engine_opt.step()
    w_ref = engine_ref._proj_states["P"].state.weights
    w_opt = engine_opt._proj_states["P"].state.weights
    torch.testing.assert_close(w_ref, w_opt)
    expected = torch.zeros_like(w_ref)
    expected[0] = 0.5
    expected[3] = 0.5
    torch.testing.assert_close(w_opt, expected)


def test_learning_scratch_buffers_reused():
    engine = _build_engine(use_scratch=True)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    engine.step()
    scratch = engine._learning_scratch["P"]
    ids = (
        id(scratch.edge_pre_idx),
        id(scratch.edge_post_idx),
        id(scratch.edge_pre),
        id(scratch.edge_post),
        id(scratch.edge_weights),
        id(scratch.arange_buf),
    )
    for _ in range(3):
        engine.step()
    scratch_after = engine._learning_scratch["P"]
    ids_after = (
        id(scratch_after.edge_pre_idx),
        id(scratch_after.edge_post_idx),
        id(scratch_after.edge_pre),
        id(scratch_after.edge_post),
        id(scratch_after.edge_weights),
        id(scratch_after.arange_buf),
    )
    assert ids == ids_after


def test_learning_scratch_arange_reused():
    engine = _build_engine(use_scratch=True)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    engine.step()
    scratch = engine._learning_scratch["P"]
    arange_id = id(scratch.arange_buf)
    for _ in range(3):
        engine.step()
    scratch_after = engine._learning_scratch["P"]
    assert id(scratch_after.arange_buf) == arange_id
