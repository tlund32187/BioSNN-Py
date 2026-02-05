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
from biosnn.core.torch_utils import resolve_device_dtype
from biosnn.learning import ThreeFactorHebbianParams, ThreeFactorHebbianRule
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    pass


class ConstantSpikesModel(INeuronModel):
    name = "constant_spikes"
    compartments = frozenset({Compartment.SOMA})

    def __init__(self, spikes: Tensor) -> None:
        self._spikes = spikes

    def init_state(self, n: int, *, ctx: StepContext) -> DummyState:
        _ = resolve_device_dtype(ctx)
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
        spikes = self._spikes.to(device=resolve_device_dtype(ctx)[0])
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: DummyState):
        _ = state
        return {}


@dataclass(slots=True)
class SparseCheckState:
    last_active_edges: int
    total_edges: int


class SparseCheckRule(ILearningRule):
    name = "sparse_check"
    supports_sparse = True

    def init_state(self, e: int, *, ctx: StepContext) -> SparseCheckState:
        _ = ctx
        return SparseCheckState(last_active_edges=0, total_edges=e)

    def step(
        self,
        state: SparseCheckState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[SparseCheckState, LearningStepResult]:
        _ = dt, t, ctx
        active = None
        if batch.extras and "active_edges" in batch.extras:
            active = batch.extras["active_edges"]
        if active is not None:
            state.last_active_edges = int(active.numel())
        return state, LearningStepResult(d_weights=torch.zeros_like(batch.weights))

    def state_tensors(self, state: SparseCheckState):
        _ = state
        return {}


def _build_topology():
    pre_idx = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 1, 2, 2, 0], dtype=torch.long)
    weights = torch.full((pre_idx.numel(),), 0.5, dtype=torch.float32)
    return SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, weights=weights)


def _build_engine(*, sparse_learning: bool) -> TorchNetworkEngine:
    pre_spikes = torch.tensor([True, False, False, False])
    post_spikes = torch.tensor([True, True, False])

    pop_pre = PopulationSpec(
        name="Pre",
        model=ConstantSpikesModel(pre_spikes),
        n=pre_spikes.numel(),
        positions=torch.zeros((pre_spikes.numel(), 3)),
    )
    pop_post = PopulationSpec(
        name="Post",
        model=ConstantSpikesModel(post_spikes),
        n=post_spikes.numel(),
        positions=torch.zeros((post_spikes.numel(), 3)),
    )
    topology = _build_topology()
    learning = ThreeFactorHebbianRule(ThreeFactorHebbianParams(lr=0.1))
    synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.5))
    proj = ProjectionSpec(
        name="Pre_to_Post",
        synapse=synapse,
        topology=topology,
        pre="Pre",
        post="Post",
        learning=learning,
        sparse_learning=sparse_learning,
    )
    return TorchNetworkEngine(populations=[pop_pre, pop_post], projections=[proj])


def test_sparse_learning_matches_dense():
    dense = _build_engine(sparse_learning=False)
    sparse = _build_engine(sparse_learning=True)
    dense.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    sparse.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    dense.step()
    sparse.step()
    w_dense = dense._proj_states["Pre_to_Post"].state.weights
    w_sparse = sparse._proj_states["Pre_to_Post"].state.weights
    torch.testing.assert_close(w_dense, w_sparse)


def test_sparse_learning_uses_active_edges():
    pre_spikes = torch.tensor([True, False, False, False])
    post_spikes = torch.tensor([True, True, False])
    pop_pre = PopulationSpec(
        name="Pre",
        model=ConstantSpikesModel(pre_spikes),
        n=pre_spikes.numel(),
        positions=torch.zeros((pre_spikes.numel(), 3)),
    )
    pop_post = PopulationSpec(
        name="Post",
        model=ConstantSpikesModel(post_spikes),
        n=post_spikes.numel(),
        positions=torch.zeros((post_spikes.numel(), 3)),
    )
    topology = _build_topology()
    synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.5))
    proj = ProjectionSpec(
        name="Pre_to_Post",
        synapse=synapse,
        topology=topology,
        pre="Pre",
        post="Post",
        learning=SparseCheckRule(),
        sparse_learning=True,
    )
    engine = TorchNetworkEngine(populations=[pop_pre, pop_post], projections=[proj])
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    engine.step()
    state = engine._proj_states["Pre_to_Post"].learning_state
    assert state is not None
    assert state.last_active_edges < state.total_edges
