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
        return state, NeuronStepResult(spikes=self._spikes)

    def state_tensors(self, state: DummyState):
        _ = state
        return {}


@dataclass(slots=True)
class SparseDeltaState:
    pass


class SparseDeltaRule(ILearningRule):
    name = "sparse_delta"
    supports_sparse = True

    def init_state(self, e: int, *, ctx: StepContext) -> SparseDeltaState:
        _ = e, ctx
        return SparseDeltaState()

    def step(
        self,
        state: SparseDeltaState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[SparseDeltaState, LearningStepResult]:
        _ = dt, t, ctx
        dw = torch.ones_like(batch.weights) * 0.5
        return state, LearningStepResult(d_weights=dw)

    def state_tensors(self, state: SparseDeltaState):
        _ = state
        return {}


def _build_engine(*, compiled_mode: bool) -> TorchNetworkEngine:
    pre_spikes = torch.tensor([True, False, False])
    post_spikes = torch.tensor([True, False])

    pop_pre = PopulationSpec(name="Pre", model=FixedSpikesModel(pre_spikes), n=pre_spikes.numel())
    pop_post = PopulationSpec(name="Post", model=FixedSpikesModel(post_spikes), n=post_spikes.numel())

    pre_idx = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    weights = torch.zeros((pre_idx.numel(),), dtype=torch.float32)
    topology = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, weights=weights)

    synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.0))
    proj = ProjectionSpec(
        name="P",
        synapse=synapse,
        topology=topology,
        pre="Pre",
        post="Post",
        learning=SparseDeltaRule(),
        sparse_learning=True,
    )

    return TorchNetworkEngine(
        populations=[pop_pre, pop_post],
        projections=[proj],
        compiled_mode=compiled_mode,
    )


def test_sparse_learning_update_matches_compiled() -> None:
    engine_dense = _build_engine(compiled_mode=False)
    engine_compiled = _build_engine(compiled_mode=True)

    config = SimulationConfig(dt=1e-3, device="cpu")
    engine_dense.reset(config=config)
    engine_compiled.reset(config=config)

    engine_dense.step()
    engine_compiled.step()

    w_dense = engine_dense._proj_states["P"].state.weights
    w_compiled = engine_compiled._proj_states["P"].state.weights
    torch.testing.assert_close(w_dense, w_compiled)

    expected = torch.zeros_like(w_dense)
    expected[0] = 0.5
    torch.testing.assert_close(w_dense, expected)
