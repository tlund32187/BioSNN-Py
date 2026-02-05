from __future__ import annotations

from pathlib import Path

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
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

torch = pytest.importorskip("torch")


class DummyNeuron(INeuronModel):
    name = "dummy"
    compartments = frozenset({Compartment.SOMA})

    def init_state(self, n: int, *, ctx: StepContext):
        _ = n, ctx
        return None

    def reset_state(self, state, *, ctx: StepContext, indices: Tensor | None = None):
        _ = state, ctx, indices
        return state

    def step(
        self,
        state,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ):
        _ = state, inputs, dt, t, ctx
        spikes = torch.zeros((1,), dtype=torch.bool)
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state):
        _ = state
        return {}


class DummyLearning(ILearningRule):
    name = "dummy_learning"
    supports_sparse = True

    def init_state(self, e: int, *, ctx: StepContext):
        _ = e, ctx
        return None

    def step(
        self,
        state,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[None, LearningStepResult]:
        _ = state, batch, dt, t, ctx
        return None, LearningStepResult(d_weights=batch.weights)

    def state_tensors(self, state):
        _ = state
        return {}


def _build_engine(synapse, *, learning: ILearningRule | None = None) -> TorchNetworkEngine:
    pop = PopulationSpec(name="Pop", model=DummyNeuron(), n=1)
    topology = SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long),
        post_idx=torch.tensor([0], dtype=torch.long),
        delay_steps=torch.tensor([0], dtype=torch.int32),
        target_compartment=Compartment.SOMA,
    )
    proj = ProjectionSpec(
        name="P",
        synapse=synapse,
        topology=topology,
        pre="Pop",
        post="Pop",
        learning=learning,
        sparse_learning=learning is not None,
    )
    return TorchNetworkEngine(populations=[pop], projections=[proj])


def test_compilation_flags_for_delayed_current():
    syn = DelayedCurrentSynapse(DelayedCurrentParams(event_driven=False))
    engine = _build_engine(syn)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    meta = engine._proj_specs[0].topology.meta or {}
    assert "edges_by_delay" in meta


def test_compilation_flags_for_sparse_matmul():
    syn = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams())
    engine = _build_engine(syn, learning=None)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    meta = engine._proj_specs[0].topology.meta or {}
    assert "W_by_delay_by_comp_csr" in meta or "W_by_delay_by_comp" in meta
    assert "edge_bucket_comp" not in meta
    assert "edge_bucket_delay" not in meta
    assert "edge_bucket_pos" not in meta
    assert "fused_W_by_comp" in meta
    assert "fused_W_delays_by_comp" in meta
    assert "fused_W_n_post_by_comp" in meta
    assert "edge_bucket_fused_pos" not in meta


def test_sparse_matmul_builds_bucket_mappings_with_learning():
    syn = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams())
    engine = _build_engine(syn, learning=DummyLearning())
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    meta = engine._proj_specs[0].topology.meta or {}
    assert "edge_bucket_comp" in meta
    assert "edge_bucket_delay" in meta
    assert "edge_bucket_pos" in meta
    assert "edge_bucket_fused_pos" in meta


def test_engine_has_no_concrete_synapse_imports():
    path = Path(__file__).resolve().parents[2] / "src" / "biosnn" / "simulation" / "engine"
    text = (path / "torch_network_engine.py").read_text(encoding="utf-8")
    assert "delayed_current" not in text
    assert "delayed_sparse_matmul" not in text
