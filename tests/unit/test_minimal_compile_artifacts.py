from __future__ import annotations

from typing import cast

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
from biosnn.monitors.weights.projection_weights_csv import ProjectionWeightsCSVMonitor
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
        _ = state
        return {}


class _DummyLearning(ILearningRule):
    name = "dummy_learning"
    supports_sparse = True

    def init_state(self, e: int, *, ctx: StepContext) -> None:
        _ = e, ctx
        return None

    def step(
        self,
        state: None,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[None, LearningStepResult]:
        _ = state, dt, t, ctx
        d_weights = torch.zeros_like(batch.weights)
        return None, LearningStepResult(d_weights=d_weights)

    def state_tensors(self, state: None) -> dict[str, Tensor]:
        _ = state
        return {}


def _build_engine(
    tmp_path,
    *,
    with_learning: bool,
    with_weights_monitor: bool,
) -> tuple[TorchNetworkEngine, str]:
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
        learning=_DummyLearning() if with_learning else None,
        sparse_learning=with_learning,
    )
    engine = TorchNetworkEngine(populations=[pop], projections=[proj])
    weights_path = str(tmp_path / "weights.csv")
    monitors = (
        [ProjectionWeightsCSVMonitor(weights_path, projections=[proj], stride=1)]
        if with_weights_monitor
        else []
    )
    engine.attach_monitors(monitors)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    return engine, weights_path


def test_minimal_compile_without_learning_or_weight_monitor(tmp_path) -> None:
    engine, _ = _build_engine(tmp_path, with_learning=False, with_weights_monitor=False)
    meta = engine._proj_specs[0].topology.meta or {}

    assert "edge_bucket_comp" not in meta
    assert "edge_bucket_delay" not in meta
    assert "edge_bucket_pos" not in meta
    assert "edge_bucket_fused_pos" not in meta
    assert "W_by_delay_by_comp" not in meta
    assert "W_by_delay_by_comp_csr" not in meta
    assert "values_by_comp" not in meta
    assert "indices_by_comp" not in meta
    assert "fused_W_by_comp" in meta
    assert "fused_W_by_comp_csr" in meta


def test_minimal_compile_builds_bucket_mapping_with_learning(tmp_path) -> None:
    engine, _ = _build_engine(tmp_path, with_learning=True, with_weights_monitor=False)
    meta = engine._proj_specs[0].topology.meta or {}

    assert "edge_bucket_comp" in meta
    assert "edge_bucket_delay" in meta
    assert "edge_bucket_pos" in meta
    assert "edge_bucket_fused_pos" in meta
    assert "W_by_delay_by_comp" not in meta
    assert "W_by_delay_by_comp_csr" not in meta
    assert "fused_W_by_comp" in meta


def test_minimal_compile_with_weight_monitor_keeps_weight_exports(tmp_path) -> None:
    engine, _ = _build_engine(tmp_path, with_learning=False, with_weights_monitor=True)
    meta = engine._proj_specs[0].topology.meta or {}

    assert "edge_bucket_comp" not in meta
    assert "W_by_delay_by_comp" not in meta
    assert "fused_W_by_comp" in meta

    engine.run(steps=2)
    last_event = engine._last_event
    assert last_event is not None and last_event.tensors is not None
    assert "proj/P/weights" in last_event.tensors
    assert (tmp_path / "weights.csv").exists()
    assert (tmp_path / "weights.csv").stat().st_size > 0
