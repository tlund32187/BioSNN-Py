from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

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
from biosnn.contracts.synapses import ReceptorKind, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    pass


class FixedSpikesModel(INeuronModel):
    name = "fixed_spikes"
    compartments = frozenset({Compartment.SOMA, Compartment.DENDRITE})

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
        dw = torch.full_like(batch.weights, 0.5)
        return state, LearningStepResult(d_weights=dw)

    def state_tensors(self, state: SparseDeltaState):
        _ = state
        return {}


def _compiled_edge_value(meta: Mapping[str, Any], *, edge_idx: int) -> Tensor:
    edge_bucket_comp = cast(Tensor, meta["edge_bucket_comp"])
    edge_bucket_delay = cast(Tensor, meta["edge_bucket_delay"])
    edge_bucket_pos = cast(Tensor, meta["edge_bucket_pos"])

    comp_id = int(edge_bucket_comp[edge_idx].item())
    delay = int(edge_bucket_delay[edge_idx].item())
    pos = int(edge_bucket_pos[edge_idx].item())
    comp = tuple(Compartment)[comp_id]

    values_by_comp = cast(Any, meta.get("values_by_comp"))
    if isinstance(values_by_comp, dict):
        values = values_by_comp[comp][delay]
        assert values is not None
        return cast(Tensor, values[pos])

    fused_by_comp = cast(Any, meta.get("fused_W_by_comp"))
    edge_bucket_fused_pos = cast(Any, meta.get("edge_bucket_fused_pos"))
    assert isinstance(fused_by_comp, dict)
    assert edge_bucket_fused_pos is not None
    fused_values = fused_by_comp[comp].values()
    fused_pos = int(edge_bucket_fused_pos[edge_idx].item())
    return cast(Tensor, fused_values[fused_pos])


def _build_engine(*, compiled_mode: bool, clamp_max: float | None = None) -> TorchNetworkEngine:
    pre_spikes = torch.tensor([True, False, False])
    post_spikes = torch.tensor([False, False])

    pop_pre = PopulationSpec(name="Pre", model=FixedSpikesModel(pre_spikes), n=pre_spikes.numel())
    pop_post = PopulationSpec(name="Post", model=FixedSpikesModel(post_spikes), n=post_spikes.numel())

    pre_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 0, 1], dtype=torch.long)
    delay_steps = torch.zeros_like(pre_idx)
    receptor = torch.tensor([0, 1, 0], dtype=torch.long)
    weights = torch.zeros((pre_idx.numel(),), dtype=torch.float32)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        receptor=receptor,
        receptor_kinds=(ReceptorKind.AMPA, ReceptorKind.NMDA),
        weights=weights,
    )

    receptor_scale = {ReceptorKind.AMPA: 2.0, ReceptorKind.NMDA: 0.5}
    synapse = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            init_weight=0.0,
            receptor_scale=receptor_scale,
            clamp_max=clamp_max,
        )
    )
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


@pytest.mark.parametrize("compiled_mode", [False, True])
def test_sparse_matmul_learning_updates(compiled_mode: bool) -> None:
    engine = _build_engine(compiled_mode=compiled_mode)
    config = SimulationConfig(dt=1e-3, device="cpu")
    engine.reset(config=config)

    engine.step()
    drive_first = engine.last_projection_drive["P"][Compartment.DENDRITE].clone()
    torch.testing.assert_close(drive_first, torch.zeros_like(drive_first))

    weights_after_first = engine._proj_states["P"].state.weights.clone()
    topo_weights_after_first = engine._proj_specs[0].topology.weights
    assert topo_weights_after_first is not None

    expected_weights = torch.zeros_like(weights_after_first)
    expected_weights[0] = 0.5
    torch.testing.assert_close(weights_after_first, expected_weights)
    torch.testing.assert_close(topo_weights_after_first, expected_weights)

    meta = engine._proj_specs[0].topology.meta or {}
    edge_scale = meta["edge_scale"]

    expected_val = weights_after_first[0] * edge_scale[0]
    compiled_val = _compiled_edge_value(meta, edge_idx=0)
    torch.testing.assert_close(compiled_val, expected_val)

    engine.step()
    drive_second = engine.last_projection_drive["P"][Compartment.DENDRITE].clone()
    expected_drive = torch.zeros_like(drive_second)
    expected_drive[0] = expected_val
    torch.testing.assert_close(drive_second, expected_drive)


@pytest.mark.parametrize("compiled_mode", [False, True])
def test_sparse_matmul_clamp_and_weight_binding(compiled_mode: bool) -> None:
    engine = _build_engine(compiled_mode=compiled_mode, clamp_max=0.25)
    config = SimulationConfig(dt=1e-3, device="cpu")
    engine.reset(config=config)

    engine.step()

    proj_state = engine._proj_states["P"].state
    topo_weights = engine._proj_specs[0].topology.weights
    assert topo_weights is not None
    assert proj_state.weights is topo_weights

    expected = torch.zeros_like(topo_weights)
    expected[0] = 0.25
    torch.testing.assert_close(topo_weights, expected)

    meta = engine._proj_specs[0].topology.meta or {}
    edge_scale = meta["edge_scale"]

    compiled_val = _compiled_edge_value(meta, edge_idx=0)
    torch.testing.assert_close(compiled_val, expected[0] * edge_scale[0])
