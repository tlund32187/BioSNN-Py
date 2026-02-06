from __future__ import annotations

import pytest

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.synapses import SynapseTopology
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)
from tests.support.test_models import DeterministicLIFModel, SpikeInputModel

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _build_engine(*, n_post: int, delay_step: int) -> TorchNetworkEngine:
    positions_pre = torch.zeros((1, 3), dtype=torch.float64)
    positions_post = torch.zeros((n_post, 3), dtype=torch.float64)
    populations = [
        PopulationSpec(name="Pre", model=SpikeInputModel(), n=1, positions=positions_pre),
        PopulationSpec(
            name="Post",
            model=DeterministicLIFModel(leak=0.0, gain=1.0, thresh=0.5, reset=0.0),
            n=n_post,
            positions=positions_post,
        ),
    ]
    topology = SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long),
        post_idx=torch.tensor([0], dtype=torch.long),
        delay_steps=torch.tensor([delay_step], dtype=torch.long),
        weights=torch.ones((1,), dtype=torch.float64),
        target_compartment=Compartment.DENDRITE,
    )
    projection = ProjectionSpec(
        name="Pre_to_Post",
        synapse=DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0)),
        topology=topology,
        pre="Pre",
        post="Post",
    )
    return TorchNetworkEngine(populations=populations, projections=[projection], fast_mode=True)


def test_ring_buffer_guard_raises() -> None:
    engine = _build_engine(n_post=1024, delay_step=255)
    config = SimulationConfig(dt=1e-3, device="cpu", dtype="float64", max_ring_mib=1.0)
    with pytest.raises(RuntimeError, match="ring buffer"):
        engine.reset(config=config)


def test_ring_buffer_meta_fields() -> None:
    engine = _build_engine(n_post=4, delay_step=1)
    config = SimulationConfig(dt=1e-3, device="cpu", dtype="float64", max_ring_mib=1.0)
    engine.reset(config=config)
    meta = engine._proj_specs[0].topology.meta or {}
    assert meta.get("ring_len") == 2
    assert meta.get("estimated_ring_bytes") is not None
    assert meta.get("estimated_ring_mib") is not None
    expected_bytes = 2 * 4 * 1 * 8
    assert int(meta["estimated_ring_bytes"]) == expected_bytes
    assert float(meta["estimated_ring_mib"]) == pytest.approx(
        expected_bytes / (1024.0 * 1024.0)
    )
