from __future__ import annotations

from dataclasses import dataclass

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class _StepResult:
    post_drive: dict[Compartment, Tensor]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_direct_fused_builder_equivalence(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    n_pre = 8
    n_post = 6
    edge_count = 20

    pre_idx = torch.randint(0, n_pre, (edge_count,), device=device, dtype=torch.long)
    post_idx = torch.randint(0, n_post, (edge_count,), device=device, dtype=torch.long)
    delay_steps = torch.randint(0, 3, (edge_count,), device=device, dtype=torch.long)
    comp_ids = torch.randint(0, len(tuple(Compartment)), (edge_count,), device=device, dtype=torch.long)
    weights = torch.randn((edge_count,), device=device, dtype=torch.float32)

    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartments=comp_ids,
        weights=weights,
    )

    topo_store = compile_topology(
        topology,
        device=device,
        dtype="float32",
        build_sparse_delay_mats=True,
        build_bucket_edge_mapping=True,
        fuse_delay_buckets=True,
        store_sparse_by_delay=True,
    )
    topo_direct = compile_topology(
        topology,
        device=device,
        dtype="float32",
        build_sparse_delay_mats=True,
        build_bucket_edge_mapping=True,
        fuse_delay_buckets=True,
        store_sparse_by_delay=False,
    )

    assert topo_direct.meta is not None
    assert "values_by_comp" not in topo_direct.meta
    assert topo_store.meta is not None
    assert topo_store.meta.get("edge_bucket_fused_pos") is not None
    assert topo_direct.meta.get("edge_bucket_fused_pos") is not None
    torch.testing.assert_close(
        topo_store.meta["edge_bucket_fused_pos"],
        topo_direct.meta["edge_bucket_fused_pos"],
    )

    synapse = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=0.0))
    ctx = StepContext(device=device, dtype="float32")
    state_store = synapse.init_state(edge_count, ctx=ctx)
    state_direct = synapse.init_state(edge_count, ctx=ctx)
    state_store.weights.copy_(weights)
    state_direct.weights.copy_(weights)

    pre_spikes = torch.rand((n_pre,), device=device) < 0.5
    inputs = SynapseInputs(pre_spikes=pre_spikes)

    _, res_store = synapse.step(state_store, topo_store, inputs, dt=1e-3, t=0.0, ctx=ctx)
    _, res_direct = synapse.step(state_direct, topo_direct, inputs, dt=1e-3, t=0.0, ctx=ctx)

    assert res_store.post_drive.keys() == res_direct.post_drive.keys()
    for comp in res_store.post_drive:
        torch.testing.assert_close(res_store.post_drive[comp], res_direct.post_drive[comp])
