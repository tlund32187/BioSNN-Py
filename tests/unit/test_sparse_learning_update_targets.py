from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _build_topology(device: str):
    pre_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=device)
    post_idx = torch.tensor([0, 1, 0, 2], dtype=torch.long, device=device)
    delay_steps = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    target_compartments = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    weights = torch.tensor([0.5, 0.25, 0.75, 0.1], dtype=torch.float32, device=device)
    topo = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
        target_compartments=target_compartments,
        weights=weights,
    )
    return compile_topology(
        topo,
        device=device,
        dtype="float32",
        build_sparse_delay_mats=True,
        build_bucket_edge_mapping=True,
        fuse_delay_buckets=True,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_learning_update_targets_match(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ctx = StepContext(device=device, dtype="float32")
    topo_both = _build_topology(device)
    topo_fused = _build_topology(device)

    syn_both = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(learning_update_target="both"))
    syn_fused = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(learning_update_target="fused_only")
    )

    active_edges = torch.tensor([0, 2], dtype=torch.long, device=device)
    d_weights = torch.tensor([0.2, -0.1], dtype=torch.float32, device=device)
    syn_both.apply_weight_updates(topo_both, active_edges, d_weights)
    syn_fused.apply_weight_updates(topo_fused, active_edges, d_weights)

    pre_spikes = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32, device=device)

    state_both = syn_both.init_state(topo_both.pre_idx.numel(), ctx=ctx)
    state_fused = syn_fused.init_state(topo_fused.pre_idx.numel(), ctx=ctx)

    state_both, out_both = syn_both.step(
        state_both,
        topo_both,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )
    state_fused, out_fused = syn_fused.step(
        state_fused,
        topo_fused,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert out_both.post_drive.keys() == out_fused.post_drive.keys()
    for comp in out_both.post_drive:
        torch.testing.assert_close(out_both.post_drive[comp], out_fused.post_drive[comp])
