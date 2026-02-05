from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentSynapse
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

torch = pytest.importorskip("torch")


def test_fused_only_storage_learning_updates():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 0, 2], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    target_compartments = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    weights = torch.tensor([0.5, 0.25, 0.75, 0.1], dtype=torch.float32)

    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
        target_compartments=target_compartments,
        weights=weights,
    )

    compiled = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
        build_sparse_delay_mats=True,
        build_bucket_edge_mapping=True,
        fuse_delay_buckets=True,
        store_sparse_by_delay=False,
    )
    meta = compiled.meta
    assert meta is not None
    assert meta.get("values_by_comp") is None
    assert meta.get("indices_by_comp") is None
    assert meta.get("fused_W_by_comp") is not None
    assert meta.get("edge_bucket_fused_pos") is not None

    fused_by_comp = meta["fused_W_by_comp"]
    edge_bucket_comp = meta["edge_bucket_comp"]
    edge_bucket_fused_pos = meta["edge_bucket_fused_pos"]
    edge_scale = meta.get("edge_scale")
    assert isinstance(fused_by_comp, dict)
    assert edge_bucket_comp is not None
    assert edge_bucket_fused_pos is not None

    prev_values = {
        comp: mat.values().clone() for comp, mat in fused_by_comp.items() if mat is not None
    }

    model = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0))
    active_edges = torch.tensor([0, 2], dtype=torch.long)
    d_weights = torch.tensor([0.2, -0.1], dtype=torch.float32)
    model.apply_weight_updates(compiled, active_edges, d_weights)

    comp_order = tuple(Compartment)
    scale_vals = None
    if edge_scale is not None:
        scale_vals = edge_scale.index_select(0, active_edges)

    for idx, edge in enumerate(active_edges.tolist()):
        comp_id = int(edge_bucket_comp[edge])
        comp = comp_order[comp_id]
        pos = int(edge_bucket_fused_pos[edge])
        expected = prev_values[comp][pos] + d_weights[idx] * (scale_vals[idx] if scale_vals is not None else 1.0)
        torch.testing.assert_close(fused_by_comp[comp].values()[pos], expected)

    sparse_state = model.init_state(pre_idx.numel(), ctx=ctx)
    ring_model = DelayedCurrentSynapse()
    ring_state = ring_model.init_state(pre_idx.numel(), ctx=ctx)
    assert compiled.weights is not None
    ring_state.weights.copy_(compiled.weights)

    pre_spikes = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    sparse_state, sparse_out = model.step(
        sparse_state,
        compiled,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )
    ring_state, ring_out = ring_model.step(
        ring_state,
        compiled,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert sparse_out.post_drive.keys() == ring_out.post_drive.keys()
    for comp in sparse_out.post_drive:
        torch.testing.assert_close(sparse_out.post_drive[comp], ring_out.post_drive[comp])
