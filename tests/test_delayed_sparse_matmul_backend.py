from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import ReceptorKind, SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

torch = pytest.importorskip("torch")


def _run(model, topology, pre_seq, *, ctx):
    state = model.init_state(topology.pre_idx.numel(), ctx=ctx)
    state.weights.copy_(topology.weights)
    outputs = []
    for step, spikes in enumerate(pre_seq):
        state, result = model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        outputs.append({comp: drive.clone() for comp, drive in result.post_drive.items()})
    return outputs


def test_sparse_matmul_matches_ring_backend():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 1, 2, 2, 0], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
    weights = torch.linspace(0.1, 1.0, steps=pre_idx.numel(), dtype=torch.float32)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
        weights=weights,
    )
    compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
        build_sparse_delay_mats=True,
    )

    torch.manual_seed(0)
    pre_seq = [(torch.rand((4,)) > 0.5).to(dtype=torch.float32) for _ in range(6)]

    ring = DelayedCurrentSynapse(DelayedCurrentParams(use_edge_buffer=False, init_weight=1.0))
    sparse = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0))

    ring_out = _run(ring, topology, pre_seq, ctx=ctx)
    sparse_out = _run(sparse, topology, pre_seq, ctx=ctx)

    for a, b in zip(ring_out, sparse_out, strict=True):
        assert a.keys() == b.keys()
        for comp in a:
            torch.testing.assert_close(a[comp], b[comp])


def test_sparse_matmul_multi_comp_receptor_matches_ring():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 1, 2, 2, 0, 2, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.int32)
    weights = torch.linspace(0.2, 0.9, steps=pre_idx.numel(), dtype=torch.float32)
    comp_order = list(Compartment)
    target_compartments = torch.tensor(
        [comp_order.index(Compartment.SOMA), comp_order.index(Compartment.DENDRITE)] * 4,
        dtype=torch.long,
    )
    receptor_ids = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)
    receptor_kinds = (ReceptorKind.AMPA, ReceptorKind.NMDA)
    receptor_scale = {ReceptorKind.AMPA: 1.0, ReceptorKind.NMDA: 0.5}
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
        target_compartments=target_compartments,
        receptor=receptor_ids,
        receptor_kinds=receptor_kinds,
        weights=weights,
        meta={"receptor_scale": receptor_scale},
    )
    compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
        build_sparse_delay_mats=True,
    )

    torch.manual_seed(1)
    pre_seq = [(torch.rand((4,)) > 0.6).to(dtype=torch.float32) for _ in range(5)]

    ring = DelayedCurrentSynapse(
        DelayedCurrentParams(use_edge_buffer=False, init_weight=1.0, receptor_scale=receptor_scale)
    )
    sparse = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(init_weight=1.0, receptor_scale=receptor_scale)
    )

    ring_out = _run(ring, topology, pre_seq, ctx=ctx)
    sparse_out = _run(sparse, topology, pre_seq, ctx=ctx)

    for a, b in zip(ring_out, sparse_out, strict=True):
        assert a.keys() == b.keys()
        for comp in a:
            torch.testing.assert_close(a[comp], b[comp])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_matmul_cuda_no_item_hot_path(monkeypatch):
    ctx = StepContext(device="cuda", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2], dtype=torch.long, device="cuda")
    post_idx = torch.tensor([0, 1, 0], dtype=torch.long, device="cuda")
    delay_steps = torch.tensor([1, 0, 2], dtype=torch.int32, device="cuda")
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
    )
    compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
    )

    model = DelayedSparseMatmulSynapse()
    state = model.init_state(pre_idx.numel(), ctx=ctx)
    pre_spikes = torch.zeros((3,), device="cuda", dtype=state.weights.dtype)

    def _no_item(self):
        raise RuntimeError(".item() in hot path")

    monkeypatch.setattr(torch.Tensor, "item", _no_item, raising=False)
    model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )
