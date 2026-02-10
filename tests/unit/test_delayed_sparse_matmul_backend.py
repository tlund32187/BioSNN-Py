from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import ReceptorKind, SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
    _nonempty_mats_by_comp,
)
from biosnn.synapses.receptors import profile_exc_ampa_nmda

pytestmark = pytest.mark.unit

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
    topology = compile_topology(
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
    topology = compile_topology(
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
    topology = compile_topology(
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_matmul_fused_matches_per_delay(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ctx = StepContext(device=device, dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long, device=device)
    post_idx = torch.tensor([0, 1, 1, 2, 2, 0], dtype=torch.long, device=device)
    delay_steps = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device=device)
    weights = torch.linspace(0.1, 1.0, steps=pre_idx.numel(), dtype=torch.float32, device=device)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
        weights=weights,
    )
    topology = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
        build_bucket_edge_mapping=True,
    )

    model = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0))
    state = model.init_state(pre_idx.numel(), ctx=ctx)
    state.weights.copy_(weights)
    pre_spikes = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32, device=device)

    state, result = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    meta = topology.meta
    assert meta is not None
    depth = int(meta["max_delay_steps"]) + 1
    n_post = int(meta["n_post"])
    expected_drive = {Compartment.SOMA: torch.zeros((n_post,), device=device)}
    expected_ring = {Compartment.SOMA: torch.zeros((depth, n_post), device=device)}
    mats_by_comp = _nonempty_mats_by_comp(topology, pre_idx.device)
    for comp, mats in mats_by_comp.items():
        for delay, mat in mats:
            contrib = torch.sparse.mm(mat, pre_spikes.unsqueeze(1)).squeeze(1)
            if delay == 0:
                expected_drive[comp].add_(contrib)
            else:
                expected_ring[comp][delay].add_(contrib)

    torch.testing.assert_close(result.post_drive[Compartment.SOMA], expected_drive[Compartment.SOMA])
    assert state.post_ring is not None
    torch.testing.assert_close(state.post_ring[Compartment.SOMA], expected_ring[Compartment.SOMA])


def test_sparse_matmul_receptor_profile_dtype_defaults_cpu_to_weights() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    topology = SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long),
        post_idx=torch.tensor([0], dtype=torch.long),
        delay_steps=torch.tensor([0], dtype=torch.int32),
        target_compartment=Compartment.SOMA,
        weights=torch.tensor([0.2], dtype=torch.float32),
    )
    topology = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
    )
    model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            init_weight=1.0,
            receptor_profile=profile_exc_ampa_nmda(),
        )
    )
    state = model.init_state(1, ctx=ctx)
    state, _ = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=torch.tensor([1.0], dtype=torch.float32)),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert state.receptor_profile_dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_matmul_receptor_profile_dtype_defaults_cuda_to_fp16() -> None:
    ctx = StepContext(device="cuda", dtype="float32")
    topology = SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long, device="cuda"),
        post_idx=torch.tensor([0], dtype=torch.long, device="cuda"),
        delay_steps=torch.tensor([0], dtype=torch.int32, device="cuda"),
        target_compartment=Compartment.SOMA,
        weights=torch.tensor([0.2], dtype=torch.float32, device="cuda"),
    )
    topology = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
    )
    model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            init_weight=1.0,
            receptor_profile=profile_exc_ampa_nmda(),
        )
    )
    state = model.init_state(1, ctx=ctx)
    state, _ = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=torch.tensor([1.0], dtype=torch.float32, device="cuda")),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert state.receptor_profile_dtype == torch.float16
