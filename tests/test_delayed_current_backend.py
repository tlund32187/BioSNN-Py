from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_current import (
    DelayedCurrentParams,
    DelayedCurrentSynapse,
    _gather_active_edges,
    _require_pre_adjacency,
)

torch = pytest.importorskip("torch")


def _run_synapse(model, topology, pre_seq, *, ctx):
    state = model.init_state(topology.pre_idx.numel(), ctx=ctx)
    state.weights.copy_(torch.linspace(0.1, 1.0, steps=state.weights.numel(), dtype=state.weights.dtype))
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
        outputs.append(result.post_drive[Compartment.SOMA].clone())
    return outputs


def test_delayed_current_ring_matches_edge_buffer():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 1, 2, 2, 0], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
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
        build_edges_by_delay=True,
    )

    torch.manual_seed(0)
    pre_seq = [(torch.rand((4,)) > 0.5).to(dtype=torch.float32) for _ in range(6)]

    legacy = DelayedCurrentSynapse(DelayedCurrentParams(use_edge_buffer=True, init_weight=1.0))
    ring = DelayedCurrentSynapse(DelayedCurrentParams(use_edge_buffer=False, init_weight=1.0))

    legacy_out = _run_synapse(legacy, topology, pre_seq, ctx=ctx)
    ring_out = _run_synapse(ring, topology, pre_seq, ctx=ctx)

    assert len(legacy_out) == len(ring_out)
    for a, b in zip(legacy_out, ring_out, strict=True):
        torch.testing.assert_close(a, b)


def test_delayed_current_ring_memory_shape():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 0], dtype=torch.long)
    delay_steps = torch.tensor([1, 2, 0], dtype=torch.int32)
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
        build_edges_by_delay=True,
    )

    model = DelayedCurrentSynapse()
    state = model.init_state(pre_idx.numel(), ctx=ctx)
    pre_spikes = torch.zeros((3,), dtype=state.weights.dtype)

    state, _ = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert state.post_ring is not None
    assert topology.meta is not None
    depth = int(topology.meta["max_delay_steps"]) + 1
    ring = state.post_ring[Compartment.SOMA]
    assert ring.shape == (depth, 2)


def test_delayed_current_edge_buffer_reuses_delay_tmp():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 0], dtype=torch.long)
    delay_steps = torch.tensor([1, 2, 0], dtype=torch.int32)
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
    )

    model = DelayedCurrentSynapse(DelayedCurrentParams(use_edge_buffer=True, init_weight=1.0))
    state = model.init_state(pre_idx.numel(), ctx=ctx)
    pre_spikes = torch.zeros((3,), dtype=state.weights.dtype)

    state, _ = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )
    assert state.delay_tmp is not None
    tmp_id = id(state.delay_tmp)

    for step in range(3):
        state, _ = model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=pre_spikes),
            dt=1e-3,
            t=(step + 1) * 1e-3,
            ctx=ctx,
        )
    assert state.delay_tmp is not None
    assert id(state.delay_tmp) == tmp_id


def test_delayed_current_event_driven_matches_dense():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 0, 1, 3], dtype=torch.long)
    post_idx = torch.tensor([0, 0, 1, 2, 2, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 2, 0, 2, 1], dtype=torch.int32)
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
        build_edges_by_delay=True,
        build_pre_adjacency=True,
    )

    torch.manual_seed(1)
    pre_seq = [(torch.rand((4,)) > 0.7).to(dtype=torch.float32) for _ in range(5)]

    dense = DelayedCurrentSynapse(DelayedCurrentParams(event_driven=False, init_weight=1.0))
    event = DelayedCurrentSynapse(DelayedCurrentParams(event_driven=True, init_weight=1.0))

    dense_out = _run_synapse(dense, topology, pre_seq, ctx=ctx)
    event_out = _run_synapse(event, topology, pre_seq, ctx=ctx)

    for a, b in zip(dense_out, event_out, strict=True):
        torch.testing.assert_close(a, b)


def test_delayed_current_event_driven_processes_fewer_edges():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 1, 2, 2, 0], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
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
        build_pre_adjacency=True,
    )

    model = DelayedCurrentSynapse(DelayedCurrentParams(event_driven=True, init_weight=1.0))
    state = model.init_state(pre_idx.numel(), ctx=ctx)
    state.weights.fill_(1.0)

    pre_spikes = torch.zeros((4,), dtype=state.weights.dtype)
    pre_spikes[0] = 1.0

    state, result = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert result.extras is not None
    processed = int(result.extras["processed_edges"].item())
    assert processed < pre_idx.numel()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_delayed_current_event_driven_gather_edges(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pre_idx = torch.tensor([2, 0, 1, 2, 3, 0], dtype=torch.long, device=device)
    post_idx = torch.tensor([0, 0, 1, 2, 1, 2], dtype=torch.long, device=device)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        target_compartment=Compartment.SOMA,
    )
    topology = compile_topology(
        topology,
        device=device,
        dtype="float32",
        build_pre_adjacency=True,
    )

    weights = torch.tensor([0.5, 1.0, 0.2, 0.8, 1.5, 0.3], dtype=torch.float32, device=device)
    pre_spikes = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32, device=device)

    active_pre = pre_spikes.nonzero(as_tuple=False).flatten()
    pre_ptr, edge_idx = _require_pre_adjacency(topology, pre_idx.device)
    edges = _gather_active_edges(active_pre, pre_ptr, edge_idx, topology)

    pre_ptr_cpu = pre_ptr.detach().cpu().tolist()
    edge_idx_cpu = edge_idx.detach().cpu().tolist()
    active_cpu = active_pre.detach().cpu().tolist()
    expected_edges: list[int] = []
    for idx in active_cpu:
        start = int(pre_ptr_cpu[idx])
        end = int(pre_ptr_cpu[idx + 1])
        expected_edges.extend(edge_idx_cpu[start:end])
    assert edges.numel() == len(expected_edges)
    assert edges.detach().cpu().tolist() == expected_edges

    expected_out = torch.zeros((3,), dtype=torch.float32)
    pre_idx_cpu = pre_idx.detach().cpu().tolist()
    post_idx_cpu = post_idx.detach().cpu().tolist()
    pre_spikes_cpu = pre_spikes.detach().cpu().tolist()
    weights_cpu = weights.detach().cpu().tolist()
    for edge in expected_edges:
        pre = pre_idx_cpu[edge]
        post = post_idx_cpu[edge]
        expected_out[post] += pre_spikes_cpu[pre] * weights_cpu[edge]

    edge_pre = pre_idx.index_select(0, edges)
    edge_post = post_idx.index_select(0, edges)
    edge_spikes = pre_spikes.index_select(0, edge_pre)
    edge_weights = weights.index_select(0, edges)
    edge_current = edge_spikes * edge_weights
    actual_out = torch.zeros((3,), device=device, dtype=edge_current.dtype)
    actual_out.index_add_(0, edge_post, edge_current)

    torch.testing.assert_close(actual_out.detach().cpu(), expected_out)


def test_delayed_current_adaptive_uses_event_driven_for_sparse(monkeypatch):
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 1, 2, 2, 0], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
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
        build_edges_by_delay=True,
        build_pre_adjacency=True,
    )

    model = DelayedCurrentSynapse(
        DelayedCurrentParams(adaptive_event_driven=True, adaptive_threshold=0.5, init_weight=1.0)
    )
    state = model.init_state(pre_idx.numel(), ctx=ctx)
    state.weights.fill_(1.0)

    called = {"event": 0, "dense": 0}
    import biosnn.synapses.dynamics.delayed_current as dc

    orig_event = dc._step_post_ring_event_driven
    orig_dense = dc._step_post_ring

    def wrap_event(*args, **kwargs):
        called["event"] += 1
        return orig_event(*args, **kwargs)

    def wrap_dense(*args, **kwargs):
        called["dense"] += 1
        return orig_dense(*args, **kwargs)

    monkeypatch.setattr(dc, "_step_post_ring_event_driven", wrap_event)
    monkeypatch.setattr(dc, "_step_post_ring", wrap_dense)

    pre_spikes = torch.zeros((4,), dtype=state.weights.dtype)
    pre_spikes[0] = 1.0

    model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert called["event"] == 1
    assert called["dense"] == 0


def test_delayed_current_adaptive_uses_dense_for_dense(monkeypatch):
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 1, 2, 2, 0], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
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
        build_edges_by_delay=True,
        build_pre_adjacency=True,
    )

    model = DelayedCurrentSynapse(
        DelayedCurrentParams(adaptive_event_driven=True, adaptive_threshold=0.1, init_weight=1.0)
    )
    state = model.init_state(pre_idx.numel(), ctx=ctx)
    state.weights.fill_(1.0)

    called = {"event": 0, "dense": 0}
    import biosnn.synapses.dynamics.delayed_current as dc

    orig_event = dc._step_post_ring_event_driven
    orig_dense = dc._step_post_ring

    def wrap_event(*args, **kwargs):
        called["event"] += 1
        return orig_event(*args, **kwargs)

    def wrap_dense(*args, **kwargs):
        called["dense"] += 1
        return orig_dense(*args, **kwargs)

    monkeypatch.setattr(dc, "_step_post_ring_event_driven", wrap_event)
    monkeypatch.setattr(dc, "_step_post_ring", wrap_dense)

    pre_spikes = torch.ones((4,), dtype=state.weights.dtype)

    model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert called["event"] == 0
    assert called["dense"] == 1


def test_delayed_current_adaptive_matches_event_driven_for_sparse():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 0, 1, 3], dtype=torch.long)
    post_idx = torch.tensor([0, 0, 1, 2, 2, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 2, 0, 2, 1], dtype=torch.int32)
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
        build_edges_by_delay=True,
        build_pre_adjacency=True,
    )

    pre_seq = [
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
    ]

    adaptive = DelayedCurrentSynapse(
        DelayedCurrentParams(adaptive_event_driven=True, adaptive_threshold=0.5, init_weight=1.0)
    )
    event = DelayedCurrentSynapse(DelayedCurrentParams(event_driven=True, init_weight=1.0))

    adaptive_out = _run_synapse(adaptive, topology, pre_seq, ctx=ctx)
    event_out = _run_synapse(event, topology, pre_seq, ctx=ctx)

    for a, b in zip(adaptive_out, event_out, strict=True):
        torch.testing.assert_close(a, b)


def test_delayed_current_adaptive_matches_dense_for_dense():
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 0, 1, 3], dtype=torch.long)
    post_idx = torch.tensor([0, 0, 1, 2, 2, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 2, 0, 2, 1], dtype=torch.int32)
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
        build_edges_by_delay=True,
        build_pre_adjacency=True,
    )

    pre_seq = [
        torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    ]

    adaptive = DelayedCurrentSynapse(
        DelayedCurrentParams(adaptive_event_driven=True, adaptive_threshold=0.1, init_weight=1.0)
    )
    dense = DelayedCurrentSynapse(DelayedCurrentParams(event_driven=False, init_weight=1.0))

    adaptive_out = _run_synapse(adaptive, topology, pre_seq, ctx=ctx)
    dense_out = _run_synapse(dense, topology, pre_seq, ctx=ctx)

    for a, b in zip(adaptive_out, dense_out, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_delayed_current_ring_cuda_device():
    ctx = StepContext(device="cuda", dtype="float32")
    pre_idx = torch.tensor([0, 1], dtype=torch.long, device="cuda")
    post_idx = torch.tensor([0, 1], dtype=torch.long, device="cuda")
    delay_steps = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
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
        build_edges_by_delay=True,
    )

    model = DelayedCurrentSynapse()
    state = model.init_state(pre_idx.numel(), ctx=ctx)
    pre_spikes = torch.zeros((2,), device="cuda", dtype=state.weights.dtype)

    state, _ = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert state.post_ring is not None
    assert state.post_ring[Compartment.SOMA].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_delayed_current_event_driven_cuda_no_item(monkeypatch):
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
        build_pre_adjacency=True,
    )

    model = DelayedCurrentSynapse(DelayedCurrentParams(event_driven=True, init_weight=1.0))
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
