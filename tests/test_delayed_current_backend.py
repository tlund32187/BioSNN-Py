from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

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
    compile_topology(topology, device=ctx.device, dtype=ctx.dtype)

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
    compile_topology(topology, device=ctx.device, dtype=ctx.dtype)

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
    compile_topology(topology, device=ctx.device, dtype=ctx.dtype)

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
    compile_topology(topology, device=ctx.device, dtype=ctx.dtype)

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
    compile_topology(topology, device=ctx.device, dtype=ctx.dtype)

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
