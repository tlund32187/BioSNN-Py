from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit


from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics import DelayedCurrentParams, DelayedCurrentSynapse

torch = pytest.importorskip("torch")


def _ctx(device: str = "cpu", dtype: str = "float32") -> StepContext:
    return StepContext(device=device, dtype=dtype)


def _topology(pre_idx, post_idx, **kwargs):
    return SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, **kwargs)


def test_post_out_reused_across_steps():
    model = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1.0))
    ctx = _ctx()
    state = model.init_state(1, ctx=ctx)
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    delay_steps = torch.tensor([1], dtype=torch.int32)
    topology = _topology(
        pre_idx,
        post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
    )
    topology = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
    )

    pre_spikes = torch.tensor([1.0], dtype=state.weights.dtype)
    state, _ = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )
    assert state.post_out is not None
    base_id = id(state.post_out[Compartment.SOMA])

    for step in range(1, 4):
        pre_spikes = torch.tensor([0.0], dtype=state.weights.dtype)
        state, _ = model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=pre_spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        assert state.post_out is not None
        assert id(state.post_out[Compartment.SOMA]) == base_id


def test_post_out_matches_ring_slot_value():
    model = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1.0))
    ctx = _ctx()
    state = model.init_state(1, ctx=ctx)
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    delay_steps = torch.tensor([1], dtype=torch.int32)
    topology = _topology(
        pre_idx,
        post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
    )
    topology = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
    )

    pre_spikes = torch.tensor([0.0], dtype=state.weights.dtype)
    state, _ = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert state.post_ring is not None
    cursor = state.cursor % 2
    ring = state.post_ring[Compartment.SOMA]
    ring[cursor].fill_(2.5)

    state, result = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=1e-3,
        ctx=ctx,
    )

    out = result.post_drive[Compartment.SOMA]
    assert out.item() == pytest.approx(2.5)
    assert ring[cursor].sum().item() == pytest.approx(0.0)
