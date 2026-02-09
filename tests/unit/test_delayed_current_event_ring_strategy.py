from __future__ import annotations

from typing import Literal

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.synapses.buffers.bucketed_event_ring import BucketedEventRing
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _build_topology(delay_steps: int) -> SynapseTopology:
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    delays = torch.tensor([delay_steps], dtype=torch.int32)
    weights = torch.tensor([1.0], dtype=torch.float32)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delays,
        weights=weights,
        target_compartment=Compartment.DENDRITE,
    )


def _run_strategy(
    *,
    device: str,
    ring_strategy: Literal["dense", "event_bucketed"],
    delay_steps: int,
    spike_step: int,
    steps: int,
) -> list[Tensor]:
    topology = _build_topology(delay_steps)
    compiled = compile_topology(
        topology,
        device=device,
        dtype="float32",
        build_pre_adjacency=True,
    )
    synapse = DelayedCurrentSynapse(
        DelayedCurrentParams(
            init_weight=1.0,
            event_driven=True,
            ring_strategy=ring_strategy,
            ring_capacity_max=2048,
            event_list_max_events=2048,
        )
    )
    ctx = StepContext(device=device, dtype="float32")
    state = synapse.init_state(compiled.pre_idx.numel(), ctx=ctx)
    if compiled.weights is not None:
        state.weights.copy_(compiled.weights)

    traces: list[Tensor] = []
    for step in range(steps):
        spikes = torch.zeros((1,), device=device, dtype=state.weights.dtype)
        if step == spike_step:
            spikes[0] = 1.0
        state, res = synapse.step(
            state,
            compiled,
            SynapseInputs(pre_spikes=spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        traces.append(res.post_drive[Compartment.DENDRITE].detach().cpu().clone())
    return traces


def _first_nonzero_step(traces: list[Tensor]) -> int | None:
    for idx, tensor in enumerate(traces):
        if torch.any(tensor != 0):
            return idx
    return None


def test_delay_impulse_matches_dense_and_bucketed_cpu() -> None:
    delay_steps = 3
    spike_step = 4
    steps = 12
    dense = _run_strategy(
        device="cpu",
        ring_strategy="dense",
        delay_steps=delay_steps,
        spike_step=spike_step,
        steps=steps,
    )
    bucketed = _run_strategy(
        device="cpu",
        ring_strategy="event_bucketed",
        delay_steps=delay_steps,
        spike_step=spike_step,
        steps=steps,
    )

    assert _first_nonzero_step(dense) == spike_step + delay_steps
    assert _first_nonzero_step(bucketed) == spike_step + delay_steps
    for a, b in zip(dense, bucketed, strict=True):
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_immediate_events_bypass_bucketed_schedule() -> None:
    topology = _build_topology(delay_steps=0)
    compiled = compile_topology(
        topology,
        device="cpu",
        dtype="float32",
        build_pre_adjacency=True,
    )
    synapse = DelayedCurrentSynapse(
        DelayedCurrentParams(
            init_weight=1.0,
            event_driven=True,
            ring_strategy="event_bucketed",
            ring_capacity_max=32,
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    state = synapse.init_state(compiled.pre_idx.numel(), ctx=ctx)
    if compiled.weights is not None:
        state.weights.copy_(compiled.weights)

    state, res = synapse.step(
        state,
        compiled,
        SynapseInputs(pre_spikes=torch.tensor([1.0], dtype=state.weights.dtype)),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )
    torch.testing.assert_close(res.post_drive[Compartment.DENDRITE], torch.tensor([1.0]))

    assert state.post_event_ring is not None
    ring = state.post_event_ring[Compartment.DENDRITE]
    assert isinstance(ring, BucketedEventRing)
    assert ring.active_count == 0


@pytest.mark.cuda
def test_delay_impulse_matches_dense_and_bucketed_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    delay_steps = 3
    spike_step = 4
    steps = 12
    dense = _run_strategy(
        device="cuda",
        ring_strategy="dense",
        delay_steps=delay_steps,
        spike_step=spike_step,
        steps=steps,
    )
    bucketed = _run_strategy(
        device="cuda",
        ring_strategy="event_bucketed",
        delay_steps=delay_steps,
        spike_step=spike_step,
        steps=steps,
    )

    assert _first_nonzero_step(dense) == spike_step + delay_steps
    assert _first_nonzero_step(bucketed) == spike_step + delay_steps
    for a, b in zip(dense, bucketed, strict=True):
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
