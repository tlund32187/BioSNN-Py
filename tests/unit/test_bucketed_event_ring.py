from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.synapses.buffers.bucketed_event_ring import BucketedEventRing
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _arrival_steps(traces: list[Tensor]) -> list[int]:
    steps: list[int] = []
    for idx, value in enumerate(traces):
        if torch.any(value != 0):
            steps.append(idx)
    return steps


def _run_bucketed_delay_alignment(
    device: str,
    *,
    depth: int,
    delay_steps: int,
    spike_step: int,
    steps: int,
) -> list[Tensor]:
    ring = BucketedEventRing(depth=depth, device=device, dtype=torch.float32, max_events=128)
    cursor = 0
    traces: list[Tensor] = []
    for step in range(steps):
        out = torch.zeros((1,), device=device, dtype=torch.float32)
        ring.pop_into(cursor, out)
        traces.append(out.detach().cpu().clone())
        if step == spike_step:
            due = torch.remainder(
                torch.tensor([cursor + delay_steps], device=device, dtype=torch.long),
                depth,
            )
            posts = torch.tensor([0], device=device, dtype=torch.long)
            vals = torch.tensor([1.0], device=device, dtype=torch.float32)
            ring.schedule(due, posts, vals)
        cursor = (cursor + 1) % depth
    return traces


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


def _run_delayed_current_bucketed(
    device: str,
    *,
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
            ring_strategy="event_bucketed",
            event_list_max_events=1024,
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


def test_bucketed_event_ring_schedule_pop_cpu() -> None:
    ring = BucketedEventRing(depth=5, device="cpu", dtype=torch.float32, max_events=16)
    due = torch.tensor([1, 3, 1, 4], dtype=torch.long)
    posts = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    vals = torch.tensor([0.5, 1.5, 2.0, 3.0], dtype=torch.float32)
    ring.schedule(due, posts, vals)

    assert ring.active_count == 4
    torch.testing.assert_close(ring.slot_counts, torch.tensor([0, 2, 0, 1, 1], dtype=torch.long))

    out = torch.zeros((3,), dtype=torch.float32)
    ring.pop_into(1, out)
    torch.testing.assert_close(out, torch.tensor([0.5, 0.0, 2.0], dtype=torch.float32))
    assert ring.active_count == 2

    out.zero_()
    ring.pop_into(3, out)
    torch.testing.assert_close(out, torch.tensor([0.0, 1.5, 0.0], dtype=torch.float32))
    assert ring.active_count == 1

    out.zero_()
    ring.pop_into(4, out)
    torch.testing.assert_close(out, torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32))
    assert ring.active_count == 0


def test_bucketed_event_ring_delay_alignment_cpu() -> None:
    delay_steps = 3
    spike_step = 5
    traces = _run_bucketed_delay_alignment(
        "cpu",
        depth=4,
        delay_steps=delay_steps,
        spike_step=spike_step,
        steps=12,
    )
    assert _arrival_steps(traces) == [spike_step + delay_steps]


def test_bucketed_event_ring_capacity_guard() -> None:
    ring = BucketedEventRing(depth=4, device="cpu", dtype=torch.float32, max_events=2)
    ring.schedule(
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    with pytest.raises(RuntimeError):
        ring.schedule(
            torch.tensor([3], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            torch.tensor([3.0], dtype=torch.float32),
        )


def test_delayed_current_event_bucketed_cpu_alignment() -> None:
    delay_steps = 3
    spike_step = 5
    traces = _run_delayed_current_bucketed(
        "cpu",
        delay_steps=delay_steps,
        spike_step=spike_step,
        steps=12,
    )
    assert _arrival_steps(traces) == [spike_step + delay_steps]


@pytest.mark.cuda
def test_bucketed_event_ring_cuda_alignment() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    delay_steps = 3
    spike_step = 5
    traces = _run_bucketed_delay_alignment(
        "cuda",
        depth=4,
        delay_steps=delay_steps,
        spike_step=spike_step,
        steps=12,
    )
    assert _arrival_steps(traces) == [spike_step + delay_steps]
