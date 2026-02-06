from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.synapses.buffers.event_list_ring import EventListRing
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


def _run_event_list(device: str, *, delay_steps: int, spike_step: int, steps: int) -> list[Tensor]:
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
            ring_strategy="event_list_proto",
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


def _arrival_steps(traces: list[Tensor]) -> list[int]:
    steps: list[int] = []
    for idx, value in enumerate(traces):
        if torch.any(value != 0):
            steps.append(idx)
    return steps


def test_event_list_ring_cpu_alignment() -> None:
    delay_steps = 3
    spike_step = 5
    steps = 12
    traces = _run_event_list("cpu", delay_steps=delay_steps, spike_step=spike_step, steps=steps)
    expected = spike_step + delay_steps
    assert _arrival_steps(traces) == [expected]


@pytest.mark.cuda
def test_event_list_ring_cuda_alignment() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    delay_steps = 3
    spike_step = 5
    steps = 12
    traces = _run_event_list("cuda", delay_steps=delay_steps, spike_step=spike_step, steps=steps)
    expected = spike_step + delay_steps
    assert _arrival_steps(traces) == [expected]


def test_event_list_ring_capacity_guard() -> None:
    ring = EventListRing(depth=4, device="cpu", dtype=torch.float32, max_events=2)
    due = torch.tensor([1, 2], dtype=torch.long)
    posts = torch.tensor([0, 1], dtype=torch.long)
    vals = torch.tensor([1.0, 2.0], dtype=torch.float32)
    ring.schedule(due, posts, vals)
    with pytest.raises(RuntimeError):
        ring.schedule(torch.tensor([3], dtype=torch.long), torch.tensor([2]), torch.tensor([3.0]))
