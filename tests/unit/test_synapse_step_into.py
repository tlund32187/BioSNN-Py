from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics import DelayedCurrentParams, DelayedCurrentSynapse

torch = pytest.importorskip("torch")


def _ctx(device: str = "cpu", dtype: str = "float32") -> StepContext:
    return StepContext(device=device, dtype=dtype)


def test_step_into_matches_step_output():
    ctx = _ctx()
    model = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1.0))
    pre_idx = torch.tensor([0, 1, 0], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 0], dtype=torch.int32)
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

    state_a = model.init_state(pre_idx.numel(), ctx=ctx)
    state_b = model.init_state(pre_idx.numel(), ctx=ctx)
    state_b.weights.copy_(state_a.weights)

    pre_seq = [
        torch.tensor([1.0, 0.0], dtype=state_a.weights.dtype),
        torch.tensor([0.0, 1.0], dtype=state_a.weights.dtype),
        torch.tensor([0.0, 0.0], dtype=state_a.weights.dtype),
    ]

    for step, pre_spikes in enumerate(pre_seq):
        state_a, result = model.step(
            state_a,
            topology,
            SynapseInputs(pre_spikes=pre_spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )

        drive = {
            Compartment.SOMA: torch.zeros((2,), device=state_b.weights.device, dtype=state_b.weights.dtype)
        }
        model.step_into(
            state_b,
            pre_spikes,
            drive,
            step,
            topology=topology,
            dt=1e-3,
            ctx=ctx,
        )

        torch.testing.assert_close(drive[Compartment.SOMA], result.post_drive[Compartment.SOMA])
