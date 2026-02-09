from __future__ import annotations

from typing import Literal

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

pytestmark = pytest.mark.acceptance

torch = pytest.importorskip("torch")


def _run_backend(
    *,
    topology: SynapseTopology,
    backend: Literal["spmm_fused", "event_driven"],
    ring_strategy: Literal["dense", "event_bucketed"] = "dense",
    pre_seq: list[Tensor],
) -> list[Tensor]:
    model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            init_weight=1.0,
            backend=backend,
            ring_strategy=ring_strategy,
            ring_capacity_max=2048,
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(topology.pre_idx.numel(), ctx=ctx)
    if topology.weights is not None:
        state.weights.copy_(topology.weights)

    traces: list[Tensor] = []
    for step, spikes in enumerate(pre_seq):
        state, result = model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        traces.append(result.post_drive[Compartment.SOMA].detach().cpu().clone())
    return traces


def _first_nonzero_step(traces: list[Tensor]) -> int | None:
    for idx, tensor in enumerate(traces):
        if torch.any(tensor != 0):
            return idx
    return None


def test_event_driven_matches_spmm_fused_sparse_backend_cpu() -> None:
    topology = SynapseTopology(
        pre_idx=torch.tensor([0, 1, 2, 0, 1, 3, 3], dtype=torch.long),
        post_idx=torch.tensor([0, 0, 1, 2, 2, 1, 0], dtype=torch.long),
        delay_steps=torch.tensor([0, 1, 2, 0, 2, 1, 3], dtype=torch.int32),
        weights=torch.tensor([0.4, 0.2, 0.6, 0.3, 0.8, 0.5, 0.7], dtype=torch.float32),
        target_compartment=Compartment.SOMA,
    )
    compiled = compile_topology(
        topology,
        device="cpu",
        dtype="float32",
        build_sparse_delay_mats=True,
        build_pre_adjacency=True,
    )

    pre_seq = [
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    ]

    fused = _run_backend(
        topology=compiled,
        backend="spmm_fused",
        pre_seq=pre_seq,
    )
    event = _run_backend(
        topology=compiled,
        backend="event_driven",
        ring_strategy="dense",
        pre_seq=pre_seq,
    )

    for fused_t, event_t in zip(fused, event, strict=True):
        torch.testing.assert_close(event_t, fused_t, rtol=0.0, atol=1e-6)


def test_event_driven_bucketed_ring_preserves_delay_alignment_cpu() -> None:
    delay_steps = 4
    spike_step = 3
    topology = SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long),
        post_idx=torch.tensor([0], dtype=torch.long),
        delay_steps=torch.tensor([delay_steps], dtype=torch.int32),
        weights=torch.tensor([1.0], dtype=torch.float32),
        target_compartment=Compartment.SOMA,
    )
    compiled = compile_topology(
        topology,
        device="cpu",
        dtype="float32",
        build_sparse_delay_mats=True,
        build_pre_adjacency=True,
    )

    pre_seq: list[Tensor] = []
    for step in range(12):
        spikes = torch.zeros((1,), dtype=torch.float32)
        if step == spike_step:
            spikes[0] = 1.0
        pre_seq.append(spikes)

    fused = _run_backend(
        topology=compiled,
        backend="spmm_fused",
        pre_seq=pre_seq,
    )
    event_bucketed = _run_backend(
        topology=compiled,
        backend="event_driven",
        ring_strategy="event_bucketed",
        pre_seq=pre_seq,
    )

    expected_step = spike_step + delay_steps
    assert _first_nonzero_step(fused) == expected_step
    assert _first_nonzero_step(event_bucketed) == expected_step
    for fused_t, event_t in zip(fused, event_bucketed, strict=True):
        torch.testing.assert_close(event_t, fused_t, rtol=0.0, atol=1e-6)
