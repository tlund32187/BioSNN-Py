from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _base_topology() -> SynapseTopology:
    pre_idx = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.long)
    delay_steps = torch.tensor([0, 2, 1, 0, 3, 1], dtype=torch.int32)
    weights = torch.linspace(0.2, 1.1, steps=pre_idx.numel(), dtype=torch.float32)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.DENDRITE,
        weights=weights,
    )


def test_fused_routing_indices_match_delays() -> None:
    topology = _base_topology()
    compiled = compile_topology(
        topology,
        device="cpu",
        dtype="float32",
        build_sparse_delay_mats=True,
        fuse_delay_buckets=True,
    )
    meta = compiled.meta or {}
    delays_by_comp = meta.get("fused_W_delays_by_comp")
    immediate_idx_by_comp = meta.get("fused_immediate_blocks_idx_by_comp")
    delayed_idx_by_comp = meta.get("fused_delayed_blocks_idx_by_comp")
    is_immediate_by_comp = meta.get("fused_is_immediate_by_comp")

    assert isinstance(delays_by_comp, dict)
    assert isinstance(immediate_idx_by_comp, dict)
    assert isinstance(delayed_idx_by_comp, dict)
    assert isinstance(is_immediate_by_comp, dict)

    for comp, delays in delays_by_comp.items():
        mask = delays == 0
        torch.testing.assert_close(is_immediate_by_comp[comp], mask)
        expected_immediate = mask.nonzero(as_tuple=False).flatten()
        expected_delayed = (~mask).nonzero(as_tuple=False).flatten()
        torch.testing.assert_close(immediate_idx_by_comp[comp], expected_immediate)
        torch.testing.assert_close(delayed_idx_by_comp[comp], expected_delayed)


def test_fused_step_matches_unfused() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    topology = _base_topology()
    fused = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
        fuse_delay_buckets=True,
    )
    unfused = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
        fuse_delay_buckets=False,
    )
    torch.manual_seed(0)
    pre_seq = [(torch.rand((3,)) > 0.4).to(dtype=torch.float32) for _ in range(4)]

    model_fused = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0))
    model_unfused = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0))
    state_fused = model_fused.init_state(fused.pre_idx.numel(), ctx=ctx)
    state_unfused = model_unfused.init_state(unfused.pre_idx.numel(), ctx=ctx)
    assert fused.weights is not None
    assert unfused.weights is not None
    state_fused.weights.copy_(fused.weights)
    state_unfused.weights.copy_(unfused.weights)

    for step, spikes in enumerate(pre_seq):
        state_fused, out_fused = model_fused.step(
            state_fused,
            fused,
            SynapseInputs(pre_spikes=spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        state_unfused, out_unfused = model_unfused.step(
            state_unfused,
            unfused,
            SynapseInputs(pre_spikes=spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        assert out_fused.post_drive.keys() == out_unfused.post_drive.keys()
        for comp in out_fused.post_drive:
            torch.testing.assert_close(out_fused.post_drive[comp], out_unfused.post_drive[comp])
