from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
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


def _step_outputs(topology: SynapseTopology, *, fused: bool) -> list[Tensor]:
    ctx = StepContext(device="cpu", dtype="float32")
    compiled = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
        fuse_delay_buckets=fused,
    )
    synapse = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0))
    state = synapse.init_state(compiled.pre_idx.numel(), ctx=ctx)
    if compiled.weights is not None:
        state.weights.copy_(compiled.weights)
    outputs: list[Tensor] = []
    torch.manual_seed(0)
    pre_seq = [(torch.rand((3,)) > 0.4).to(dtype=torch.float32) for _ in range(4)]
    for step, spikes in enumerate(pre_seq):
        state, out = synapse.step(
            state,
            compiled,
            SynapseInputs(pre_spikes=spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        outputs.append(out.post_drive[Compartment.DENDRITE].detach().cpu().clone())
    return outputs


def test_fused_routing_metadata_present() -> None:
    topology = _base_topology()
    compiled = compile_topology(
        topology,
        device="cpu",
        dtype="float32",
        build_sparse_delay_mats=True,
        fuse_delay_buckets=True,
    )
    meta = compiled.meta or {}
    delays_long = meta.get("fused_W_delays_long_by_comp")
    d_by_comp = meta.get("fused_D_by_comp")
    assert isinstance(delays_long, dict)
    assert isinstance(d_by_comp, dict)
    for comp, delays in delays_long.items():
        assert delays.dtype == torch.long
        if hasattr(delays, "device"):
            assert delays.device.type == "cpu"
        d_val = d_by_comp.get(comp)
        assert d_val == int(delays.numel())


def test_fused_step_matches_unfused_with_metadata() -> None:
    topology = _base_topology()
    fused = _step_outputs(topology, fused=True)
    unfused = _step_outputs(topology, fused=False)
    assert len(fused) == len(unfused)
    for a, b in zip(fused, unfused, strict=True):
        torch.testing.assert_close(a, b)
