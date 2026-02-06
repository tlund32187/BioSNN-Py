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


def _run(model: DelayedSparseMatmulSynapse, topology: SynapseTopology, pre_seq, *, ctx):
    state = model.init_state(topology.pre_idx.numel(), ctx=ctx)
    if topology.weights is not None:
        state.weights.copy_(topology.weights)
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
        outputs.append({comp: drive.clone() for comp, drive in result.post_drive.items()})
    return outputs


def test_fused_csr_matches_coo_cpu() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.long)
    delay_steps = torch.tensor([0, 2, 1, 0, 3, 1], dtype=torch.int32)
    weights = torch.linspace(0.2, 1.1, steps=pre_idx.numel(), dtype=torch.float32)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.DENDRITE,
        weights=weights,
    )
    topology = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
        fuse_delay_buckets=True,
        build_fused_csr=True,
    )

    meta = topology.meta or {}
    assert isinstance(meta.get("fused_W_by_comp_coo"), dict)
    assert isinstance(meta.get("fused_W_by_comp_csr"), dict)

    torch.manual_seed(0)
    pre_seq = [(torch.rand((3,)) > 0.4).to(dtype=torch.float32) for _ in range(5)]

    coo_model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(init_weight=1.0, fused_layout="coo")
    )
    csr_model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(init_weight=1.0, fused_layout="csr")
    )

    out_coo = _run(coo_model, topology, pre_seq, ctx=ctx)
    out_csr = _run(csr_model, topology, pre_seq, ctx=ctx)

    for a, b in zip(out_coo, out_csr, strict=True):
        assert a.keys() == b.keys()
        for comp in a:
            torch.testing.assert_close(a[comp], b[comp])
