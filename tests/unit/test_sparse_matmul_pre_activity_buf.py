from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_sparse_matmul import DelayedSparseMatmulSynapse

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _build_topology() -> SynapseTopology:
    pre_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 0, 1], dtype=torch.long)
    delay_steps = torch.zeros_like(pre_idx)
    weights = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
        weights=weights,
    )
    return topology


def test_pre_activity_buf_reused() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    topology = _build_topology()
    topology = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
    )

    model = DelayedSparseMatmulSynapse()
    state = model.init_state(topology.pre_idx.numel(), ctx=ctx)
    pre_spikes = torch.tensor([True, False, True], dtype=torch.bool)

    buf_id = None
    for step in range(20):
        state, _ = model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=pre_spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        assert state.pre_activity_buf is not None
        if buf_id is None:
            buf_id = id(state.pre_activity_buf)
        else:
            assert id(state.pre_activity_buf) == buf_id


def test_pre_activity_buf_correctness() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    topology = _build_topology()
    topology = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
    )

    model = DelayedSparseMatmulSynapse()
    state = model.init_state(topology.pre_idx.numel(), ctx=ctx)
    pre_spikes = torch.tensor([True, False, True], dtype=torch.bool)

    state, result = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    meta = topology.meta or {}
    mats_by_comp = meta.get("W_by_delay_by_comp_csr") or meta["W_by_delay_by_comp"]
    mats = mats_by_comp[Compartment.SOMA]
    mat = mats[0]
    assert mat is not None
    expected = torch.sparse.mm(mat, pre_spikes.to(dtype=state.weights.dtype).unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(result.post_drive[Compartment.SOMA], expected)
