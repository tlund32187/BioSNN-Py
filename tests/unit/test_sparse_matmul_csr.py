from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseTopology

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _csr_supported() -> bool:
    return hasattr(torch.Tensor, "to_sparse_csr")


@pytest.mark.skipif(not _csr_supported(), reason="CSR sparse tensors not supported")
def test_sparse_mats_are_csr() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1], dtype=torch.long)
    post_idx = torch.tensor([0, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 1], dtype=torch.int32)
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
        build_sparse_delay_mats=True,
    )

    meta = topology.meta or {}
    mats_by_comp = meta["W_by_delay_by_comp_csr"]
    mat = mats_by_comp[Compartment.SOMA][0]
    assert mat is not None
    assert mat.is_sparse_csr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_csr_matches_coo(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not _csr_supported():
        pytest.skip("CSR sparse tensors not supported")

    ctx = StepContext(device=device, dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=device)
    post_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    delay_steps = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
        meta={"keep_sparse_coo": True},
    )
    topology = compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_sparse_delay_mats=True,
    )

    meta = topology.meta or {}
    mats_coo = meta["W_by_delay_by_comp"][Compartment.SOMA]
    mats_csr = meta["W_by_delay_by_comp_csr"][Compartment.SOMA]
    pre_activity = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=torch.float32)

    for delay in range(len(mats_coo)):
        mat_coo = mats_coo[delay]
        mat_csr = mats_csr[delay]
        if mat_coo is None and mat_csr is None:
            continue
        assert mat_coo is not None and mat_csr is not None
        out_coo = torch.sparse.mm(mat_coo, pre_activity.unsqueeze(1)).squeeze(1)
        out_csr = torch.sparse.mm(mat_csr, pre_activity.unsqueeze(1)).squeeze(1)
        torch.testing.assert_close(out_coo, out_csr)
