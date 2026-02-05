from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseTopology

torch = pytest.importorskip("torch")


def test_nonempty_mats_by_comp_delays() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    pre_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    post_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 2, 0, 2], dtype=torch.int32)
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
    nonempty = meta.get("nonempty_mats_by_comp_csr") or meta["nonempty_mats_by_comp"]
    nonempty = nonempty[Compartment.SOMA]
    delays = [delay for delay, _ in nonempty]
    assert delays == [0, 2]

    fused_by_comp = meta.get("fused_W_by_comp")
    fused_delays_by_comp = meta.get("fused_W_delays_by_comp")
    fused_n_post_by_comp = meta.get("fused_W_n_post_by_comp")
    assert isinstance(fused_by_comp, dict)
    assert isinstance(fused_delays_by_comp, dict)
    assert isinstance(fused_n_post_by_comp, dict)

    fused = fused_by_comp[Compartment.SOMA]
    fused_delays = fused_delays_by_comp[Compartment.SOMA].tolist()
    assert fused_delays == delays
    n_post = int(meta["n_post"])
    n_pre = int(meta["n_pre"])
    assert fused.shape == (len(delays) * n_post, n_pre)
