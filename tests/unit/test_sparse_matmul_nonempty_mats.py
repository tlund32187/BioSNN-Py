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
    compile_topology(
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
