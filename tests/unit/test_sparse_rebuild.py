from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit


from biosnn.connectivity.sparse_rebuild import rebuild_sparse_delay_mats
from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import ReceptorKind, SynapseTopology
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

torch = pytest.importorskip("torch")


def _clone_values(values_by_comp):
    cloned = {}
    for comp, values in values_by_comp.items():
        cloned[comp] = [val.clone() if val is not None else None for val in values]
    return cloned


def test_sparse_rebuild_matches_updated_values() -> None:
    comp_order = tuple(Compartment)
    soma_id = comp_order.index(Compartment.SOMA)
    dend_id = comp_order.index(Compartment.DENDRITE)

    pre_idx = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    post_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    receptor = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    target_compartments = torch.tensor([soma_id, dend_id, soma_id, dend_id], dtype=torch.long)
    weights = torch.zeros((pre_idx.numel(),), dtype=torch.float32)

    receptor_scale = {ReceptorKind.AMPA: 2.0, ReceptorKind.NMDA: 0.5}
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        receptor=receptor,
        receptor_kinds=(ReceptorKind.AMPA, ReceptorKind.NMDA),
        weights=weights,
        target_compartment=Compartment.DENDRITE,
        target_compartments=target_compartments,
        meta={"receptor_scale": receptor_scale},
    )

    topology = compile_topology(
        topology,
        device="cpu",
        dtype=weights.dtype,
        build_sparse_delay_mats=True,
        build_bucket_edge_mapping=True,
    )

    synapse = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(init_weight=0.0, receptor_scale=receptor_scale)
    )
    active_edges = torch.tensor([0, 3], dtype=torch.long)
    dW = torch.tensor([0.5, -0.25], dtype=weights.dtype)
    synapse.apply_weight_updates(topology, active_edges, dW)

    meta_before = topology.meta or {}
    values_before = _clone_values(meta_before["values_by_comp"])

    topology = rebuild_sparse_delay_mats(
        topology,
        device="cpu",
        dtype=weights.dtype,
        build_bucket_edge_mapping=True,
    )

    meta_after = topology.meta or {}
    values_after = meta_after["values_by_comp"]
    for comp, values in values_before.items():
        for idx, val in enumerate(values):
            new_val = values_after[comp][idx]
            if val is None:
                assert new_val is None
            else:
                assert new_val is not None
                torch.testing.assert_close(new_val, val)
