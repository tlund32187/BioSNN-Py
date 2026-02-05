import pytest

from biosnn.connectivity.topology_compile import (
    _build_compiled_artifacts,
    _finalize_topology,
    _normalize_topology_inputs,
)
from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import SynapseTopology

torch = pytest.importorskip("torch")


def test_compile_phase_helpers_smoke():
    pre_idx = torch.tensor([0, 1], dtype=torch.long)
    post_idx = torch.tensor([0, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 1], dtype=torch.int32)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
    )

    normalized, base_meta = _normalize_topology_inputs(topology, device="cpu", dtype="float32")
    assert base_meta["n_pre"] == 2
    assert base_meta["n_post"] == 2
    assert base_meta["max_delay_steps"] == 1

    compiled_meta = _build_compiled_artifacts(
        topology,
        normalized,
        base_meta,
        build_edges_by_delay=True,
        build_pre_adjacency=True,
        build_sparse_delay_mats=False,
        build_bucket_edge_mapping=False,
        fuse_delay_buckets=True,
        store_sparse_by_delay=True,
    )
    assert "edges_by_delay" in compiled_meta
    assert "pre_ptr" in compiled_meta
    assert "edge_idx" in compiled_meta

    finalized = _finalize_topology(topology, normalized, base_meta, compiled_meta)
    assert finalized is not topology
    assert finalized.meta is not None
    assert finalized.meta["n_pre"] == 2
