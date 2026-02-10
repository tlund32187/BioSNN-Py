from __future__ import annotations

import pytest

from biosnn.synapses.dynamics.delayed_sparse_matmul import DelayedSparseMatmulSynapse
from biosnn.tasks.logic_gates.topologies import build_logic_gate_ff, build_logic_gate_xor

pytestmark = pytest.mark.unit


def _assert_projection_profiles(topology) -> None:
    for projection in topology.projections:
        assert isinstance(projection.synapse, DelayedSparseMatmulSynapse)
        assert projection.synapse.params.receptor_profile is not None


def test_logic_gate_ff_uses_receptor_profiles() -> None:
    _, topology, _ = build_logic_gate_ff("and", device="cpu", seed=7)
    _assert_projection_profiles(topology)


def test_logic_gate_xor_uses_receptor_profiles() -> None:
    _, topology, _ = build_logic_gate_xor(device="cpu", seed=11)
    _assert_projection_profiles(topology)
