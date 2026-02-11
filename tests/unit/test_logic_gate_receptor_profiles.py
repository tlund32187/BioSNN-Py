from __future__ import annotations

import pytest

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import ReceptorKind
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


def test_logic_gate_ff_splits_hidden_out_and_targets_compartments() -> None:
    _, topology, _ = build_logic_gate_ff(
        "and",
        device="cpu",
        seed=13,
        run_spec={
            "synapse": {
                "backend": "event_driven",
                "ring_strategy": "event_bucketed",
                "fused_layout": "csr",
                "ring_dtype": "float16",
                "receptor_mode": "ei_ampa_nmda_gabaa_gabab",
            }
        },
    )
    projections = {proj.name: proj for proj in topology.projections}

    assert "Hidden->Out" not in projections
    assert "HiddenExcit->Out" in projections
    assert "HiddenInhib->Out" in projections

    assert projections["In->Hidden"].topology.target_compartment == Compartment.DENDRITE
    assert projections["HiddenExcit->Out"].topology.target_compartment == Compartment.DENDRITE
    assert projections["HiddenInhib->Out"].topology.target_compartment == Compartment.SOMA
    assert projections["HiddenInhib->HiddenExcit"].topology.target_compartment == Compartment.SOMA

    syn = projections["HiddenInhib->Out"].synapse
    assert isinstance(syn, DelayedSparseMatmulSynapse)
    assert syn.params.backend == "event_driven"
    assert syn.params.ring_strategy == "event_bucketed"
    assert syn.params.fused_layout == "csr"
    assert syn.params.ring_dtype == "float16"
    assert syn.params.receptor_profile is not None
    assert ReceptorKind.GABA_A in syn.params.receptor_profile.kinds
    assert ReceptorKind.GABA_B in syn.params.receptor_profile.kinds


def test_logic_gate_ff_inhibitory_profile_uses_gabaa_only_mode() -> None:
    _, topology, _ = build_logic_gate_ff(
        "and",
        device="cpu",
        seed=17,
        run_spec={"synapse": {"receptor_mode": "ei_ampa_nmda_gabaa"}},
    )
    projections = {proj.name: proj for proj in topology.projections}
    syn = projections["HiddenInhib->Out"].synapse

    assert isinstance(syn, DelayedSparseMatmulSynapse)
    assert syn.params.receptor_profile is not None
    assert syn.params.receptor_profile.kinds == (ReceptorKind.GABA_A,)
