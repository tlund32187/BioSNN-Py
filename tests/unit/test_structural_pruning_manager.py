from __future__ import annotations

import pytest

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import SynapseTopology
from biosnn.plasticity.structural import StructuralPlasticityManager, StructuralPruningConfig
from biosnn.simulation.network import ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _projection(*, name: str = "A_to_B") -> ProjectionSpec:
    topology = SynapseTopology(
        pre_idx=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        post_idx=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        weights=torch.tensor([0.05, 0.05, 0.05, 0.05], dtype=torch.float32),
        target_compartment=Compartment.DENDRITE,
        meta={"n_pre": 2, "n_post": 2},
    )
    return ProjectionSpec(
        name=name,
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.0)),
        topology=topology,
        pre="A",
        post="B",
    )


def test_structural_manager_prunes_low_usage_edges() -> None:
    proj = _projection()
    manager = StructuralPlasticityManager(
        StructuralPruningConfig(
            enabled=True,
            prune_interval_steps=1,
            usage_alpha=1.0,
            w_min=0.1,
            usage_min=0.1,
            k_min_out=0,
            k_min_in=0,
            max_prune_fraction_per_interval=1.0,
        )
    )
    manager.reset([proj], device="cpu", dtype=torch.float32)
    manager.record_projection_activity(
        projection=proj,
        pre_spikes=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    decisions = manager.maybe_prune(
        step_idx=1,
        projections=[proj],
        weights_by_projection={proj.name: proj.topology.weights},
    )

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.pruned_edges == 2
    assert decision.new_edges == 2
    torch.testing.assert_close(
        decision.topology.pre_idx,
        torch.tensor([0, 0], dtype=torch.long),
    )


def test_structural_manager_never_prunes_all_edges() -> None:
    proj = _projection()
    manager = StructuralPlasticityManager(
        StructuralPruningConfig(
            enabled=True,
            prune_interval_steps=1,
            usage_alpha=1.0,
            w_min=1.0,
            usage_min=2.0,
            k_min_out=0,
            k_min_in=0,
            max_prune_fraction_per_interval=1.0,
        )
    )
    manager.reset([proj], device="cpu", dtype=torch.float32)
    manager.record_projection_activity(
        projection=proj,
        pre_spikes=torch.zeros((2,), dtype=torch.float32),
    )
    decisions = manager.maybe_prune(
        step_idx=1,
        projections=[proj],
        weights_by_projection={proj.name: proj.topology.weights},
    )

    assert len(decisions) == 1
    assert decisions[0].new_edges == 1
    assert decisions[0].pruned_edges == 3


def test_structural_manager_respects_prune_interval() -> None:
    proj = _projection()
    manager = StructuralPlasticityManager(
        StructuralPruningConfig(
            enabled=True,
            prune_interval_steps=5,
            usage_alpha=1.0,
            w_min=1.0,
            usage_min=2.0,
        )
    )
    manager.reset([proj], device="cpu", dtype=torch.float32)
    manager.record_projection_activity(
        projection=proj,
        pre_spikes=torch.zeros((2,), dtype=torch.float32),
    )
    decisions = manager.maybe_prune(
        step_idx=4,
        projections=[proj],
        weights_by_projection={proj.name: proj.topology.weights},
    )
    assert decisions == []
