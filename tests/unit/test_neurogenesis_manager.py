from __future__ import annotations

import pytest

from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import SynapseTopology
from biosnn.plasticity.structural import NeurogenesisConfig, NeurogenesisManager
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _build_specs() -> tuple[list[PopulationSpec], list[ProjectionSpec]]:
    input_pop = PopulationSpec(name="Input0", model=GLIFModel(), n=2)
    hidden_pop = PopulationSpec(name="Hidden0", model=GLIFModel(), n=4)
    output_pop = PopulationSpec(name="Output0", model=GLIFModel(), n=2)

    in_to_hidden = ProjectionSpec(
        name="Input0_to_Hidden0",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.0)),
        topology=SynapseTopology(
            pre_idx=torch.tensor([0, 1], dtype=torch.long),
            post_idx=torch.tensor([0, 1], dtype=torch.long),
            delay_steps=torch.tensor([0, 1], dtype=torch.long),
            weights=torch.tensor([0.05, 0.05], dtype=torch.float32),
            target_compartment=Compartment.DENDRITE,
            meta={"n_pre": 2, "n_post": 4},
        ),
        pre="Input0",
        post="Hidden0",
    )
    hidden_to_out = ProjectionSpec(
        name="Hidden0_to_Output0",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.0)),
        topology=SynapseTopology(
            pre_idx=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            post_idx=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            delay_steps=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            weights=torch.tensor([0.04, 0.04, 0.04, 0.04], dtype=torch.float32),
            target_compartment=Compartment.DENDRITE,
            meta={"n_pre": 4, "n_post": 2},
        ),
        pre="Hidden0",
        post="Output0",
    )
    return [input_pop, hidden_pop, output_pop], [in_to_hidden, hidden_to_out]


def test_neurogenesis_manager_grows_hidden_population_and_marks_newborn_edges() -> None:
    populations, projections = _build_specs()
    manager = NeurogenesisManager(
        NeurogenesisConfig(
            enabled=True,
            growth_interval_steps=2,
            add_neurons_per_event=2,
            max_total_neurons=64,
            connectivity_p=0.5,
            activity_threshold=1.0,
            plateau_intervals=1,
        )
    )
    manager.reset(populations)
    manager.record_step(spikes_by_pop={"Hidden0": torch.zeros((4,), dtype=torch.bool)})
    decision = manager.maybe_grow(
        step_idx=2,
        populations=populations,
        projections=projections,
    )

    assert decision is not None
    assert decision.target_population == "Hidden0"
    assert decision.added_neurons == 2
    assert decision.added_edges > 0

    hidden = next(pop for pop in decision.populations if pop.name == "Hidden0")
    assert hidden.n == 6
    for proj in decision.projections:
        if proj.pre != "Hidden0" and proj.post != "Hidden0":
            continue
        meta = proj.topology.meta or {}
        newborn_mask = meta.get("newborn_edge_mask")
        assert isinstance(newborn_mask, torch.Tensor)
        assert newborn_mask.dtype == torch.bool
        assert newborn_mask.numel() == proj.topology.pre_idx.numel()
        assert bool(newborn_mask.any())
        assert int(meta.get("newborn_until_step", 0)) >= 2


def test_neurogenesis_manager_respects_total_neuron_cap() -> None:
    populations, projections = _build_specs()
    total_before = sum(pop.n for pop in populations)
    cap = total_before + 1
    manager = NeurogenesisManager(
        NeurogenesisConfig(
            enabled=True,
            growth_interval_steps=1,
            add_neurons_per_event=4,
            max_total_neurons=cap,
            connectivity_p=0.5,
            activity_threshold=1.0,
            plateau_intervals=1,
        )
    )
    manager.reset(populations)
    manager.record_step(spikes_by_pop={"Hidden0": torch.zeros((4,), dtype=torch.bool)})

    decision = manager.maybe_grow(
        step_idx=1,
        populations=populations,
        projections=projections,
    )
    assert decision is not None
    assert decision.added_neurons == 1
    assert sum(pop.n for pop in decision.populations) == cap

    manager.on_structure_changed(decision.populations)
    grown_hidden_n = next(pop.n for pop in decision.populations if pop.name == "Hidden0")
    manager.record_step(
        spikes_by_pop={"Hidden0": torch.zeros((grown_hidden_n,), dtype=torch.bool)}
    )
    no_second_growth = manager.maybe_grow(
        step_idx=2,
        populations=decision.populations,
        projections=decision.projections,
    )

    assert no_second_growth is None
    assert manager.scalars()["neurogenesis/total_neurons"] == float(cap)
