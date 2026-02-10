from __future__ import annotations

import pytest

from biosnn.contracts.synapses import SynapseTopology
from biosnn.io.graph.population_topology_json import build_population_topology_payload

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_population_topology_payload_stats():
    populations = [
        {"name": "A", "n": 2, "model_name": "glif", "layer": 0},
        {"name": "B", "n": 1, "model_name": "adex_2c", "layer": 1},
    ]

    topology = SynapseTopology(
        pre_idx=torch.tensor([0, 1], dtype=torch.long),
        post_idx=torch.tensor([0, 0], dtype=torch.long),
        weights=torch.tensor([0.5, -0.25], dtype=torch.float32),
        delay_steps=torch.tensor([2, 4], dtype=torch.int32),
    )

    projections = [
        {"name": "A_to_B", "pre": "A", "post": "B", "topology": topology},
    ]

    payload = build_population_topology_payload(populations, projections)
    assert payload["mode"] == "population"

    nodes = {node["id"]: node for node in payload["nodes"]}
    assert nodes["A"]["n_neurons"] == 2
    assert nodes["B"]["n_neurons"] == 1

    edge = payload["edges"][0]
    assert edge["n_synapses"] == 2
    assert edge["mean_weight"] == pytest.approx(0.125)
    assert edge["mean_delay_steps"] == pytest.approx(3.0)


def test_population_topology_infers_layers_from_role_and_short_names() -> None:
    populations = [
        {"name": "In", "n": 2, "model_name": "lif_3c", "meta": {"role": "input"}},
        {"name": "Hidden", "n": 2, "model_name": "lif_3c", "meta": {"role": "hidden"}},
        {"name": "Out", "n": 1, "model_name": "lif_3c", "meta": {"role": "output"}},
    ]
    topology = SynapseTopology(
        pre_idx=torch.tensor([0, 1], dtype=torch.long),
        post_idx=torch.tensor([0, 0], dtype=torch.long),
        weights=torch.tensor([0.1, 0.2], dtype=torch.float32),
        delay_steps=torch.tensor([0, 1], dtype=torch.int32),
    )
    projections = [
        {"name": "In->Hidden", "pre": "In", "post": "Hidden", "topology": topology},
        {"name": "Hidden->Out", "pre": "Hidden", "post": "Out", "topology": topology},
    ]

    payload = build_population_topology_payload(populations, projections)
    nodes = {node["id"]: node for node in payload["nodes"]}
    assert nodes["In"]["layer"] == 0
    assert nodes["Hidden"]["layer"] == 1
    assert nodes["Out"]["layer"] == 2
