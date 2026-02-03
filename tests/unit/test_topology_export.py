import pytest

from biosnn.contracts.synapses import ReceptorKind, SynapseTopology
from biosnn.synapses import export_topology_json

torch = pytest.importorskip("torch")


def test_export_topology_json_basic():
    pre_idx = torch.tensor([0, 1], dtype=torch.long)
    post_idx = torch.tensor([0, 0], dtype=torch.long)
    pre_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    post_pos = torch.tensor([[0.5, 1.0, 0.0]])
    receptor = torch.tensor([0, 1], dtype=torch.long)
    weights = torch.tensor([0.2, -0.3])

    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        receptor=receptor,
        receptor_kinds=(ReceptorKind.AMPA, ReceptorKind.GABA),
        pre_pos=pre_pos,
        post_pos=post_pos,
    )

    payload = export_topology_json(topology, weights=weights)

    assert len(payload["nodes"]) == 3
    assert len(payload["edges"]) == 2
    assert payload["edges"][0]["receptor"] == "ampa"
    assert payload["edges"][1]["receptor"] == "gaba"
