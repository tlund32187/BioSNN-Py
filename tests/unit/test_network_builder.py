import pytest
pytestmark = pytest.mark.unit


from biosnn.api.builders.network_builder import ErdosRenyi, Init, NetworkBuilder
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.synapses import SynapseTopology
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentSynapse

torch = pytest.importorskip("torch")


def _make_builder(seed: int) -> NetworkBuilder:
    return (
        NetworkBuilder()
        .device("cpu")
        .dtype("float32")
        .seed(seed)
        .population("a", n=8, neuron=GLIFModel())
        .population("b", n=4, neuron=GLIFModel())
    )


def test_build_minimal_network() -> None:
    builder = _make_builder(123).projection(
        "a",
        "b",
        synapse=DelayedCurrentSynapse(),
        topology=ErdosRenyi(p=0.4),
        weights=Init.constant(0.05),
    )

    spec = builder.build()

    assert len(spec.populations) == 2
    assert len(spec.projections) == 1
    assert spec.projections[0].pre == "a"
    assert spec.projections[0].post == "b"
    assert spec.projections[0].topology.weights is not None


def test_builder_duplicate_population_name() -> None:
    builder = NetworkBuilder().population("dup", n=2, neuron=GLIFModel())
    with pytest.raises(ValueError, match="already exists"):
        builder.population("dup", n=3, neuron=GLIFModel())


def test_builder_missing_population() -> None:
    builder = NetworkBuilder().population("a", n=2, neuron=GLIFModel()).projection(
        "a",
        "missing",
        synapse=DelayedCurrentSynapse(),
        topology=ErdosRenyi(p=0.2),
    )
    with pytest.raises(ValueError, match="post population"):
        builder.build()


def test_builder_dim_mismatch() -> None:
    pre_idx = torch.tensor([0, 8], dtype=torch.long)
    post_idx = torch.tensor([0, 1], dtype=torch.long)
    topo = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx)
    builder = _make_builder(1).projection(
        "a",
        "b",
        synapse=DelayedCurrentSynapse(),
        topology=topo,
    )
    with pytest.raises(ValueError, match="pre_idx exceeds"):
        builder.build()


def test_builder_reproducible_with_seed() -> None:
    b1 = _make_builder(7).projection(
        "a",
        "b",
        synapse=DelayedCurrentSynapse(),
        topology=ErdosRenyi(p=0.3),
        weights=Init.normal(mean=0.0, std=0.1),
    )
    b2 = _make_builder(7).projection(
        "a",
        "b",
        synapse=DelayedCurrentSynapse(),
        topology=ErdosRenyi(p=0.3),
        weights=Init.normal(mean=0.0, std=0.1),
    )

    spec1 = b1.build()
    spec2 = b2.build()
    topo1 = spec1.projections[0].topology
    topo2 = spec2.projections[0].topology

    assert torch.equal(topo1.pre_idx, topo2.pre_idx)
    assert torch.equal(topo1.post_idx, topo2.post_idx)
    assert topo1.weights is not None
    assert topo2.weights is not None
    assert torch.allclose(topo1.weights, topo2.weights)
