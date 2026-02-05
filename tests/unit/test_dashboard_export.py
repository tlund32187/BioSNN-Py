import pytest

from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.neurons import Compartment, NeuronInputs, StepContext
from biosnn.contracts.synapses import SynapseTopology
from biosnn.io.dashboard_export import export_dashboard_snapshot, export_neuron_snapshot

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_export_dashboard_snapshot(tmp_path):
    pre_idx = torch.tensor([0, 1], dtype=torch.long)
    post_idx = torch.tensor([0, 0], dtype=torch.long)
    weights = torch.tensor([0.2, -0.3])
    v_soma = torch.tensor([-0.06, -0.07])
    topology = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx)

    paths = export_dashboard_snapshot(
        topology,
        weights,
        out_dir=tmp_path,
        neuron_tensors={"v_soma": v_soma},
        neuron_spikes=torch.tensor([1, 0]),
    )

    assert paths["topology"].exists()
    assert paths["synapse"].exists()
    assert paths["neuron"].exists()


def test_export_neuron_snapshot(tmp_path):
    model = GLIFModel()
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(2, ctx=ctx)
    drive = torch.zeros(2, device=state.v_soma.device, dtype=state.v_soma.dtype)
    inputs = NeuronInputs(drive={Compartment.SOMA: drive})
    state, result = model.step(state, inputs, dt=1e-3, t=0.0, ctx=ctx)

    path = export_neuron_snapshot(model, state, result, path=tmp_path / "neuron.csv")

    assert path.exists()
