import pytest

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.adex_3c import AdEx3CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.biophysics.models.template_neuron import TemplateNeuronModel
from biosnn.contracts.neurons import NeuronInputs, StepContext

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "model_factory",
    [
        GLIFModel,
        AdEx2CompModel,
        AdEx3CompModel,
        TemplateNeuronModel,
    ],
)
def test_neuron_model_contracts(model_factory) -> None:
    model = model_factory()
    n = 6
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(n, ctx=ctx)

    device = torch.device("cpu")
    dtype = torch.float32
    drive = {comp: torch.zeros((n,), device=device, dtype=dtype) for comp in model.compartments}

    new_state, result = model.step(state, NeuronInputs(drive=drive), dt=1e-3, t=0.0, ctx=ctx)

    assert result.spikes.shape == (n,)
    assert result.spikes.dtype is torch.bool

    for tensor in model.state_tensors(new_state).values():
        assert tensor.device.type == "cpu"
        assert tensor.dtype == dtype
