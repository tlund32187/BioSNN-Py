import pytest

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.neurons import Compartment, NeuronInputs, StepContext

torch = pytest.importorskip("torch")


def test_glif_step_shapes():
    model = GLIFModel()
    n = 4
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(n, ctx=ctx)

    drive = torch.zeros(n, device=state.v_soma.device, dtype=state.v_soma.dtype)
    inputs = NeuronInputs(drive={Compartment.SOMA: drive})
    next_state, result = model.step(state, inputs, dt=0.001, t=0.0, ctx=ctx)

    assert result.spikes.shape == (n,)
    assert result.spikes.dtype == torch.bool
    assert result.membrane is not None
    assert result.membrane[Compartment.SOMA].shape == (n,)

    tensors = model.state_tensors(next_state)
    assert "v_soma" in tensors
    assert tensors["v_soma"].shape == (n,)


def test_glif_reset_subset():
    model = GLIFModel()
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(3, ctx=ctx)

    state.v_soma[:] = torch.tensor([1.0, 2.0, 3.0], device=state.v_soma.device)
    indices = torch.tensor([0, 2], device=state.v_soma.device, dtype=torch.long)
    model.reset_state(state, ctx=ctx, indices=indices)

    assert state.v_soma[0].item() == pytest.approx(model.params.v_rest)
    assert state.v_soma[1].item() == pytest.approx(2.0)
    assert state.v_soma[2].item() == pytest.approx(model.params.v_rest)


def test_adex2c_step_shapes():
    model = AdEx2CompModel()
    n = 5
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(n, ctx=ctx)

    drive_s = torch.zeros(n, device=state.v_soma.device, dtype=state.v_soma.dtype)
    drive_d = torch.zeros(n, device=state.v_dend.device, dtype=state.v_dend.dtype)
    inputs = NeuronInputs(drive={Compartment.SOMA: drive_s, Compartment.DENDRITE: drive_d})
    next_state, result = model.step(state, inputs, dt=0.001, t=0.0, ctx=ctx)

    assert result.spikes.shape == (n,)
    assert result.spikes.dtype == torch.bool
    assert result.membrane is not None
    assert result.membrane[Compartment.SOMA].shape == (n,)
    assert result.membrane[Compartment.DENDRITE].shape == (n,)

    tensors = model.state_tensors(next_state)
    assert "v_soma" in tensors
    assert "v_dend" in tensors
    assert "w" in tensors


def test_adex2c_reset_subset():
    model = AdEx2CompModel()
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(4, ctx=ctx)

    state.v_soma[:] = torch.tensor([1.0, 2.0, 3.0, 4.0], device=state.v_soma.device)
    state.v_dend[:] = torch.tensor([5.0, 6.0, 7.0, 8.0], device=state.v_dend.device)
    state.w[:] = torch.tensor([0.1, 0.2, 0.3, 0.4], device=state.w.device)

    indices = torch.tensor([1, 3], device=state.v_soma.device, dtype=torch.long)
    model.reset_state(state, ctx=ctx, indices=indices)

    assert state.v_soma[0].item() == pytest.approx(1.0)
    assert state.v_soma[1].item() == pytest.approx(model.params.e_l_s)
    assert state.v_soma[3].item() == pytest.approx(model.params.e_l_s)
    assert state.v_dend[1].item() == pytest.approx(model.params.e_l_d)
    assert state.w[1].item() == pytest.approx(0.0)
