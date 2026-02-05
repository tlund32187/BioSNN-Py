import pytest
pytestmark = pytest.mark.unit


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


def test_adex2c_regression_snapshot():
    model = AdEx2CompModel()
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(2, ctx=ctx)

    state.v_soma.copy_(
        torch.tensor([-0.07, -0.06], device=state.v_soma.device, dtype=state.v_soma.dtype)
    )
    state.v_dend.copy_(
        torch.tensor([-0.07, -0.05], device=state.v_dend.device, dtype=state.v_dend.dtype)
    )
    state.w.zero_()
    state.refrac_left.zero_()
    state.spike_hold_left.zero_()

    drive_s = torch.tensor([1.0e-9, 2.0e-9], device=state.v_soma.device, dtype=state.v_soma.dtype)
    drive_d = torch.tensor([0.5e-9, 0.0], device=state.v_dend.device, dtype=state.v_dend.dtype)
    inputs = NeuronInputs(drive={Compartment.SOMA: drive_s, Compartment.DENDRITE: drive_d})

    next_state, result = model.step(state, inputs, dt=1.0e-3, t=0.0, ctx=ctx)

    expected_v_soma = torch.tensor(
        [-0.06499999761581421, -0.05024932324886322],
        device=state.v_soma.device,
        dtype=state.v_soma.dtype,
    )
    expected_v_dend = torch.tensor(
        [-0.06750000268220901, -0.051249999552965164],
        device=state.v_dend.device,
        dtype=state.v_dend.dtype,
    )
    expected_w = torch.tensor(
        [5.0000026227637814e-14, 1.9750676517448634e-13],
        device=state.w.device,
        dtype=state.w.dtype,
    )

    torch.testing.assert_close(next_state.v_soma, expected_v_soma, rtol=1e-5, atol=1e-10)
    torch.testing.assert_close(next_state.v_dend, expected_v_dend, rtol=1e-5, atol=1e-10)
    torch.testing.assert_close(next_state.w, expected_w, rtol=1e-5, atol=1e-10)
    assert result.spikes.tolist() == [False, False]
