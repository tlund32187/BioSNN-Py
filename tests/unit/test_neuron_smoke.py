import pytest

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.neurons import (
    AdEx2CompParams,
    Compartment,
    GLIFParams,
    NeuronInputs,
    StepContext,
)

torch = pytest.importorskip("torch")


def _ctx(device: str = "cpu", dtype: str = "float32", **extras: object) -> StepContext:
    return StepContext(device=device, dtype=dtype, extras=extras or None)


def _drive(value: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((n,), value, device=device, dtype=dtype)


def _step_glif(
    model: GLIFModel,
    state,
    drive: torch.Tensor,
    *,
    dt: float,
    t: float,
    ctx: StepContext,
):
    inputs = NeuronInputs(drive={Compartment.SOMA: drive})
    return model.step(state, inputs, dt=dt, t=t, ctx=ctx)


def _step_adex(
    model: AdEx2CompModel,
    state,
    drive_s: torch.Tensor,
    drive_d: torch.Tensor,
    *,
    dt: float,
    t: float,
    ctx: StepContext,
):
    inputs = NeuronInputs(
        drive={Compartment.SOMA: drive_s, Compartment.DENDRITE: drive_d}
    )
    return model.step(state, inputs, dt=dt, t=t, ctx=ctx)


def test_glif_smoke_init_shapes():
    model = GLIFModel()
    n = 4
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)

    assert state.v_soma.shape == (n,)
    assert state.v_soma.device.type == "cpu"
    assert state.v_soma.dtype == torch.float32

    drive = torch.zeros(n, device=state.v_soma.device, dtype=state.v_soma.dtype)
    next_state, result = _step_glif(model, state, drive, dt=1e-3, t=0.0, ctx=ctx)

    assert result.spikes.shape == (n,)
    assert result.membrane is not None
    assert result.membrane[Compartment.SOMA].shape == (n,)
    assert next_state.theta.shape == (n,)


def test_adex_smoke_init_shapes():
    model = AdEx2CompModel()
    n = 4
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)

    assert state.v_soma.shape == (n,)
    assert state.v_dend.shape == (n,)
    assert state.w.shape == (n,)
    assert state.v_soma.device.type == "cpu"

    drive_s = torch.zeros(n, device=state.v_soma.device, dtype=state.v_soma.dtype)
    drive_d = torch.zeros(n, device=state.v_dend.device, dtype=state.v_dend.dtype)
    next_state, result = _step_adex(
        model, state, drive_s, drive_d, dt=1e-3, t=0.0, ctx=ctx
    )

    assert result.spikes.shape == (n,)
    assert result.membrane is not None
    assert result.membrane[Compartment.SOMA].shape == (n,)
    assert result.membrane[Compartment.DENDRITE].shape == (n,)
    assert next_state.w.shape == (n,)


def test_glif_rest_stability_no_drive():
    model = GLIFModel()
    n = 3
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)
    drive = torch.zeros(n, device=state.v_soma.device, dtype=state.v_soma.dtype)

    for step in range(10):
        state, result = _step_glif(model, state, drive, dt=1e-3, t=step * 1e-3, ctx=ctx)
        assert not bool(result.spikes.any())

    expected = torch.full((n,), model.params.v_rest, device=state.v_soma.device, dtype=state.v_soma.dtype)
    torch.testing.assert_close(state.v_soma, expected, atol=1e-6, rtol=0.0)


def test_adex_rest_stability_no_drive():
    model = AdEx2CompModel()
    n = 3
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)
    drive_s = torch.zeros(n, device=state.v_soma.device, dtype=state.v_soma.dtype)
    drive_d = torch.zeros(n, device=state.v_dend.device, dtype=state.v_dend.dtype)

    for step in range(10):
        state, result = _step_adex(
            model, state, drive_s, drive_d, dt=1e-3, t=step * 1e-3, ctx=ctx
        )
        assert not bool(result.spikes.any())

    expected_s = torch.full(
        (n,), model.params.e_l_s, device=state.v_soma.device, dtype=state.v_soma.dtype
    )
    expected_d = torch.full(
        (n,), model.params.e_l_d, device=state.v_dend.device, dtype=state.v_dend.dtype
    )
    torch.testing.assert_close(state.v_soma, expected_s, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(state.v_dend, expected_d, atol=1e-5, rtol=0.0)


def test_glif_spikes_under_constant_drive():
    model = GLIFModel()
    n = 2
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)
    drive = _drive(2e-9, n, state.v_soma.device, state.v_soma.dtype)

    spike_any = False
    for step in range(30):
        state, result = _step_glif(model, state, drive, dt=1e-3, t=step * 1e-3, ctx=ctx)
        spike_any = spike_any or bool(result.spikes.any())

    assert spike_any


def test_adex_spikes_under_constant_drive():
    model = AdEx2CompModel()
    n = 2
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)
    drive_s = _drive(5e-8, n, state.v_soma.device, state.v_soma.dtype)
    drive_d = _drive(0.0, n, state.v_dend.device, state.v_dend.dtype)

    spike_any = False
    for step in range(10):
        state, result = _step_adex(
            model, state, drive_s, drive_d, dt=1e-3, t=step * 1e-3, ctx=ctx
        )
        spike_any = spike_any or bool(result.spikes.any())

    assert spike_any


def test_glif_refractory_enforced():
    params = GLIFParams(refrac_period=0.003, spike_hold_time=0.0, enable_threshold_adaptation=False)
    model = GLIFModel(params)
    n = 1
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)
    drive = _drive(5e-8, n, state.v_soma.device, state.v_soma.dtype)

    spikes = []
    for step in range(6):
        state, result = _step_glif(model, state, drive, dt=1e-3, t=step * 1e-3, ctx=ctx)
        spikes.append(bool(result.spikes.item()))

    assert spikes[0]
    assert spikes[1] is False
    assert spikes[2] is False
    assert spikes[3] is False


def test_adex_refractory_enforced():
    params = AdEx2CompParams(refrac_period=0.003, spike_hold_time=0.0)
    model = AdEx2CompModel(params)
    n = 1
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)
    drive_s = _drive(5e-8, n, state.v_soma.device, state.v_soma.dtype)
    drive_d = _drive(0.0, n, state.v_dend.device, state.v_dend.dtype)

    spikes = []
    for step in range(6):
        state, result = _step_adex(
            model, state, drive_s, drive_d, dt=1e-3, t=step * 1e-3, ctx=ctx
        )
        spikes.append(bool(result.spikes.item()))

    assert spikes[0]
    assert spikes[1] is False
    assert spikes[2] is False
    assert spikes[3] is False


def test_glif_spike_hold_duration():
    params = GLIFParams(spike_hold_time=0.002, refrac_period=0.0, enable_threshold_adaptation=False)
    model = GLIFModel(params)
    n = 1
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)

    drive_on = _drive(5e-8, n, state.v_soma.device, state.v_soma.dtype)
    drive_off = _drive(0.0, n, state.v_soma.device, state.v_soma.dtype)

    spikes = []
    state, result = _step_glif(model, state, drive_on, dt=1e-3, t=0.0, ctx=ctx)
    spikes.append(bool(result.spikes.item()))
    for step in range(1, 4):
        state, result = _step_glif(model, state, drive_off, dt=1e-3, t=step * 1e-3, ctx=ctx)
        spikes.append(bool(result.spikes.item()))

    assert spikes[0]
    assert spikes[1]
    assert spikes[2] is False


def test_adex_spike_hold_duration():
    params = AdEx2CompParams(spike_hold_time=0.002, refrac_period=0.0)
    model = AdEx2CompModel(params)
    n = 1
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)

    drive_on = _drive(5e-8, n, state.v_soma.device, state.v_soma.dtype)
    drive_off = _drive(0.0, n, state.v_soma.device, state.v_soma.dtype)

    spikes = []
    state, result = _step_adex(model, state, drive_on, drive_off, dt=1e-3, t=0.0, ctx=ctx)
    spikes.append(bool(result.spikes.item()))
    for step in range(1, 4):
        state, result = _step_adex(model, state, drive_off, drive_off, dt=1e-3, t=step * 1e-3, ctx=ctx)
        spikes.append(bool(result.spikes.item()))

    assert spikes[0]
    assert spikes[1]
    assert spikes[2] is False


def test_glif_threshold_adaptation_toggle():
    n = 1
    drive_value = 5e-8

    params_on = GLIFParams(enable_threshold_adaptation=True)
    model_on = GLIFModel(params_on)
    ctx = _ctx()
    state_on = model_on.init_state(n, ctx=ctx)
    drive = _drive(drive_value, n, state_on.v_soma.device, state_on.v_soma.dtype)
    state_on, result_on = _step_glif(model_on, state_on, drive, dt=1e-3, t=0.0, ctx=ctx)

    params_off = GLIFParams(enable_threshold_adaptation=False)
    model_off = GLIFModel(params_off)
    state_off = model_off.init_state(n, ctx=ctx)
    state_off, result_off = _step_glif(model_off, state_off, drive, dt=1e-3, t=0.0, ctx=ctx)

    assert bool(result_on.spikes.item())
    assert bool(result_off.spikes.item())
    assert state_on.theta.item() > 0.0
    assert state_off.theta.item() == 0.0


def test_cpu_determinism_glif():
    model = GLIFModel()
    n = 3
    ctx = _ctx()
    state_a = model.init_state(n, ctx=ctx)
    state_b = model.init_state(n, ctx=ctx)
    drive = _drive(1e-9, n, state_a.v_soma.device, state_a.v_soma.dtype)

    spikes_a = []
    spikes_b = []
    for step in range(5):
        state_a, result_a = _step_glif(model, state_a, drive, dt=1e-3, t=step * 1e-3, ctx=ctx)
        state_b, result_b = _step_glif(model, state_b, drive, dt=1e-3, t=step * 1e-3, ctx=ctx)
        spikes_a.append(result_a.spikes.clone())
        spikes_b.append(result_b.spikes.clone())

    torch.testing.assert_close(state_a.v_soma, state_b.v_soma)
    torch.testing.assert_close(state_a.refrac_left, state_b.refrac_left)
    torch.testing.assert_close(state_a.spike_hold_left, state_b.spike_hold_left)
    torch.testing.assert_close(state_a.theta, state_b.theta)
    for a, b in zip(spikes_a, spikes_b, strict=True):
        assert torch.equal(a, b)


def test_cpu_determinism_adex():
    model = AdEx2CompModel()
    n = 3
    ctx = _ctx()
    state_a = model.init_state(n, ctx=ctx)
    state_b = model.init_state(n, ctx=ctx)
    drive_s = _drive(1e-9, n, state_a.v_soma.device, state_a.v_soma.dtype)
    drive_d = _drive(5e-10, n, state_a.v_dend.device, state_a.v_dend.dtype)

    spikes_a = []
    spikes_b = []
    for step in range(5):
        state_a, result_a = _step_adex(
            model, state_a, drive_s, drive_d, dt=1e-3, t=step * 1e-3, ctx=ctx
        )
        state_b, result_b = _step_adex(
            model, state_b, drive_s, drive_d, dt=1e-3, t=step * 1e-3, ctx=ctx
        )
        spikes_a.append(result_a.spikes.clone())
        spikes_b.append(result_b.spikes.clone())

    torch.testing.assert_close(state_a.v_soma, state_b.v_soma)
    torch.testing.assert_close(state_a.v_dend, state_b.v_dend)
    torch.testing.assert_close(state_a.w, state_b.w)
    torch.testing.assert_close(state_a.refrac_left, state_b.refrac_left)
    torch.testing.assert_close(state_a.spike_hold_left, state_b.spike_hold_left)
    for a, b in zip(spikes_a, spikes_b, strict=True):
        assert torch.equal(a, b)


def test_input_shape_validation_glif():
    model = GLIFModel()
    n = 3
    ctx = _ctx(validate_shapes=True)
    state = model.init_state(n, ctx=ctx)
    bad_drive = torch.zeros((n, 1), device=state.v_soma.device, dtype=state.v_soma.dtype)

    with pytest.raises(ValueError):
        _step_glif(model, state, bad_drive, dt=1e-3, t=0.0, ctx=ctx)


def test_input_shape_validation_adex():
    model = AdEx2CompModel()
    n = 3
    ctx = _ctx(validate_shapes=True)
    state = model.init_state(n, ctx=ctx)
    drive_s = _drive(0.0, n, state.v_soma.device, state.v_soma.dtype)
    bad_drive_d = torch.zeros((n + 1,), device=state.v_dend.device, dtype=state.v_dend.dtype)

    with pytest.raises(ValueError):
        _step_adex(model, state, drive_s, bad_drive_d, dt=1e-3, t=0.0, ctx=ctx)


def test_adex_dendrite_soma_coupling():
    model = AdEx2CompModel()
    n = 3
    ctx = _ctx()
    state = model.init_state(n, ctx=ctx)
    drive_s = _drive(0.0, n, state.v_soma.device, state.v_soma.dtype)
    drive_d = _drive(2e-8, n, state.v_dend.device, state.v_dend.dtype)

    v_soma_before = state.v_soma.clone()
    state, _ = _step_adex(model, state, drive_s, drive_d, dt=1e-3, t=0.0, ctx=ctx)

    assert bool(torch.all(state.v_soma > v_soma_before))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_glif_cuda_smoke():
    model = GLIFModel()
    n = 4
    ctx = _ctx(device="cuda", dtype="float32")
    state = model.init_state(n, ctx=ctx)
    drive = _drive(1e-9, n, state.v_soma.device, state.v_soma.dtype)
    state, result = _step_glif(model, state, drive, dt=1e-3, t=0.0, ctx=ctx)

    assert state.v_soma.is_cuda
    assert result.spikes.is_cuda
    assert result.membrane is not None
    assert result.membrane[Compartment.SOMA].is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_adex_cuda_smoke():
    model = AdEx2CompModel()
    n = 4
    ctx = _ctx(device="cuda", dtype="float32")
    state = model.init_state(n, ctx=ctx)
    drive_s = _drive(1e-9, n, state.v_soma.device, state.v_soma.dtype)
    drive_d = _drive(1e-9, n, state.v_dend.device, state.v_dend.dtype)
    state, result = _step_adex(model, state, drive_s, drive_d, dt=1e-3, t=0.0, ctx=ctx)

    assert state.v_soma.is_cuda
    assert state.v_dend.is_cuda
    assert result.spikes.is_cuda
    assert result.membrane is not None
    assert result.membrane[Compartment.DENDRITE].is_cuda
