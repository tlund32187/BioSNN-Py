import pytest

from biosnn.biophysics.models.lif_3c import LIF3CompModel, LIF3CompParams
from biosnn.contracts.neurons import Compartment, NeuronInputs, StepContext

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_lif_3c_step_shapes_and_membrane_outputs() -> None:
    model = LIF3CompModel()
    n = 4
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(n, ctx=ctx)

    drive = {
        Compartment.SOMA: torch.zeros(n, device=state.v.device, dtype=state.v.dtype),
        Compartment.DENDRITE: torch.zeros(n, device=state.v.device, dtype=state.v.dtype),
        Compartment.AXON: torch.zeros(n, device=state.v.device, dtype=state.v.dtype),
    }
    state, result = model.step(state, NeuronInputs(drive=drive), dt=1e-3, t=0.0, ctx=ctx)

    assert state.v.shape == (n, 3)
    assert result.spikes.shape == (n,)
    assert result.spikes.dtype is torch.bool
    assert result.membrane is not None
    assert result.membrane[Compartment.SOMA].shape == (n,)
    assert result.membrane[Compartment.DENDRITE].shape == (n,)
    assert result.membrane[Compartment.AXON].shape == (n,)


def test_lif_3c_dendrite_drive_couples_and_spikes() -> None:
    params = LIF3CompParams(
        v_rest=0.0,
        v_reset=0.0,
        v_thresh=0.5,
        tau_m_soma=0.02,
        tau_m_dend=0.02,
        tau_m_axon=0.02,
        axial_g_dend_soma=2.0,
        axial_g_soma_axon=0.5,
        refrac_period=0.0,
        spike_hold_time=0.0,
        reset_axon_on_spike=True,
    )
    model = LIF3CompModel(params)
    n = 4
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(n, ctx=ctx)

    drive = {
        Compartment.SOMA: torch.zeros(n, device=state.v.device, dtype=state.v.dtype),
        Compartment.DENDRITE: torch.full((n,), 1.0, device=state.v.device, dtype=state.v.dtype),
        Compartment.AXON: torch.zeros(n, device=state.v.device, dtype=state.v.dtype),
    }

    soma_start = state.v[:, 0].clone()
    soma_peak = soma_start.clone()
    spike_seen = False
    for step in range(8):
        state, result = model.step(
            state,
            NeuronInputs(drive=drive),
            dt=1e-2,
            t=step * 1e-2,
            ctx=ctx,
        )
        soma_peak = torch.maximum(soma_peak, state.v[:, 0])
        spike_seen = spike_seen or bool(result.spikes.any())

    assert bool(torch.all(soma_peak > soma_start))
    assert spike_seen

