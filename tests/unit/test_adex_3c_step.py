from __future__ import annotations

import pytest

from biosnn.biophysics.models.adex_3c import AdEx3CompModel, AdEx3CompParams
from biosnn.contracts.neurons import Compartment, NeuronInputs, StepContext

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_adex_3c_step_shapes_and_membrane_outputs() -> None:
    model = AdEx3CompModel()
    n = 4
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(n, ctx=ctx)

    drive = {
        Compartment.SOMA: torch.zeros(n, device=state.v_soma.device, dtype=state.v_soma.dtype),
        Compartment.DENDRITE: torch.zeros(n, device=state.v_dend.device, dtype=state.v_dend.dtype),
        Compartment.AXON: torch.zeros(n, device=state.v_axon.device, dtype=state.v_axon.dtype),
    }
    state, result = model.step(state, NeuronInputs(drive=drive), dt=1e-3, t=0.0, ctx=ctx)

    assert result.spikes.shape == (n,)
    assert result.spikes.dtype is torch.bool
    assert result.membrane is not None
    assert result.membrane[Compartment.SOMA].shape == (n,)
    assert result.membrane[Compartment.DENDRITE].shape == (n,)
    assert result.membrane[Compartment.AXON].shape == (n,)

    tensors = model.state_tensors(state)
    assert tensors["v_soma"].shape == (n,)
    assert tensors["v_dend"].shape == (n,)
    assert tensors["v_axon"].shape == (n,)
    assert tensors["w"].shape == (n,)
    assert tensors["w_soma"].shape == (n,)
    assert tensors["w_axon"].shape == (n,)


def test_adex_3c_dendrite_drive_couples_and_spikes() -> None:
    params = AdEx3CompParams(
        c_s=1.0,
        g_l_s=1.0,
        e_l_s=0.0,
        c_d=1.0,
        g_l_d=1.0,
        e_l_d=0.0,
        c_a=1.0,
        g_l_a=1.0,
        e_l_a=0.0,
        g_c_ds=2.0,
        g_c_sa=1.5,
        v_t_s=0.5,
        delta_t_s=0.0,
        v_t_a=0.5,
        delta_t_a=0.0,
        v_reset=0.0,
        v_spike=0.15,
        a_s=0.0,
        b_s=0.0,
        tau_w_s=1.0,
        a_a=0.0,
        b_a=0.0,
        tau_w_a=1.0,
        refrac_period=0.0,
        spike_hold_time=0.0,
        spike_source=Compartment.SOMA,
        reset_axon_on_spike=True,
    )
    model = AdEx3CompModel(params)
    n = 4
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(n, ctx=ctx)

    drive = {
        Compartment.SOMA: torch.zeros(n, device=state.v_soma.device, dtype=state.v_soma.dtype),
        Compartment.DENDRITE: torch.full((n,), 1.2, device=state.v_dend.device, dtype=state.v_dend.dtype),
        Compartment.AXON: torch.zeros(n, device=state.v_axon.device, dtype=state.v_axon.dtype),
    }

    soma_start = state.v_soma.clone()
    soma_peak = soma_start.clone()
    spike_seen = False
    for step in range(16):
        state, result = model.step(
            state,
            NeuronInputs(drive=drive),
            dt=1e-1,
            t=step * 1e-1,
            ctx=ctx,
        )
        soma_peak = torch.maximum(soma_peak, state.v_soma)
        spike_seen = spike_seen or bool(result.spikes.any())

    assert bool(torch.all(soma_peak > soma_start))
    assert spike_seen


def test_adex_3c_axon_spike_source_works() -> None:
    params = AdEx3CompParams(
        c_s=1.0,
        g_l_s=1.0,
        e_l_s=0.0,
        c_d=1.0,
        g_l_d=1.0,
        e_l_d=0.0,
        c_a=1.0,
        g_l_a=1.0,
        e_l_a=0.0,
        g_c_ds=0.0,
        g_c_sa=0.0,
        v_t_s=0.5,
        delta_t_s=0.0,
        v_t_a=0.5,
        delta_t_a=0.0,
        v_reset=0.0,
        v_spike=0.6,
        a_s=0.0,
        b_s=0.0,
        tau_w_s=1.0,
        a_a=0.0,
        b_a=0.0,
        tau_w_a=1.0,
        refrac_period=0.0,
        spike_hold_time=0.0,
        spike_source=Compartment.AXON,
    )
    model = AdEx3CompModel(params)
    n = 2
    ctx = StepContext(device="cpu", dtype="float32")
    state = model.init_state(n, ctx=ctx)

    drive = {
        Compartment.SOMA: torch.zeros(n, device=state.v_soma.device, dtype=state.v_soma.dtype),
        Compartment.DENDRITE: torch.zeros(n, device=state.v_dend.device, dtype=state.v_dend.dtype),
        Compartment.AXON: torch.full((n,), 1.0, device=state.v_axon.device, dtype=state.v_axon.dtype),
    }

    spike_seen = False
    for step in range(16):
        state, result = model.step(
            state,
            NeuronInputs(drive=drive),
            dt=1e-1,
            t=step * 1e-1,
            ctx=ctx,
        )
        spike_seen = spike_seen or bool(result.spikes.any())
    assert spike_seen
