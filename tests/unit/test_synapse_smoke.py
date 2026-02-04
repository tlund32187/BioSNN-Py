import pytest

from biosnn.connectivity import DelayParams, compute_delay_steps
from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import ReceptorKind, SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics import DelayedCurrentParams, DelayedCurrentSynapse

torch = pytest.importorskip("torch")


def _ctx(device: str = "cpu", dtype: str = "float32", **extras: object) -> StepContext:
    return StepContext(device=device, dtype=dtype, extras=extras or None)


def _topology(pre_idx, post_idx, **kwargs):
    return SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, **kwargs)


def test_synapse_init_shapes():
    model = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.25))
    ctx = _ctx()
    state = model.init_state(3, ctx=ctx)

    assert state.weights.shape == (3,)
    assert state.weights.device.type == "cpu"
    assert state.weights.dtype == torch.float32
    assert state.weights.tolist() == [0.25, 0.25, 0.25]


def test_synapse_ring_buffer_delay():
    model = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1.0))
    ctx = _ctx()
    state = model.init_state(1, ctx=ctx)
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    delay_steps = torch.tensor([2], dtype=torch.int32)
    topology = _topology(pre_idx, post_idx, delay_steps=delay_steps, target_compartment=Compartment.SOMA)
    compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
    )

    spikes = [1.0, 0.0, 0.0, 0.0]
    outputs = []
    for step, val in enumerate(spikes):
        pre_spikes = torch.tensor([val], dtype=state.weights.dtype)
        state, result = model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=pre_spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        outputs.append(result.post_drive[Compartment.SOMA].item())

    assert outputs[0] == pytest.approx(0.0)
    assert outputs[1] == pytest.approx(0.0)
    assert outputs[2] == pytest.approx(1.0)


def test_synapse_receptor_scale_per_edge():
    model = DelayedCurrentSynapse(
        DelayedCurrentParams(receptor_scale={ReceptorKind.GABA: -1.0})
    )
    ctx = _ctx()
    state = model.init_state(2, ctx=ctx)
    state.weights.fill_(1.0)
    pre_idx = torch.tensor([0, 1], dtype=torch.long)
    post_idx = torch.tensor([0, 1], dtype=torch.long)
    receptor = torch.tensor([0, 1], dtype=torch.long)
    topology = _topology(
        pre_idx,
        post_idx,
        receptor=receptor,
        receptor_kinds=(ReceptorKind.AMPA, ReceptorKind.GABA),
        target_compartment=Compartment.SOMA,
    )
    compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
    )

    pre_spikes = torch.tensor([1.0, 1.0], dtype=state.weights.dtype)
    state, result = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    drive = result.post_drive[Compartment.SOMA]
    assert drive.shape == (2,)
    assert drive[0].item() == pytest.approx(1.0)
    assert drive[1].item() == pytest.approx(-1.0)


def test_synapse_target_compartments_per_edge():
    model = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1.0))
    ctx = _ctx()
    state = model.init_state(2, ctx=ctx)
    pre_idx = torch.tensor([0, 1], dtype=torch.long)
    post_idx = torch.tensor([0, 0], dtype=torch.long)
    target_compartments = torch.tensor([0, 1], dtype=torch.long)
    topology = _topology(
        pre_idx,
        post_idx,
        target_compartments=target_compartments,
    )
    compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
    )

    pre_spikes = torch.tensor([1.0, 1.0], dtype=state.weights.dtype)
    state, result = model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )

    assert result.post_drive[Compartment.SOMA].item() == pytest.approx(1.0)
    assert result.post_drive[Compartment.DENDRITE].item() == pytest.approx(1.0)


def test_synapse_input_shape_validation():
    model = DelayedCurrentSynapse()
    ctx = _ctx(validate_shapes=True)
    state = model.init_state(1, ctx=ctx)
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    topology = _topology(pre_idx, post_idx)
    compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
    )

    pre_spikes = torch.zeros((1, 1), dtype=state.weights.dtype)
    with pytest.raises(ValueError):
        model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=pre_spikes),
            dt=1e-3,
            t=0.0,
            ctx=ctx,
        )


def test_delay_steps_from_positions():
    pre_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    post_pos = torch.tensor([[0.0, 3.0, 0.0]])
    pre_idx = torch.tensor([0, 1], dtype=torch.long)
    post_idx = torch.tensor([0, 0], dtype=torch.long)
    myelin = torch.tensor([0.0, 1.0])
    params = DelayParams(base_velocity=1.0, myelin_scale=1.0, use_ceil=True)

    steps = compute_delay_steps(
        pre_pos,
        post_pos,
        pre_idx,
        post_idx,
        dt=1.0,
        myelin=myelin,
        params=params,
    )

    assert steps.dtype == torch.int32
    assert steps.tolist() == [3, 2]
