from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import ReceptorKind, SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)
from biosnn.synapses.receptors import ReceptorProfile

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _single_edge_topology(*, weight: float = 1.0) -> SynapseTopology:
    return SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long),
        post_idx=torch.tensor([0], dtype=torch.long),
        delay_steps=torch.tensor([0], dtype=torch.int32),
        weights=torch.tensor([weight], dtype=torch.float32),
        target_compartment=Compartment.SOMA,
    )


def _compile_single_edge(topology: SynapseTopology, *, build_pre_adjacency: bool) -> SynapseTopology:
    return compile_topology(
        topology,
        device="cpu",
        dtype="float32",
        build_sparse_delay_mats=not build_pre_adjacency,
        build_pre_adjacency=build_pre_adjacency,
    )


def test_conductance_reversal_potential_flips_current_sign() -> None:
    profile = ReceptorProfile(
        kinds=(ReceptorKind.AMPA,),
        mix={ReceptorKind.AMPA: 1.0},
        tau={ReceptorKind.AMPA: 5e-3},
        sign={ReceptorKind.AMPA: 1.0},
    )
    params = DelayedSparseMatmulParams(
        init_weight=1.0,
        receptor_profile=profile,
        conductance_mode=True,
        reversal_potential={ReceptorKind.AMPA: 0.0},
    )
    model = DelayedSparseMatmulSynapse(params)
    ctx = StepContext(device="cpu", dtype="float32")
    topology = _compile_single_edge(_single_edge_topology(weight=1.0), build_pre_adjacency=False)
    state = model.init_state(1, ctx=ctx)
    if topology.weights is not None:
        state.weights.copy_(topology.weights)

    spikes = torch.tensor([1.0], dtype=torch.float32)
    state, res_hyper = model.step(
        state,
        topology,
        SynapseInputs(
            pre_spikes=spikes,
            meta={"post_membrane": {Compartment.SOMA: torch.tensor([-0.070], dtype=torch.float32)}},
        ),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )
    current_hyper = float(res_hyper.post_drive[Compartment.SOMA][0].item())

    state, res_depolarized = model.step(
        state,
        topology,
        SynapseInputs(
            pre_spikes=spikes,
            meta={"post_membrane": {Compartment.SOMA: torch.tensor([0.020], dtype=torch.float32)}},
        ),
        dt=1e-3,
        t=1e-3,
        ctx=ctx,
    )
    current_depolarized = float(res_depolarized.post_drive[Compartment.SOMA][0].item())

    assert current_hyper > 0.0
    assert current_depolarized < 0.0


def test_nmda_mg_block_is_voltage_dependent() -> None:
    profile = ReceptorProfile(
        kinds=(ReceptorKind.NMDA,),
        mix={ReceptorKind.NMDA: 1.0},
        tau={ReceptorKind.NMDA: 30e-3},
        sign={ReceptorKind.NMDA: 1.0},
    )
    params = DelayedSparseMatmulParams(
        init_weight=1.0,
        receptor_profile=profile,
        conductance_mode=True,
        nmda_voltage_block=True,
        reversal_potential={ReceptorKind.NMDA: 0.0},
    )
    model = DelayedSparseMatmulSynapse(params)
    ctx = StepContext(device="cpu", dtype="float32")
    topology = _compile_single_edge(_single_edge_topology(weight=1.0), build_pre_adjacency=False)

    spikes = torch.tensor([1.0], dtype=torch.float32)

    state_low = model.init_state(1, ctx=ctx)
    if topology.weights is not None:
        state_low.weights.copy_(topology.weights)
    state_low, res_low = model.step(
        state_low,
        topology,
        SynapseInputs(
            pre_spikes=spikes,
            meta={"post_membrane": {Compartment.SOMA: torch.tensor([-0.070], dtype=torch.float32)}},
        ),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )
    current_low = float(res_low.post_drive[Compartment.SOMA][0].item())

    state_high = model.init_state(1, ctx=ctx)
    if topology.weights is not None:
        state_high.weights.copy_(topology.weights)
    state_high, res_high = model.step(
        state_high,
        topology,
        SynapseInputs(
            pre_spikes=spikes,
            meta={"post_membrane": {Compartment.SOMA: torch.tensor([-0.020], dtype=torch.float32)}},
        ),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )
    current_high = float(res_high.post_drive[Compartment.SOMA][0].item())

    assert current_low > 0.0
    assert current_high > current_low


def test_stp_enabled_forces_event_driven_compilation_requirements() -> None:
    model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            stp_enabled=True,
            backend="spmm_fused",
        )
    )
    reqs = model.compilation_requirements()
    assert reqs["needs_pre_adjacency"] is True
    assert reqs["needs_sparse_delay_mats"] is False


def test_stp_depresses_repeated_spike_responses() -> None:
    model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            init_weight=1.0,
            stp_enabled=True,
            stp_u=0.5,
            stp_tau_rec=0.2,
            stp_tau_facil=0.0,
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    topology = _compile_single_edge(_single_edge_topology(weight=1.0), build_pre_adjacency=True)
    state = model.init_state(1, ctx=ctx)
    if topology.weights is not None:
        state.weights.copy_(topology.weights)

    currents: list[float] = []
    for step in range(3):
        state, result = model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=torch.tensor([1.0], dtype=torch.float32)),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        currents.append(float(result.post_drive[Compartment.SOMA][0].item()))

    assert currents[1] < currents[0]
    assert currents[2] <= currents[1]


def test_conductance_mode_requires_post_membrane_meta() -> None:
    profile = ReceptorProfile(
        kinds=(ReceptorKind.AMPA,),
        mix={ReceptorKind.AMPA: 1.0},
        tau={ReceptorKind.AMPA: 5e-3},
        sign={ReceptorKind.AMPA: 1.0},
    )
    model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            init_weight=1.0,
            receptor_profile=profile,
            conductance_mode=True,
            reversal_potential={ReceptorKind.AMPA: 0.0},
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    topology = _compile_single_edge(_single_edge_topology(weight=1.0), build_pre_adjacency=False)
    state = model.init_state(1, ctx=ctx)
    if topology.weights is not None:
        state.weights.copy_(topology.weights)

    with pytest.raises(ValueError):
        model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=torch.tensor([1.0], dtype=torch.float32)),
            dt=1e-3,
            t=0.0,
            ctx=ctx,
        )
