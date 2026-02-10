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

pytestmark = pytest.mark.acceptance

torch = pytest.importorskip("torch")


def _run_single_spike_trace(
    *,
    profile: ReceptorProfile,
    device: str,
    steps: int = 10,
) -> list[float]:
    topology = SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long, device=device),
        post_idx=torch.tensor([0], dtype=torch.long, device=device),
        delay_steps=torch.tensor([0], dtype=torch.int32, device=device),
        weights=torch.tensor([1.0], dtype=torch.float32, device=device),
        target_compartment=Compartment.SOMA,
    )
    compiled = compile_topology(
        topology,
        device=device,
        dtype="float32",
        build_sparse_delay_mats=True,
    )
    model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            init_weight=1.0,
            receptor_profile=profile,
        )
    )
    ctx = StepContext(device=device, dtype="float32")
    state = model.init_state(compiled.pre_idx.numel(), ctx=ctx)
    if compiled.weights is not None:
        state.weights.copy_(compiled.weights)

    trace: list[float] = []
    for step in range(steps):
        spikes = torch.zeros((1,), device=device, dtype=state.weights.dtype)
        if step == 0:
            spikes[0] = 1.0
        state, result = model.step(
            state,
            compiled,
            SynapseInputs(pre_spikes=spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        value = result.post_drive[Compartment.SOMA].detach().cpu().reshape(-1)[0]
        trace.append(float(value))
    return trace


def test_multi_receptor_profile_shows_fast_decay_plus_slow_tail_cpu() -> None:
    ampa_only = ReceptorProfile(
        kinds=(ReceptorKind.AMPA,),
        mix={ReceptorKind.AMPA: 1.0},
        tau={ReceptorKind.AMPA: 2e-3},
        sign={ReceptorKind.AMPA: 1.0},
    )
    ampa_nmda = ReceptorProfile(
        kinds=(ReceptorKind.AMPA, ReceptorKind.NMDA),
        mix={ReceptorKind.AMPA: 1.0, ReceptorKind.NMDA: 1.0},
        tau={ReceptorKind.AMPA: 2e-3, ReceptorKind.NMDA: 30e-3},
        sign={ReceptorKind.AMPA: 1.0, ReceptorKind.NMDA: 1.0},
    )

    ampa_trace = _run_single_spike_trace(profile=ampa_only, device="cpu")
    multi_trace = _run_single_spike_trace(profile=ampa_nmda, device="cpu")

    assert multi_trace[0] > 0.0
    assert multi_trace[1] < multi_trace[0]
    assert multi_trace[5] > ampa_trace[5] * 3.0
    assert multi_trace[-1] > 0.0


@pytest.mark.cuda
def test_multi_receptor_profile_shows_fast_decay_plus_slow_tail_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    ampa_nmda = ReceptorProfile(
        kinds=(ReceptorKind.AMPA, ReceptorKind.NMDA),
        mix={ReceptorKind.AMPA: 1.0, ReceptorKind.NMDA: 1.0},
        tau={ReceptorKind.AMPA: 2e-3, ReceptorKind.NMDA: 30e-3},
        sign={ReceptorKind.AMPA: 1.0, ReceptorKind.NMDA: 1.0},
    )
    trace = _run_single_spike_trace(profile=ampa_nmda, device="cuda")
    assert trace[1] < trace[0]
    assert trace[-1] > 0.0
