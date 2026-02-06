from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import ISynapseModel, SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    build: Callable[[str | None], ISynapseModel]
    compile_kwargs: dict[str, bool]


_CASES = [
    _Case(
        name="sparse",
        build=lambda ring_dtype: DelayedSparseMatmulSynapse(
            DelayedSparseMatmulParams(init_weight=1.0, ring_dtype=ring_dtype)
        ),
        compile_kwargs={"build_sparse_delay_mats": True},
    ),
    _Case(
        name="current",
        build=lambda ring_dtype: DelayedCurrentSynapse(
            DelayedCurrentParams(init_weight=1.0, ring_dtype=ring_dtype)
        ),
        compile_kwargs={"build_edges_by_delay": True},
    ),
]


def _compile_topology(*, device: str, compile_kwargs: dict[str, bool]) -> SynapseTopology:
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    delay_steps = torch.tensor([2], dtype=torch.int32)
    weights = torch.tensor([1.0], dtype=torch.float32)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        weights=weights,
        target_compartment=Compartment.DENDRITE,
    )
    return compile_topology(topology, device=device, dtype="float32", **compile_kwargs)


def _run_impulse(case: _Case, *, device: str, ring_dtype: str | None) -> list[Tensor]:
    topology = _compile_topology(device=device, compile_kwargs=case.compile_kwargs)
    synapse = case.build(ring_dtype)
    ctx = StepContext(device=device, dtype="float32")
    state = synapse.init_state(topology.pre_idx.numel(), ctx=ctx)
    if topology.weights is not None:
        state.weights.copy_(topology.weights)

    traces: list[Tensor] = []
    for step in range(6):
        spikes = torch.zeros((1,), device=device, dtype=state.weights.dtype)
        if step == 0:
            spikes[0] = 1.0
        state, result = synapse.step(
            state,
            topology,
            SynapseInputs(pre_spikes=spikes),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        traces.append(result.post_drive[Compartment.DENDRITE].detach().cpu().clone())
    return traces


def _first_nonzero_step(traces: list[Tensor]) -> int | None:
    for idx, tensor in enumerate(traces):
        if torch.any(tensor != 0):
            return idx
    return None


def _assert_traces_close(
    baseline: list[Tensor],
    candidate: list[Tensor],
    *,
    atol: float,
) -> None:
    base_step = _first_nonzero_step(baseline)
    cand_step = _first_nonzero_step(candidate)
    assert base_step is not None, "baseline produced no drive"
    assert cand_step == base_step, "arrival step mismatch"
    for base, cand in zip(baseline, candidate, strict=True):
        torch.testing.assert_close(cand.to(dtype=base.dtype), base, rtol=0.0, atol=atol)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_ring_dtype_bfloat16_cpu_matches_baseline(case: _Case) -> None:
    baseline = _run_impulse(case, device="cpu", ring_dtype=None)
    try:
        candidate = _run_impulse(case, device="cpu", ring_dtype="bfloat16")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    _assert_traces_close(baseline, candidate, atol=1e-2)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_ring_dtype_float16_cuda_matches_baseline(case: _Case) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    baseline = _run_impulse(case, device="cuda", ring_dtype=None)
    candidate = _run_impulse(case, device="cuda", ring_dtype="float16")
    _assert_traces_close(baseline, candidate, atol=1e-3)
