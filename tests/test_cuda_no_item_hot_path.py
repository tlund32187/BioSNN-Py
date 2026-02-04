from __future__ import annotations

from dataclasses import dataclass

import pytest

from biosnn.biophysics.models._torch_utils import resolve_device_dtype
from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.tensor import Tensor
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    dummy: Tensor


class SilentNeuronModel(INeuronModel):
    name = "silent"
    compartments = frozenset({Compartment.SOMA})

    def init_state(self, n: int, *, ctx: StepContext) -> DummyState:
        device, dtype = resolve_device_dtype(ctx)
        return DummyState(dummy=torch.zeros((n,), device=device, dtype=dtype))

    def reset_state(
        self,
        state: DummyState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> DummyState:
        if indices is None:
            state.dummy.zero_()
        else:
            state.dummy[indices] = 0.0
        return state

    def step(
        self,
        state: DummyState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[DummyState, NeuronStepResult]:
        spikes = torch.zeros_like(state.dummy)
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: DummyState):
        return {"dummy": state.dummy}


def test_cuda_step_no_item(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    calls = {"count": 0}
    original_item = torch.Tensor.item

    def _wrapped(self, *args, **kwargs):
        calls["count"] += 1
        return original_item(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", _wrapped, raising=True)

    pop = PopulationSpec(name="A", model=SilentNeuronModel(), n=4)
    engine = TorchNetworkEngine(populations=[pop], projections=[], fast_mode=True)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cuda"))
    for _ in range(3):
        engine.step()

    assert calls["count"] == 0
