from __future__ import annotations

from dataclasses import dataclass

import pytest

from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import resolve_device_dtype
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


class CaptureMonitor:
    name = "capture"

    def __init__(self) -> None:
        self.event = None

    def on_step(self, event):
        self.event = event

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


def test_fast_mode_skips_merge(monkeypatch):
    from biosnn.simulation.engine import torch_network_engine as tne

    def _fail(*args, **kwargs):
        raise AssertionError("_merge_population_tensors should not be called in fast_mode")

    monkeypatch.setattr(tne, "_merge_population_tensors", _fail)

    pop = PopulationSpec(name="A", model=SilentNeuronModel(), n=2)
    engine = TorchNetworkEngine(populations=[pop], projections=[], fast_mode=True)
    monitor = CaptureMonitor()
    engine.attach_monitors([monitor])
    engine.reset(config=SimulationConfig(dt=1e-3))
    engine.step()

    assert monitor.event is not None
    assert monitor.event.spikes is None
    assert "pop/A/spikes" in monitor.event.tensors
    assert monitor.event.tensors["pop/A/spikes"].dtype == torch.bool
    assert "pop/A/dummy" in monitor.event.tensors


def test_fast_mode_false_uses_merge(monkeypatch):
    from biosnn.simulation.engine import torch_network_engine as tne

    called = {"count": 0}

    def _fake(*args, **kwargs):
        called["count"] += 1
        return {}

    monkeypatch.setattr(tne, "_merge_population_tensors", _fake)

    pop = PopulationSpec(name="A", model=SilentNeuronModel(), n=2)
    engine = TorchNetworkEngine(populations=[pop], projections=[], fast_mode=False)
    monitor = CaptureMonitor()
    engine.attach_monitors([monitor])
    engine.reset(config=SimulationConfig(dt=1e-3))
    engine.step()

    assert called["count"] == 1
    assert monitor.event is not None
    assert monitor.event.spikes is not None
