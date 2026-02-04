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


def test_no_monitors_skips_event_payload(monkeypatch):
    from biosnn.simulation.engine import torch_network_engine as tne

    def _fail(*args, **kwargs):
        raise AssertionError("_merge_population_tensors should not be called without monitors")

    monkeypatch.setattr(tne, "_merge_population_tensors", _fail)

    pop = PopulationSpec(name="A", model=SilentNeuronModel(), n=4)
    engine = TorchNetworkEngine(populations=[pop], projections=[], fast_mode=False)
    engine.reset(config=SimulationConfig(dt=1e-3))

    for _ in range(10):
        engine.step()

    assert engine._last_event is not None
    assert engine._last_event.tensors in (None, {})


def test_no_monitors_skips_event_payload_compiled(monkeypatch):
    class _ExplodeDict(dict):
        def __setitem__(self, key, value):  # noqa: ANN001
            raise AssertionError("compiled event payload should not be touched without monitors")

        def __getitem__(self, key):  # noqa: ANN001
            raise AssertionError("compiled event payload should not be touched without monitors")

        def get(self, key, default=None):  # noqa: ANN001
            raise AssertionError("compiled event payload should not be touched without monitors")

    pop = PopulationSpec(name="A", model=SilentNeuronModel(), n=4)
    engine = TorchNetworkEngine(
        populations=[pop],
        projections=[],
        fast_mode=False,
        compiled_mode=True,
    )
    engine.reset(config=SimulationConfig(dt=1e-3))
    engine._compiled_event_tensors = _ExplodeDict()
    engine._compiled_pop_tensor_views = _ExplodeDict()

    for _ in range(5):
        engine.step()

    assert engine._last_event is not None
    assert engine._last_event.tensors in (None, {})
