from __future__ import annotations

from dataclasses import dataclass

import pytest

from biosnn.biophysics.models._torch_utils import resolve_device_dtype
from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.learning import ThreeFactorHebbianParams, ThreeFactorHebbianRule
from biosnn.neuromodulators import GlobalScalarField, GlobalScalarParams
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import ModulatorSpec, PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class DummyState:
    dummy: Tensor


class SpikingNeuronModel(INeuronModel):
    name = "spiking"
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
        spikes = torch.ones_like(state.dummy)
        return state, NeuronStepResult(spikes=spikes)

    def state_tensors(self, state: DummyState):
        return {}


def _release(amount: float) -> ModulatorRelease:
    return ModulatorRelease(
        kind=ModulatorKind.DOPAMINE,
        positions=torch.zeros((1, 3), dtype=torch.float32),
        amount=torch.tensor([amount], dtype=torch.float32),
    )


def test_network_engine_learning_updates_weights():
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    weights = torch.tensor([0.5], dtype=torch.float32)
    topology = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, weights=weights)

    pop_a = PopulationSpec(name="A", model=SpikingNeuronModel(), n=1, positions=torch.zeros((1, 3)))
    pop_b = PopulationSpec(name="B", model=SpikingNeuronModel(), n=1, positions=torch.zeros((1, 3)))

    learning = ThreeFactorHebbianRule(ThreeFactorHebbianParams(lr=0.1))
    synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.5))

    proj = ProjectionSpec(
        name="A_to_B",
        synapse=synapse,
        topology=topology,
        pre="A",
        post="B",
        learning=learning,
    )

    field = GlobalScalarField(kinds=(ModulatorKind.DOPAMINE,), params=GlobalScalarParams(decay_tau=10.0))
    mod = ModulatorSpec(name="da", field=field, kinds=(ModulatorKind.DOPAMINE,))

    def releases_fn(t: float, step: int, ctx: StepContext):
        return [_release(1.0)]

    engine = TorchNetworkEngine(
        populations=[pop_a, pop_b],
        projections=[proj],
        modulators=[mod],
        releases_fn=releases_fn,
    )
    engine.reset(config=SimulationConfig(dt=1e-3))

    initial = engine._proj_states["A_to_B"].state.weights.clone()
    engine.step()
    updated = engine._proj_states["A_to_B"].state.weights
    assert float(updated.mean().item()) > float(initial.mean().item())


def test_network_engine_learning_dopamine_gate():
    pre_idx = torch.tensor([0], dtype=torch.long)
    post_idx = torch.tensor([0], dtype=torch.long)
    weights = torch.tensor([0.5], dtype=torch.float32)
    topology = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, weights=weights)

    pop_a = PopulationSpec(name="A", model=SpikingNeuronModel(), n=1, positions=torch.zeros((1, 3)))
    pop_b = PopulationSpec(name="B", model=SpikingNeuronModel(), n=1, positions=torch.zeros((1, 3)))

    learning = ThreeFactorHebbianRule(ThreeFactorHebbianParams(lr=0.1))
    synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.5))

    proj = ProjectionSpec(
        name="A_to_B",
        synapse=synapse,
        topology=topology,
        pre="A",
        post="B",
        learning=learning,
    )

    field = GlobalScalarField(kinds=(ModulatorKind.DOPAMINE,), params=GlobalScalarParams(decay_tau=10.0))
    mod = ModulatorSpec(name="da", field=field, kinds=(ModulatorKind.DOPAMINE,))

    def releases_fn(t: float, step: int, ctx: StepContext):
        return [_release(0.0)]

    engine = TorchNetworkEngine(
        populations=[pop_a, pop_b],
        projections=[proj],
        modulators=[mod],
        releases_fn=releases_fn,
    )
    engine.reset(config=SimulationConfig(dt=1e-3))

    initial = engine._proj_states["A_to_B"].state.weights.clone()
    engine.step()
    updated = engine._proj_states["A_to_B"].state.weights
    assert float(updated.mean().item()) == pytest.approx(float(initial.mean().item()))
