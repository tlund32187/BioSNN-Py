from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import resolve_device_dtype
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import ModulatorSpec, PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse
from tests.support.test_models import SpikeInputModel

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class _FieldState:
    scalar: Tensor


class _PositionField:
    name = "position_field"

    def init_state(self, *, ctx: StepContext) -> _FieldState:
        device, dtype = resolve_device_dtype(ctx)
        return _FieldState(scalar=torch.zeros((), device=device, dtype=dtype))

    def step(
        self,
        state: _FieldState,
        *,
        releases: Sequence[ModulatorRelease],
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> _FieldState:
        _ = releases, dt, t, ctx
        return state

    def sample_at(
        self,
        state: _FieldState,
        *,
        positions: Tensor,
        kind: ModulatorKind,
        ctx: StepContext,
    ) -> Tensor:
        _ = state, kind, ctx
        return positions[:, 0]

    def state_tensors(self, state: _FieldState) -> Mapping[str, Tensor]:
        _ = state
        return {}


@dataclass(slots=True)
class _RuleState:
    pass


class _CaptureModulatorRule(ILearningRule):
    name = "capture_modulator"
    supports_sparse = False

    def __init__(self) -> None:
        self.last_modulators: Mapping[ModulatorKind, Tensor] | None = None

    def init_state(self, e: int, *, ctx: StepContext) -> _RuleState:
        _ = e, ctx
        return _RuleState()

    def step(
        self,
        state: _RuleState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[_RuleState, LearningStepResult]:
        _ = dt, t, ctx
        self.last_modulators = batch.modulators
        return state, LearningStepResult(d_weights=torch.zeros_like(batch.weights))

    def state_tensors(self, state: _RuleState) -> Mapping[str, Tensor]:
        _ = state
        return {}


def test_compiled_learning_receives_edge_sampled_modulators() -> None:
    n_pre = 2
    engine, learning_rule = _build_compiled_engine(n_pre=n_pre)

    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu", dtype="float32"))
    cache = engine._compiled_edge_mods_by_proj["P"]
    edge_key = engine._modulator_subsystem.EDGE_POSITIONS_KEY
    assert edge_key in cache
    edge_positions = cache[edge_key]
    assert edge_positions.shape == (n_pre, 3)
    torch.testing.assert_close(edge_positions[:, 0], torch.full((n_pre,), 10.0))

    engine.step()
    mods = learning_rule.last_modulators
    assert mods is not None
    assert all(isinstance(kind, ModulatorKind) for kind in mods)
    assert ModulatorKind.DOPAMINE in mods
    torch.testing.assert_close(mods[ModulatorKind.DOPAMINE], torch.full((n_pre,), 10.0))


def test_compiled_edge_cache_supports_monitor_sample_cap() -> None:
    n_pre = 8
    engine, _ = _build_compiled_engine(n_pre=n_pre)

    engine.reset(
        config=SimulationConfig(
            dt=1e-3,
            device="cpu",
            dtype="float32",
            max_edges_mod_sample=3,
        )
    )
    cache = engine._compiled_edge_mods_by_proj["P"]
    sample_key = engine._modulator_subsystem.EDGE_MONITOR_SAMPLE_IDX_KEY
    assert sample_key in cache
    sample_idx = cache[sample_key]
    assert sample_idx.dtype == torch.long
    assert int(sample_idx.numel()) == 3
    assert int(sample_idx.max().item()) < n_pre


def _build_compiled_engine(*, n_pre: int) -> tuple[TorchNetworkEngine, _CaptureModulatorRule]:
    pre_positions = torch.stack(
        (
            torch.arange(n_pre, dtype=torch.float32),
            torch.zeros((n_pre,), dtype=torch.float32),
            torch.zeros((n_pre,), dtype=torch.float32),
        ),
        dim=1,
    )
    post_positions = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)

    pop_pre = PopulationSpec(name="Pre", model=SpikeInputModel(), n=n_pre, positions=pre_positions)
    pop_post = PopulationSpec(name="Post", model=SpikeInputModel(), n=1, positions=post_positions)

    pre_idx = torch.arange(n_pre, dtype=torch.long)
    post_idx = torch.zeros((n_pre,), dtype=torch.long)
    weights = torch.zeros((n_pre,), dtype=torch.float32)
    topology = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, weights=weights)

    learning_rule = _CaptureModulatorRule()
    projection = ProjectionSpec(
        name="P",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.0)),
        topology=topology,
        pre="Pre",
        post="Post",
        learning=learning_rule,
        sparse_learning=False,
    )

    mod_spec = ModulatorSpec(
        name="dopamine",
        field=_PositionField(),
        kinds=(ModulatorKind.DOPAMINE,),
    )

    def external_drive_fn(
        t: float,
        step: int,
        pop_name: str,
        ctx: StepContext,
    ) -> Mapping[Compartment, Tensor]:
        _ = t
        if step != 0:
            return {}
        device, dtype = resolve_device_dtype(ctx)
        if pop_name == "Pre":
            return {Compartment.SOMA: torch.ones((n_pre,), device=device, dtype=dtype)}
        if pop_name == "Post":
            return {Compartment.SOMA: torch.ones((1,), device=device, dtype=dtype)}
        return {}

    engine = TorchNetworkEngine(
        populations=[pop_pre, pop_post],
        projections=[projection],
        modulators=[mod_spec],
        external_drive_fn=external_drive_fn,
        compiled_mode=True,
        fast_mode=True,
    )
    return engine, learning_rule
