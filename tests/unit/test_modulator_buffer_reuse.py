"""Verify that step_compiled() reuses preallocated buffers across steps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pytest

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.neurons import StepContext
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import resolve_device_dtype
from biosnn.simulation.engine.subsystems.modulators import (
    CompiledEdgeModCache,
    ModulatorSubsystem,
)
from biosnn.simulation.network import ModulatorSpec, PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _FieldState:
    scalar: Tensor


class _ConstantField:
    """Returns a constant value at every position for testing."""

    name = "constant_field"

    def __init__(self, value: float = 1.0) -> None:
        self._value = value

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
        return cast(
            Tensor,
            torch.full(
                (positions.shape[0],), self._value, device=positions.device, dtype=positions.dtype
            ),
        )

    def state_tensors(self, state: _FieldState) -> Mapping[str, Tensor]:
        _ = state
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_fixtures(
    *, n_pre: int = 4, value: float = 2.0
) -> tuple[
    ModulatorSubsystem,
    list[ModulatorSpec],
    dict[str, _FieldState],
    list[PopulationSpec],
    list[ProjectionSpec],
    Mapping[str, PopulationSpec],
    tuple[ModulatorKind, ...],
]:
    pre_positions = torch.stack(
        (
            torch.arange(n_pre, dtype=torch.float32),
            torch.zeros(n_pre, dtype=torch.float32),
            torch.zeros(n_pre, dtype=torch.float32),
        ),
        dim=1,
    )
    post_positions = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)

    pop_pre = PopulationSpec(name="Pre", model=None, n=n_pre, positions=pre_positions)  # type: ignore[arg-type]
    pop_post = PopulationSpec(name="Post", model=None, n=1, positions=post_positions)  # type: ignore[arg-type]

    pre_idx = torch.arange(n_pre, dtype=torch.long)
    post_idx = torch.zeros(n_pre, dtype=torch.long)
    weights = torch.zeros(n_pre, dtype=torch.float32)
    topology = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, weights=weights)

    projection = ProjectionSpec(
        name="P",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.0)),
        topology=topology,
        pre="Pre",
        post="Post",
    )

    field = _ConstantField(value=value)
    mod_spec = ModulatorSpec(
        name="dopamine",
        field=field,
        kinds=(ModulatorKind.DOPAMINE,),
    )

    ctx = StepContext(device="cpu", dtype="float32")
    mod_states: dict[str, _FieldState] = {"dopamine": field.init_state(ctx=ctx)}

    pop_specs = [pop_pre, pop_post]
    proj_specs = [projection]
    pop_map = {p.name: p for p in pop_specs}
    kinds = (ModulatorKind.DOPAMINE,)

    subsystem = ModulatorSubsystem()
    return subsystem, [mod_spec], mod_states, pop_specs, proj_specs, pop_map, kinds


def _run_compiled_step(
    subsystem: ModulatorSubsystem,
    mod_specs: list[ModulatorSpec],
    mod_states: dict[str, _FieldState],
    pop_specs: list[PopulationSpec],
    proj_specs: list[ProjectionSpec],
    pop_map: Mapping[str, PopulationSpec],
    kinds: tuple[ModulatorKind, ...],
    mod_by_pop: dict[str, dict[ModulatorKind, Tensor]],
    edge_mods_by_proj: dict[str, CompiledEdgeModCache],
    *,
    step_idx: int = 0,
) -> tuple[dict[str, dict[ModulatorKind, Tensor]], dict[str, CompiledEdgeModCache]]:
    ctx = StepContext(device="cpu", dtype="float32")
    return subsystem.step_compiled(
        mod_specs=mod_specs,
        mod_states=mod_states,
        pop_specs=pop_specs,
        proj_specs=proj_specs,
        t=step_idx * 1e-3,
        step=step_idx,
        dt=1e-3,
        ctx=ctx,
        device="cpu",
        dtype=torch.float32,
        kinds=kinds,
        pop_map=pop_map,
        mod_by_pop=mod_by_pop,
        edge_mods_by_proj=edge_mods_by_proj,
        releases=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compiled_step_reuses_pop_buffers() -> None:
    """After 2 compiled steps the per-pop per-kind tensors are the same objects."""
    subsystem, mod_specs, mod_states, pop_specs, proj_specs, pop_map, kinds = _build_test_fixtures()
    mod_by_pop: dict[str, dict[ModulatorKind, Tensor]] = {}
    edge_mods_by_proj: dict[str, CompiledEdgeModCache] = {}

    # Step 1 — allocates buffers.
    _run_compiled_step(
        subsystem,
        mod_specs,
        mod_states,
        pop_specs,
        proj_specs,
        pop_map,
        kinds,
        mod_by_pop,
        edge_mods_by_proj,
        step_idx=0,
    )

    # Capture tensor ids after step 1.
    pop_buf_ids: dict[str, dict[ModulatorKind, int]] = {}
    for name, pop_dict in mod_by_pop.items():
        pop_buf_ids[name] = {kind: id(tensor) for kind, tensor in pop_dict.items()}

    # Step 2 — must reuse the same tensor objects.
    _run_compiled_step(
        subsystem,
        mod_specs,
        mod_states,
        pop_specs,
        proj_specs,
        pop_map,
        kinds,
        mod_by_pop,
        edge_mods_by_proj,
        step_idx=1,
    )

    for name, pop_dict in mod_by_pop.items():
        for kind, tensor in pop_dict.items():
            assert id(tensor) == pop_buf_ids[name][kind], (
                f"Pop buffer for {name}/{kind} was reallocated instead of reused"
            )


def test_compiled_step_reuses_edge_buffers() -> None:
    """After 2 compiled steps the per-proj per-kind tensors are the same objects."""
    subsystem, mod_specs, mod_states, pop_specs, proj_specs, pop_map, kinds = _build_test_fixtures()
    mod_by_pop: dict[str, dict[ModulatorKind, Tensor]] = {}
    edge_mods_by_proj: dict[str, CompiledEdgeModCache] = {}

    # Step 1.
    _run_compiled_step(
        subsystem,
        mod_specs,
        mod_states,
        pop_specs,
        proj_specs,
        pop_map,
        kinds,
        mod_by_pop,
        edge_mods_by_proj,
        step_idx=0,
    )

    # Capture tensor ids for ModulatorKind entries only (skip string keys).
    edge_buf_ids: dict[str, dict[ModulatorKind, int]] = {}
    for name, proj_dict in edge_mods_by_proj.items():
        edge_buf_ids[name] = {
            kind: id(tensor)
            for kind, tensor in proj_dict.items()
            if isinstance(kind, ModulatorKind)
        }

    # Step 2.
    _run_compiled_step(
        subsystem,
        mod_specs,
        mod_states,
        pop_specs,
        proj_specs,
        pop_map,
        kinds,
        mod_by_pop,
        edge_mods_by_proj,
        step_idx=1,
    )

    for name, proj_dict in edge_mods_by_proj.items():
        for kind, tensor in proj_dict.items():
            if not isinstance(kind, ModulatorKind):
                continue
            assert id(tensor) == edge_buf_ids[name][kind], (
                f"Edge buffer for {name}/{kind} was reallocated instead of reused"
            )


def test_compiled_step_values_are_correct_after_reuse() -> None:
    """Buffers contain the right accumulated modulator value after each step."""
    value = 3.5
    n_pre = 4
    subsystem, mod_specs, mod_states, pop_specs, proj_specs, pop_map, kinds = _build_test_fixtures(
        n_pre=n_pre, value=value
    )
    mod_by_pop: dict[str, dict[ModulatorKind, Tensor]] = {}
    edge_mods_by_proj: dict[str, CompiledEdgeModCache] = {}

    for step_idx in range(3):
        _run_compiled_step(
            subsystem,
            mod_specs,
            mod_states,
            pop_specs,
            proj_specs,
            pop_map,
            kinds,
            mod_by_pop,
            edge_mods_by_proj,
            step_idx=step_idx,
        )

        # Population-level: every neuron in Pre and Post should get `value`.
        for pop_name, pop_dict in mod_by_pop.items():
            for kind, tensor in pop_dict.items():
                expected_size = next(p.n for p in pop_specs if p.name == pop_name)
                assert tensor.shape == (expected_size,)
                label = f"step {step_idx}, pop {pop_name}, kind {kind}"
                torch.testing.assert_close(
                    tensor,
                    torch.full_like(tensor, value),
                    msg=label,
                )

        # Edge-level: every edge should get `value`.
        for proj_name, proj_dict in edge_mods_by_proj.items():
            for edge_key, tensor in proj_dict.items():
                if not isinstance(edge_key, ModulatorKind):
                    continue
                assert tensor.shape == (n_pre,)
                label = f"step {step_idx}, proj {proj_name}, kind {edge_key}"
                torch.testing.assert_close(
                    tensor,
                    torch.full_like(tensor, value),
                    msg=label,
                )
