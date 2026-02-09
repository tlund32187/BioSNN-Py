"""Neuromodulator stepping subsystem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.neurons import StepContext
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.network.specs import ModulatorSpec, PopulationSpec, ProjectionSpec


class ModulatorSubsystem:
    """Owns modulator field stepping and per-edge sampling."""

    def collect_modulator_kinds(self, mod_specs: Sequence[ModulatorSpec]) -> tuple[ModulatorKind, ...]:
        kinds: list[ModulatorKind] = []
        for spec in mod_specs:
            for kind in spec.kinds:
                if kind not in kinds:
                    kinds.append(kind)
        return tuple(kinds)

    def step(
        self,
        *,
        mod_specs: Sequence[ModulatorSpec],
        mod_states: dict[str, Any],
        pop_specs: Sequence[PopulationSpec],
        proj_specs: Sequence[ProjectionSpec],
        t: float,
        step: int,
        dt: float,
        ctx: StepContext,
        device: Any,
        dtype: Any,
        kinds: tuple[ModulatorKind, ...],
        releases: Sequence[ModulatorRelease],
    ) -> tuple[
        dict[str, dict[ModulatorKind, Tensor]] | None,
        dict[str, dict[ModulatorKind, Tensor]] | None,
    ]:
        _ = step
        if not mod_specs:
            return None, None

        torch = require_torch()
        mod_by_pop: dict[str, dict[ModulatorKind, Tensor]] = {}
        edge_mods_by_proj: dict[str, dict[ModulatorKind, Tensor]] = {}
        pop_map = {spec.name: spec for spec in pop_specs}

        for spec in mod_specs:
            state = mod_states[spec.name]
            rel = [r for r in releases if r.kind in spec.kinds]
            mod_states[spec.name] = spec.field.step(state, releases=rel, dt=dt, t=t, ctx=ctx)

        for pop in pop_specs:
            if pop.positions is None:
                continue
            for kind in kinds:
                total = torch.zeros((pop.n,), device=device, dtype=dtype)
                for spec in mod_specs:
                    if kind not in spec.kinds:
                        continue
                    state = mod_states[spec.name]
                    sampled = spec.field.sample_at(state, positions=pop.positions, kind=kind, ctx=ctx)
                    total = total + sampled
                mod_by_pop.setdefault(pop.name, {})[kind] = total

        for proj in proj_specs:
            post_pop = pop_map[proj.post]
            mods = mod_by_pop.get(post_pop.name)
            edge_len = self._edge_count(proj.topology)
            if mods is None:
                edge_mods_by_proj[proj.name] = {
                    kind: torch.zeros((edge_len,), device=device, dtype=dtype) for kind in kinds
                }
                continue
            edge_mods: dict[ModulatorKind, Tensor] = {}
            for kind in kinds:
                mod_post = mods.get(kind)
                if mod_post is None:
                    edge_mods[kind] = torch.zeros((edge_len,), device=device, dtype=dtype)
                else:
                    edge_mods[kind] = mod_post[proj.topology.post_idx]
            edge_mods_by_proj[proj.name] = edge_mods

        return mod_by_pop, edge_mods_by_proj

    def step_compiled(
        self,
        *,
        mod_specs: Sequence[ModulatorSpec],
        mod_states: dict[str, Any],
        pop_specs: Sequence[PopulationSpec],
        proj_specs: Sequence[ProjectionSpec],
        t: float,
        step: int,
        dt: float,
        ctx: StepContext,
        device: Any,
        dtype: Any,
        kinds: tuple[ModulatorKind, ...],
        pop_map: Mapping[str, PopulationSpec],
        mod_by_pop: dict[str, dict[ModulatorKind, Tensor]],
        edge_mods_by_proj: dict[str, dict[ModulatorKind, Tensor]],
        releases: Sequence[ModulatorRelease],
    ) -> tuple[dict[str, dict[ModulatorKind, Tensor]], dict[str, dict[ModulatorKind, Tensor]]]:
        _ = step
        if not mod_specs:
            return mod_by_pop, edge_mods_by_proj

        torch = require_torch()
        for spec in mod_specs:
            state = mod_states[spec.name]
            rel = [r for r in releases if r.kind in spec.kinds]
            mod_states[spec.name] = spec.field.step(state, releases=rel, dt=dt, t=t, ctx=ctx)

        for pop in pop_specs:
            pop_dict = mod_by_pop.get(pop.name)
            if pop_dict is None:
                pop_dict = {}
                mod_by_pop[pop.name] = pop_dict
            else:
                pop_dict.clear()
            if pop.positions is None:
                continue
            for kind in kinds:
                total = torch.zeros((pop.n,), device=device, dtype=dtype)
                for spec in mod_specs:
                    if kind not in spec.kinds:
                        continue
                    state = mod_states[spec.name]
                    sampled = spec.field.sample_at(state, positions=pop.positions, kind=kind, ctx=ctx)
                    total = total + sampled
                pop_dict[kind] = total

        for proj in proj_specs:
            proj_dict = edge_mods_by_proj.get(proj.name)
            if proj_dict is None:
                proj_dict = {}
                edge_mods_by_proj[proj.name] = proj_dict
            else:
                proj_dict.clear()
            post_pop = pop_map[proj.post]
            mods = mod_by_pop.get(post_pop.name)
            edge_len = self._edge_count(proj.topology)
            if mods is None:
                for kind in kinds:
                    proj_dict[kind] = torch.zeros((edge_len,), device=device, dtype=dtype)
                continue
            for kind in kinds:
                mod_post = mods.get(kind)
                if mod_post is None:
                    proj_dict[kind] = torch.zeros((edge_len,), device=device, dtype=dtype)
                else:
                    proj_dict[kind] = mod_post[proj.topology.post_idx]

        return mod_by_pop, edge_mods_by_proj

    def _edge_count(self, topology: Any) -> int:
        if hasattr(topology.pre_idx, "numel"):
            return int(topology.pre_idx.numel())
        try:
            return len(topology.pre_idx)
        except TypeError:
            return 0


__all__ = ["ModulatorSubsystem"]
