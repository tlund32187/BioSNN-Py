"""Neuromodulator stepping subsystem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.neurons import StepContext
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.network.specs import ModulatorSpec, PopulationSpec, ProjectionSpec

_EDGE_POSITIONS_KEY = "__edge_positions__"
_EDGE_MONITOR_SAMPLE_IDX_KEY = "__edge_monitor_sample_idx__"

CompiledEdgeModCache = dict[ModulatorKind | str, Tensor]


class ModulatorSubsystem:
    """Owns modulator field stepping and per-edge sampling."""

    EDGE_POSITIONS_KEY = _EDGE_POSITIONS_KEY
    EDGE_MONITOR_SAMPLE_IDX_KEY = _EDGE_MONITOR_SAMPLE_IDX_KEY

    def collect_modulator_kinds(self, mod_specs: Sequence[ModulatorSpec]) -> tuple[ModulatorKind, ...]:
        kinds: list[ModulatorKind] = []
        for spec in mod_specs:
            for kind in spec.kinds:
                if kind not in kinds:
                    kinds.append(kind)
        return tuple(kinds)

    def prepare_compiled_edge_cache(
        self,
        *,
        proj_specs: Sequence[ProjectionSpec],
        pop_map: Mapping[str, PopulationSpec],
        edge_mods_by_proj: dict[str, CompiledEdgeModCache],
        device: Any,
        dtype: Any,
        max_edges_mod_sample: int | None = None,
    ) -> None:
        for proj in proj_specs:
            proj_cache = edge_mods_by_proj.get(proj.name)
            if proj_cache is None:
                proj_cache = {}
                edge_mods_by_proj[proj.name] = proj_cache
            edge_positions = self._edge_positions_for_projection(
                proj=proj,
                pop_map=pop_map,
                device=device,
                dtype=dtype,
            )
            proj_cache[self.EDGE_POSITIONS_KEY] = edge_positions
            if (
                max_edges_mod_sample is not None
                and max_edges_mod_sample > 0
                and edge_positions.shape[0] > max_edges_mod_sample
            ):
                proj_cache[self.EDGE_MONITOR_SAMPLE_IDX_KEY] = self._build_monitor_edge_sample(
                    edge_count=edge_positions.shape[0],
                    sample_size=max_edges_mod_sample,
                    device=device,
                )
            else:
                proj_cache.pop(self.EDGE_MONITOR_SAMPLE_IDX_KEY, None)

    def learning_modulators_for_projection(
        self,
        cache: Mapping[ModulatorKind, Tensor] | Mapping[ModulatorKind | str, Tensor] | None,
    ) -> Mapping[ModulatorKind, Tensor] | None:
        if cache is None:
            return None
        mods = {
            kind: value
            for kind, value in cache.items()
            if isinstance(kind, ModulatorKind)
        }
        return mods or None

    def get_population_levels(
        self,
        *,
        kind: ModulatorKind,
        population_name: str,
        mod_by_pop: Mapping[str, Mapping[ModulatorKind, Tensor]] | None,
        like: Tensor | None = None,
    ) -> Tensor | None:
        if mod_by_pop is None:
            return None
        pop_levels = mod_by_pop.get(population_name)
        if pop_levels is None:
            return None
        levels = pop_levels.get(kind)
        if levels is None:
            return None
        if like is not None and (levels.device != like.device or levels.dtype != like.dtype):
            levels = levels.to(device=like.device, dtype=like.dtype)
        return levels

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
        edge_mods_by_proj: dict[str, CompiledEdgeModCache],
        releases: Sequence[ModulatorRelease],
    ) -> tuple[dict[str, dict[ModulatorKind, Tensor]], dict[str, CompiledEdgeModCache]]:
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
            edge_positions = proj_dict.get(self.EDGE_POSITIONS_KEY)
            if not isinstance(edge_positions, torch.Tensor):
                edge_positions = self._edge_positions_for_projection(
                    proj=proj,
                    pop_map=pop_map,
                    device=device,
                    dtype=dtype,
                )
                proj_dict[self.EDGE_POSITIONS_KEY] = edge_positions
            for key in tuple(proj_dict.keys()):
                if isinstance(key, ModulatorKind) and key not in kinds:
                    proj_dict.pop(key, None)
            edge_len = int(edge_positions.shape[0])
            for kind in kinds:
                total = torch.zeros((edge_len,), device=device, dtype=dtype)
                for spec in mod_specs:
                    if kind not in spec.kinds:
                        continue
                    state = mod_states[spec.name]
                    sampled = spec.field.sample_at(
                        state,
                        positions=edge_positions,
                        kind=kind,
                        ctx=ctx,
                    )
                    total = total + sampled
                proj_dict[kind] = total

        return mod_by_pop, edge_mods_by_proj

    def _build_monitor_edge_sample(self, *, edge_count: int, sample_size: int, device: Any) -> Tensor:
        torch = require_torch()
        if edge_count <= sample_size:
            return cast(Tensor, torch.arange(edge_count, device=device, dtype=torch.long))
        stride = max(edge_count // sample_size, 1)
        sample = torch.arange(0, edge_count, stride, device=device, dtype=torch.long)
        return cast(Tensor, sample[:sample_size])

    def _edge_positions_for_projection(
        self,
        *,
        proj: ProjectionSpec,
        pop_map: Mapping[str, PopulationSpec],
        device: Any,
        dtype: Any,
    ) -> Tensor:
        torch = require_torch()
        edge_len = self._edge_count(proj.topology)
        if edge_len <= 0:
            return cast(Tensor, torch.empty((0, 3), device=device, dtype=dtype))

        post_idx = proj.topology.post_idx
        edge_positions: Tensor | None = None
        topology_post_pos = proj.topology.post_pos
        if topology_post_pos is not None:
            if int(topology_post_pos.shape[0]) == edge_len:
                edge_positions = cast(Tensor, topology_post_pos)
            elif int(topology_post_pos.shape[0]) == int(pop_map[proj.post].n):
                edge_positions = cast(Tensor, topology_post_pos.index_select(0, post_idx))

        if edge_positions is None:
            post_positions = pop_map[proj.post].positions
            if post_positions is None:
                edge_positions = cast(Tensor, torch.zeros((edge_len, 3), device=device, dtype=dtype))
            else:
                edge_positions = cast(Tensor, post_positions.index_select(0, post_idx))

        if edge_positions.device != device or edge_positions.dtype != dtype:
            edge_positions = cast(Tensor, edge_positions.to(device=device, dtype=dtype))

        if edge_positions.ndim != 2 or edge_positions.shape[1] != 3:
            normalized = torch.zeros((edge_len, 3), device=device, dtype=dtype)
            if edge_positions.ndim == 2 and edge_positions.shape[1] > 0:
                width = min(3, int(edge_positions.shape[1]))
                normalized[:, :width] = edge_positions[:, :width]
            edge_positions = cast(Tensor, normalized)

        return cast(Tensor, edge_positions.contiguous())

    def _edge_count(self, topology: Any) -> int:
        if hasattr(topology.pre_idx, "numel"):
            return int(topology.pre_idx.numel())
        try:
            return len(topology.pre_idx)
        except TypeError:
            return 0


__all__ = ["ModulatorSubsystem"]
