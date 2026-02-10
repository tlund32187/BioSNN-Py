"""Online neurogenesis manager for sparse network growth."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.network.specs import PopulationSpec, ProjectionSpec


@dataclass(frozen=True, slots=True)
class NeurogenesisConfig:
    """Configuration for occasional growth events."""

    enabled: bool = False
    growth_interval_steps: int = 500
    add_neurons_per_event: int = 4
    newborn_plasticity_multiplier: float = 1.5
    newborn_duration_steps: int = 250
    max_total_neurons: int = 20000
    target_population_contains: str = "hidden"
    connectivity_p: float = 0.05
    activity_threshold: float = 0.10
    plateau_intervals: int = 2
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class NeurogenesisDecision:
    populations: tuple[PopulationSpec, ...]
    projections: tuple[ProjectionSpec, ...]
    target_population: str
    added_neurons: int
    added_edges: int


class NeurogenesisManager:
    """Tracks plateau signals and emits growth decisions."""

    def __init__(self, config: NeurogenesisConfig | None = None) -> None:
        self.config = config or NeurogenesisConfig()
        self._target_population: str | None = None
        self._activity_ema: float = 0.0
        self._interval_trigger_streak: int = 0
        self._events_total: int = 0
        self._last_scalars: dict[str, float] = {}

    def reset(
        self,
        populations: Sequence[PopulationSpec],
    ) -> None:
        self._target_population = self._resolve_target_population(populations)
        self._activity_ema = 0.0
        self._interval_trigger_streak = 0
        self._events_total = 0
        self._update_scalars(
            populations=populations,
            added_neurons=0,
            added_edges=0,
            trigger_score=0.0,
            grew=False,
        )

    def scalars(self) -> Mapping[str, float]:
        return self._last_scalars

    def on_structure_changed(self, populations: Sequence[PopulationSpec]) -> None:
        """Refresh target pointers after external topology/population edits."""
        new_target = self._resolve_target_population(populations)
        if new_target != self._target_population:
            self._target_population = new_target
            self._activity_ema = 0.0
            self._interval_trigger_streak = 0
        self._update_scalars(
            populations=populations,
            added_neurons=0,
            added_edges=0,
            trigger_score=0.0,
            grew=False,
        )

    def record_step(
        self,
        *,
        spikes_by_pop: Mapping[str, Tensor],
    ) -> None:
        if not self.config.enabled:
            return
        target = self._target_population
        if target is None:
            return
        spikes = spikes_by_pop.get(target)
        if spikes is None or spikes.numel() == 0:
            return
        activity = float(spikes.float().mean().item())
        alpha = 0.05
        self._activity_ema = (1.0 - alpha) * self._activity_ema + alpha * activity

    def maybe_grow(
        self,
        *,
        step_idx: int,
        populations: Sequence[PopulationSpec],
        projections: Sequence[ProjectionSpec],
    ) -> NeurogenesisDecision | None:
        if not self.config.enabled:
            self._update_scalars(
                populations=populations,
                added_neurons=0,
                added_edges=0,
                trigger_score=0.0,
                grew=False,
            )
            return None

        interval = max(1, int(self.config.growth_interval_steps))
        if step_idx <= 0 or (step_idx % interval) != 0:
            self._update_scalars(
                populations=populations,
                added_neurons=0,
                added_edges=0,
                trigger_score=0.0,
                grew=False,
            )
            return None

        target_name = self._target_population or self._resolve_target_population(populations)
        if target_name is None:
            self._update_scalars(
                populations=populations,
                added_neurons=0,
                added_edges=0,
                trigger_score=0.0,
                grew=False,
            )
            return None

        trigger_score = 1.0 if self._activity_ema <= float(self.config.activity_threshold) else 0.0
        if trigger_score > 0.0:
            self._interval_trigger_streak += 1
        else:
            self._interval_trigger_streak = 0

        if self._interval_trigger_streak < max(1, int(self.config.plateau_intervals)):
            self._update_scalars(
                populations=populations,
                added_neurons=0,
                added_edges=0,
                trigger_score=trigger_score,
                grew=False,
            )
            return None

        pop_by_name = {pop.name: pop for pop in populations}
        target_pop = pop_by_name.get(target_name)
        if target_pop is None:
            self._update_scalars(
                populations=populations,
                added_neurons=0,
                added_edges=0,
                trigger_score=0.0,
                grew=False,
            )
            return None

        total_neurons = sum(int(pop.n) for pop in populations)
        remaining = max(0, int(self.config.max_total_neurons) - total_neurons)
        add_n = min(max(1, int(self.config.add_neurons_per_event)), remaining)
        if add_n <= 0:
            self._update_scalars(
                populations=populations,
                added_neurons=0,
                added_edges=0,
                trigger_score=trigger_score,
                grew=False,
            )
            return None

        old_n = int(target_pop.n)
        new_n = old_n + add_n
        new_target = self._grow_population_spec(target_pop, add_n=add_n)
        updated_populations = tuple(
            new_target if pop.name == target_name else pop for pop in populations
        )
        updated_sizes = {pop.name: int(pop.n) for pop in updated_populations}

        updated_projections: list[ProjectionSpec] = []
        total_added_edges = 0
        for proj in projections:
            if proj.pre != target_name and proj.post != target_name:
                updated_projections.append(proj)
                continue
            grown_proj, added_edges = self._grow_projection(
                proj,
                old_target_n=old_n,
                new_target_n=new_n,
                step_idx=step_idx,
                n_pre=updated_sizes[proj.pre],
                n_post=updated_sizes[proj.post],
            )
            total_added_edges += added_edges
            updated_projections.append(grown_proj)

        self._events_total += 1
        self._interval_trigger_streak = 0
        self._update_scalars(
            populations=updated_populations,
            added_neurons=add_n,
            added_edges=total_added_edges,
            trigger_score=trigger_score,
            grew=True,
        )
        if self.config.verbose:
            print(
                "[neurogenesis] "
                f"step={int(step_idx)} "
                f"target={target_name} "
                f"added_neurons={int(add_n)} "
                f"added_edges={int(total_added_edges)} "
                f"total_neurons={int(sum(pop.n for pop in updated_populations))}"
            )
        return NeurogenesisDecision(
            populations=updated_populations,
            projections=tuple(updated_projections),
            target_population=target_name,
            added_neurons=add_n,
            added_edges=total_added_edges,
        )

    def _grow_population_spec(self, pop: PopulationSpec, *, add_n: int) -> PopulationSpec:
        torch = require_torch()
        old_n = int(pop.n)
        new_n = old_n + int(add_n)
        positions = pop.positions
        if positions is not None:
            device = positions.device
            dtype = positions.dtype
            if positions.numel() == 0:
                new_positions = torch.zeros((new_n, 3), device=device, dtype=dtype)
            else:
                mean_pos = positions.mean(dim=0, keepdim=True)
                jitter_scale = 0.01
                jitter = torch.randn((add_n, positions.shape[1]), device=device, dtype=dtype) * jitter_scale
                newborn = mean_pos.expand(add_n, -1) + jitter
                new_positions = torch.cat([positions, newborn], dim=0)
        else:
            new_positions = None
        meta = dict(pop.meta) if pop.meta else {}
        meta["neurogenesis_last_added"] = int(add_n)
        meta["neurogenesis_total_n"] = int(new_n)
        return PopulationSpec(
            name=pop.name,
            model=pop.model,
            n=new_n,
            frame=pop.frame,
            positions=new_positions,
            meta=meta,
        )

    def _grow_projection(
        self,
        proj: ProjectionSpec,
        *,
        old_target_n: int,
        new_target_n: int,
        step_idx: int,
        n_pre: int,
        n_post: int,
    ) -> tuple[ProjectionSpec, int]:
        torch = require_torch()
        topology = proj.topology
        device = topology.pre_idx.device
        add_n = new_target_n - old_target_n
        if add_n <= 0:
            return proj, 0

        pre_new = self._new_pre_edges(
            proj=proj,
            old_target_n=old_target_n,
            add_n=add_n,
            n_post=n_post,
            device=device,
        )
        post_new = self._new_post_edges(
            proj=proj,
            old_target_n=old_target_n,
            add_n=add_n,
            n_pre=n_pre,
            device=device,
        )
        all_pre = [topology.pre_idx]
        all_post = [topology.post_idx]
        if pre_new is not None:
            all_pre.append(pre_new[0])
            all_post.append(pre_new[1])
        if post_new is not None:
            all_pre.append(post_new[0])
            all_post.append(post_new[1])

        pre_idx_new = torch.cat(all_pre) if len(all_pre) > 1 else all_pre[0]
        post_idx_new = torch.cat(all_post) if len(all_post) > 1 else all_post[0]

        old_edges = int(topology.pre_idx.numel())
        new_edges = int(pre_idx_new.numel())
        added_edges = max(0, new_edges - old_edges)

        weights_new = _grow_optional_tensor(
            topology.weights,
            added=added_edges,
            fill=_new_weight_fill(
                weights=topology.weights,
                multiplier=float(self.config.newborn_plasticity_multiplier),
            ),
        )
        delay_fill = int(_median_int_tensor(topology.delay_steps))
        delay_steps_new = _grow_optional_int_tensor(topology.delay_steps, added=added_edges, fill=delay_fill)
        target_comps_new = _grow_optional_int_tensor(
            topology.target_compartments,
            added=added_edges,
            fill=_compartment_id(topology.target_compartment),
        )
        receptor_new = _grow_optional_int_tensor(topology.receptor, added=added_edges, fill=0)
        edge_dist_new = _grow_edge_distance(
            topology=topology,
            pre_idx_new=pre_idx_new,
            post_idx_new=post_idx_new,
            old_edges=old_edges,
        )
        myelin_fill = _mean_tensor(topology.myelin, default=0.5)
        myelin_new = _grow_optional_tensor(topology.myelin, added=added_edges, fill=myelin_fill)

        meta = {}
        if topology.meta:
            for key, value in topology.meta.items():
                if key.startswith("fused_"):
                    continue
                if key.startswith("edge_bucket_"):
                    continue
                if key in {"W_by_delay", "W_by_delay_csr", "W_by_delay_by_comp", "W_by_delay_by_comp_csr"}:
                    continue
                if key in {"values_by_comp", "indices_by_comp", "scale_by_comp"}:
                    continue
                if key in {"nonempty_mats_by_comp", "nonempty_mats_by_comp_csr", "edges_by_delay"}:
                    continue
                if key in {"pre_ptr", "edge_idx"}:
                    continue
                meta[key] = value

        newborn_mask_old = topology.meta.get("newborn_edge_mask") if topology.meta else None
        torch = require_torch()
        newborn_mask = torch.zeros((new_edges,), device=device, dtype=torch.bool)
        if torch.is_tensor(newborn_mask_old):
            newborn_mask_old_t = cast(Tensor, newborn_mask_old)
            if newborn_mask_old_t.numel() == old_edges:
                newborn_mask[:old_edges] = newborn_mask_old_t.to(
                    device=device,
                    dtype=torch.bool,
                )
        if added_edges > 0:
            newborn_mask[old_edges:] = True
        meta["newborn_edge_mask"] = newborn_mask
        meta["newborn_until_step"] = int(step_idx + max(1, int(self.config.newborn_duration_steps)))
        meta["newborn_plasticity_multiplier"] = float(self.config.newborn_plasticity_multiplier)

        grown_topology = SynapseTopology(
            pre_idx=pre_idx_new,
            post_idx=post_idx_new,
            delay_steps=delay_steps_new,
            edge_dist=edge_dist_new,
            target_compartment=topology.target_compartment,
            target_compartments=target_comps_new,
            receptor=receptor_new,
            receptor_kinds=topology.receptor_kinds,
            weights=weights_new,
            pre_pos=topology.pre_pos,
            post_pos=topology.post_pos,
            myelin=myelin_new,
            meta=meta,
        )
        return (
            ProjectionSpec(
                name=proj.name,
                synapse=proj.synapse,
                topology=grown_topology,
                pre=proj.pre,
                post=proj.post,
                learning=proj.learning,
                learn_every=proj.learn_every,
                sparse_learning=proj.sparse_learning,
                meta=proj.meta,
            ),
            added_edges,
        )

    def _new_pre_edges(
        self,
        *,
        proj: ProjectionSpec,
        old_target_n: int,
        add_n: int,
        n_post: int,
        device: Any,
    ) -> tuple[Tensor, Tensor] | None:
        if proj.pre != self._target_population:
            return None
        torch = require_torch()
        fan_out = max(1, int(round(float(self.config.connectivity_p) * max(1, n_post))))
        rows = torch.arange(add_n, device=device, dtype=torch.long).repeat_interleave(fan_out)
        pre_idx = rows + int(old_target_n)
        post_idx = torch.randint(0, int(n_post), (rows.numel(),), device=device, dtype=torch.long)
        return pre_idx, post_idx

    def _new_post_edges(
        self,
        *,
        proj: ProjectionSpec,
        old_target_n: int,
        add_n: int,
        n_pre: int,
        device: Any,
    ) -> tuple[Tensor, Tensor] | None:
        if proj.post != self._target_population:
            return None
        torch = require_torch()
        fan_in = max(1, int(round(float(self.config.connectivity_p) * max(1, n_pre))))
        cols = torch.arange(add_n, device=device, dtype=torch.long).repeat_interleave(fan_in)
        post_idx = cols + int(old_target_n)
        pre_idx = torch.randint(0, int(n_pre), (cols.numel(),), device=device, dtype=torch.long)
        return pre_idx, post_idx

    def _resolve_target_population(
        self,
        populations: Sequence[PopulationSpec],
    ) -> str | None:
        token = str(self.config.target_population_contains).strip().lower()
        for pop in populations:
            if token and token in pop.name.lower():
                return pop.name
        for pop in populations:
            if pop.name.lower().startswith("hidden"):
                return pop.name
        return populations[0].name if populations else None

    def _update_scalars(
        self,
        *,
        populations: Sequence[PopulationSpec],
        added_neurons: int,
        added_edges: int,
        trigger_score: float,
        grew: bool,
    ) -> None:
        total_neurons = int(sum(pop.n for pop in populations))
        self._last_scalars = {
            "neurogenesis/events_total": float(self._events_total),
            "neurogenesis/added_neurons_interval": float(max(0, int(added_neurons))),
            "neurogenesis/added_edges_interval": float(max(0, int(added_edges))),
            "neurogenesis/total_neurons": float(total_neurons),
            "neurogenesis/trigger_score": float(trigger_score),
            "neurogenesis/target_activity_ema": float(self._activity_ema),
            "neurogenesis/grew": 1.0 if grew else 0.0,
        }


def _new_weight_fill(*, weights: Tensor | None, multiplier: float) -> float:
    if weights is None or weights.numel() == 0:
        return 0.01 * multiplier
    mean_abs = float(weights.abs().mean().item())
    if mean_abs <= 0.0:
        mean_abs = 0.01
    sign = 1.0
    mean_val = float(weights.mean().item())
    if mean_val < 0.0:
        sign = -1.0
    return sign * mean_abs * float(multiplier)


def _grow_optional_tensor(tensor: Tensor | None, *, added: int, fill: float) -> Tensor | None:
    if tensor is None:
        return None
    if added <= 0:
        return tensor
    torch = require_torch()
    tail = torch.full((int(added),), float(fill), device=tensor.device, dtype=tensor.dtype)
    return cast(Tensor, torch.cat([tensor, tail], dim=0))


def _grow_optional_int_tensor(tensor: Tensor | None, *, added: int, fill: int) -> Tensor | None:
    if tensor is None:
        return None
    if added <= 0:
        return tensor
    torch = require_torch()
    tail = torch.full((int(added),), int(fill), device=tensor.device, dtype=tensor.dtype)
    return cast(Tensor, torch.cat([tensor, tail], dim=0))


def _grow_edge_distance(
    *,
    topology: SynapseTopology,
    pre_idx_new: Tensor,
    post_idx_new: Tensor,
    old_edges: int,
) -> Tensor | None:
    if topology.edge_dist is None:
        return None
    if topology.pre_pos is None or topology.post_pos is None:
        return _grow_optional_tensor(topology.edge_dist, added=max(0, int(pre_idx_new.numel()) - old_edges), fill=0.0)
    if int(pre_idx_new.numel()) <= old_edges:
        return topology.edge_dist
    torch = require_torch()
    pre_sel = topology.pre_pos.index_select(0, pre_idx_new[old_edges:])
    post_sel = topology.post_pos.index_select(0, post_idx_new[old_edges:])
    extra = torch.linalg.norm(pre_sel - post_sel, dim=1)
    return cast(Tensor, torch.cat([topology.edge_dist, extra.to(device=topology.edge_dist.device, dtype=topology.edge_dist.dtype)], dim=0))


def _median_int_tensor(tensor: Tensor | None) -> int:
    if tensor is None or tensor.numel() == 0:
        return 0
    torch = require_torch()
    return int(torch.median(tensor.to(dtype=torch.float32)).item())


def _mean_tensor(tensor: Tensor | None, *, default: float) -> float:
    if tensor is None or tensor.numel() == 0:
        return float(default)
    return float(tensor.float().mean().item())


def _compartment_id(compartment: Compartment) -> int:
    comp_order = tuple(Compartment)
    try:
        return int(comp_order.index(compartment))
    except ValueError:
        return 0


__all__ = [
    "NeurogenesisConfig",
    "NeurogenesisDecision",
    "NeurogenesisManager",
]
