"""Activity-based structural pruning manager."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.network.specs import ProjectionSpec


@dataclass(frozen=True, slots=True)
class StructuralPruningConfig:
    """Configuration for periodic edge pruning."""

    enabled: bool = False
    prune_interval_steps: int = 250
    usage_alpha: float = 0.01
    w_min: float = 0.01
    usage_min: float = 0.01
    k_min_out: int = 0
    k_min_in: int = 0
    max_prune_fraction_per_interval: float = 0.1
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class ProjectionPruneDecision:
    projection_name: str
    topology: SynapseTopology
    old_edges: int
    new_edges: int
    pruned_edges: int


class StructuralPlasticityManager:
    """Tracks per-edge usage and emits topology-prune decisions."""

    def __init__(self, config: StructuralPruningConfig | None = None) -> None:
        self.config = config or StructuralPruningConfig()
        self._usage_ema_by_proj: dict[str, Tensor] = {}
        self._edge_counts: dict[str, int] = {}
        self._last_scalars: dict[str, float] = {}

    def reset(
        self,
        projections: Sequence[ProjectionSpec],
        *,
        device: Any,
        dtype: Any,
    ) -> None:
        torch = require_torch()
        self._usage_ema_by_proj = {}
        self._edge_counts = {}
        for proj in projections:
            edge_count = _edge_count(proj.topology)
            self._usage_ema_by_proj[proj.name] = torch.zeros((edge_count,), device=device, dtype=dtype)
            self._edge_counts[proj.name] = edge_count
        self._update_scalars(pruned_edges=0, projections_pruned=0)

    def scalars(self) -> Mapping[str, float]:
        return self._last_scalars

    def record_projection_activity(
        self,
        *,
        projection: ProjectionSpec,
        pre_spikes: Tensor,
        d_weights: Tensor | None = None,
        active_edges: Tensor | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        usage = self._usage_ema_by_proj.get(projection.name)
        if usage is None:
            return
        edge_count = _edge_count(projection.topology)
        if usage.numel() != edge_count:
            torch = require_torch()
            usage = torch.zeros((edge_count,), device=pre_spikes.device, dtype=usage.dtype)
            self._usage_ema_by_proj[projection.name] = usage
            self._edge_counts[projection.name] = edge_count
        if usage.numel() == 0:
            return

        alpha = _clamp01(float(self.config.usage_alpha))
        if alpha <= 0.0:
            return
        usage.mul_(1.0 - alpha)

        used_dw = False
        if d_weights is not None and d_weights.numel() > 0:
            abs_dw = d_weights.abs()
            if abs_dw.numel() == usage.numel():
                usage.add_(abs_dw.to(device=usage.device, dtype=usage.dtype), alpha=alpha)
                used_dw = True
            elif active_edges is not None and active_edges.numel() == abs_dw.numel():
                idx = active_edges.to(device=usage.device, dtype=require_torch().long)
                usage.index_add_(
                    0,
                    idx,
                    abs_dw.to(device=usage.device, dtype=usage.dtype) * alpha,
                )
                used_dw = True

        if used_dw:
            return
        edge_pre = pre_spikes.index_select(0, projection.topology.pre_idx)
        usage.add_(edge_pre.to(device=usage.device, dtype=usage.dtype), alpha=alpha)

    def maybe_prune(
        self,
        *,
        step_idx: int,
        projections: Sequence[ProjectionSpec],
        weights_by_projection: Mapping[str, Tensor | None],
    ) -> list[ProjectionPruneDecision]:
        if not self.config.enabled:
            self._update_scalars(pruned_edges=0, projections_pruned=0)
            return []
        interval = max(1, int(self.config.prune_interval_steps))
        if step_idx <= 0 or (step_idx % interval) != 0:
            self._update_scalars(pruned_edges=0, projections_pruned=0)
            return []

        decisions: list[ProjectionPruneDecision] = []
        total_pruned = 0
        for proj in projections:
            weights = weights_by_projection.get(proj.name)
            if weights is None or weights.numel() == 0:
                self._edge_counts[proj.name] = int(weights.numel()) if weights is not None else 0
                continue

            usage = self._usage_ema_by_proj.get(proj.name)
            if usage is None or usage.numel() != weights.numel():
                torch = require_torch()
                usage = torch.zeros((weights.numel(),), device=weights.device, dtype=weights.dtype)
                self._usage_ema_by_proj[proj.name] = usage

            keep_mask = self._compute_keep_mask(
                topology=proj.topology,
                weights=weights,
                usage=usage,
            )
            old_edges = int(weights.numel())
            new_edges = int(keep_mask.sum().item())
            if new_edges >= old_edges:
                self._edge_counts[proj.name] = old_edges
                continue

            keep_idx = keep_mask.nonzero(as_tuple=False).flatten()
            new_topology = self.compact_topology(proj.topology, keep_idx=keep_idx, weights=weights)
            self._usage_ema_by_proj[proj.name] = usage.index_select(0, keep_idx)
            self._edge_counts[proj.name] = new_edges

            pruned = old_edges - new_edges
            total_pruned += pruned
            decisions.append(
                ProjectionPruneDecision(
                    projection_name=proj.name,
                    topology=new_topology,
                    old_edges=old_edges,
                    new_edges=new_edges,
                    pruned_edges=pruned,
                )
            )

        self._update_scalars(pruned_edges=total_pruned, projections_pruned=len(decisions))
        if self.config.verbose and decisions:
            total_edges = sum(self._edge_counts.values())
            print(
                "[structural_prune] "
                f"step={int(step_idx)} "
                f"projections_pruned={len(decisions)} "
                f"edges_pruned={int(total_pruned)} "
                f"edges_total={int(total_edges)}"
            )
        return decisions

    def compact_topology(
        self,
        topology: SynapseTopology,
        *,
        keep_idx: Tensor,
        weights: Tensor | None = None,
    ) -> SynapseTopology:
        filtered_weights = weights if weights is not None else topology.weights
        return SynapseTopology(
            pre_idx=topology.pre_idx.index_select(0, keep_idx),
            post_idx=topology.post_idx.index_select(0, keep_idx),
            delay_steps=_maybe_index_select(topology.delay_steps, keep_idx),
            edge_dist=_maybe_index_select(topology.edge_dist, keep_idx),
            target_compartment=topology.target_compartment,
            target_compartments=_maybe_index_select(topology.target_compartments, keep_idx),
            receptor=_maybe_index_select(topology.receptor, keep_idx),
            receptor_kinds=topology.receptor_kinds,
            weights=_maybe_index_select(filtered_weights, keep_idx),
            pre_pos=topology.pre_pos,
            post_pos=topology.post_pos,
            myelin=_maybe_index_select(topology.myelin, keep_idx),
            meta=None,
        )

    def _compute_keep_mask(
        self,
        *,
        topology: SynapseTopology,
        weights: Tensor,
        usage: Tensor,
    ) -> Tensor:
        torch = require_torch()
        edge_count = int(weights.numel())
        if edge_count == 0:
            return cast(Tensor, torch.zeros((0,), device=weights.device, dtype=torch.bool))

        prune_candidates = (weights.abs() <= float(self.config.w_min)) & (
            usage <= float(self.config.usage_min)
        )
        if not bool(prune_candidates.any()):
            return cast(Tensor, ~torch.zeros_like(prune_candidates))

        drop_mask = prune_candidates.clone()

        max_fraction = _clamp01(float(self.config.max_prune_fraction_per_interval))
        if max_fraction <= 0.0:
            drop_mask.zero_()
        elif max_fraction < 1.0:
            max_drop = max(1, int(edge_count * max_fraction))
            drop_idx = drop_mask.nonzero(as_tuple=False).flatten()
            if drop_idx.numel() > max_drop:
                drop_score = usage.index_select(0, drop_idx)
                order = torch.argsort(drop_score, descending=False)
                selected = drop_idx.index_select(0, order[:max_drop])
                drop_mask.zero_()
                drop_mask[selected] = True

        n_pre = _infer_n_pre(topology)
        n_post = _infer_n_post(topology)
        if n_pre is not None and int(self.config.k_min_out) > 0:
            drop_mask = _enforce_degree_floor(
                drop_mask=drop_mask,
                node_idx=topology.pre_idx,
                n_nodes=n_pre,
                k_min=max(0, int(self.config.k_min_out)),
            )
        if n_post is not None and int(self.config.k_min_in) > 0:
            drop_mask = _enforce_degree_floor(
                drop_mask=drop_mask,
                node_idx=topology.post_idx,
                n_nodes=n_post,
                k_min=max(0, int(self.config.k_min_in)),
            )

        keep_mask = ~drop_mask
        if int(keep_mask.sum().item()) <= 0:
            keep_mask = torch.zeros_like(drop_mask)
            keep_idx = int((usage + weights.abs()).argmax().item())
            keep_mask[keep_idx] = True
        return keep_mask

    def _update_scalars(self, *, pruned_edges: int, projections_pruned: int) -> None:
        total_edges = int(sum(self._edge_counts.values()))
        self._last_scalars = {
            "prune/edges_total": float(total_edges),
            "prune/edges_pruned_interval": float(max(0, int(pruned_edges))),
            "prune/projections_pruned_interval": float(max(0, int(projections_pruned))),
        }


def _edge_count(topology: SynapseTopology) -> int:
    if hasattr(topology.pre_idx, "numel"):
        return int(topology.pre_idx.numel())
    try:
        return len(topology.pre_idx)
    except TypeError:
        return 0


def _infer_n_pre(topology: SynapseTopology) -> int | None:
    meta = topology.meta or {}
    n_pre = meta.get("n_pre")
    if n_pre is not None:
        try:
            return int(n_pre)
        except (TypeError, ValueError):
            return None
    if topology.pre_pos is not None and hasattr(topology.pre_pos, "shape"):
        return int(topology.pre_pos.shape[0])
    if topology.pre_idx.numel():
        return int(topology.pre_idx.max().item()) + 1
    return None


def _infer_n_post(topology: SynapseTopology) -> int | None:
    meta = topology.meta or {}
    n_post = meta.get("n_post")
    if n_post is not None:
        try:
            return int(n_post)
        except (TypeError, ValueError):
            return None
    if topology.post_pos is not None and hasattr(topology.post_pos, "shape"):
        return int(topology.post_pos.shape[0])
    if topology.post_idx.numel():
        return int(topology.post_idx.max().item()) + 1
    return None


def _enforce_degree_floor(
    *,
    drop_mask: Tensor,
    node_idx: Tensor,
    n_nodes: int,
    k_min: int,
) -> Tensor:
    torch = require_torch()
    if n_nodes <= 0 or k_min <= 0 or node_idx.numel() == 0:
        return drop_mask
    total_degree = torch.bincount(node_idx, minlength=int(n_nodes))
    dropped_nodes = node_idx.index_select(0, drop_mask.nonzero(as_tuple=False).flatten())
    dropped_degree = torch.bincount(dropped_nodes, minlength=int(n_nodes))
    remaining = total_degree - dropped_degree
    violating = remaining < int(k_min)
    if not bool(violating.any()):
        return drop_mask
    violating_by_edge = violating.index_select(0, node_idx)
    return cast(Tensor, drop_mask & (~violating_by_edge))


def _maybe_index_select(tensor: Tensor | None, keep_idx: Tensor) -> Tensor | None:
    if tensor is None:
        return None
    return cast(Tensor, tensor.index_select(0, keep_idx))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "ProjectionPruneDecision",
    "StructuralPlasticityManager",
    "StructuralPruningConfig",
]
