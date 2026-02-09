"""Learning subsystem for TorchNetworkEngine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from biosnn.contracts.learning import LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.network.specs import ProjectionSpec

from .buffers import LearningScratch
from .models import NetworkRequirements


class LearningSubsystem:
    """Executes active-edge gathering and weight update application."""

    def projection_network_requirements(
        self,
        proj: ProjectionSpec,
        *,
        base_requirements: NetworkRequirements,
    ) -> NetworkRequirements:
        params = getattr(proj.synapse, "params", None)
        wants_by_delay_sparse = bool(getattr(params, "store_sparse_by_delay", False))
        needs_bucket_edge_mapping = bool(
            proj.learning is not None or getattr(params, "enable_sparse_updates", False)
        )
        return base_requirements.merge(
            NetworkRequirements(
                needs_bucket_edge_mapping=needs_bucket_edge_mapping,
                needs_by_delay_sparse=wants_by_delay_sparse,
            )
        )

    def require_weights(self, state: Any, proj_name: str) -> Tensor:
        if not hasattr(state, "weights"):
            raise RuntimeError(f"Projection {proj_name} learning requires synapse weights")
        return cast(Tensor, state.weights)

    def apply_weight_clamp(self, weights: Tensor, proj: ProjectionSpec) -> None:
        clamp_min = None
        clamp_max = None
        if proj.meta:
            clamp_min = proj.meta.get("clamp_min")
            clamp_max = proj.meta.get("clamp_max")
        if hasattr(proj.synapse, "params"):
            params = proj.synapse.params
            clamp_min = getattr(params, "clamp_min", clamp_min)
            clamp_max = getattr(params, "clamp_max", clamp_max)
        if clamp_min is None and clamp_max is None:
            return
        weights.clamp_(min=clamp_min, max=clamp_max)
        sync_fn = getattr(proj.synapse, "sync_sparse_values", None)
        if callable(sync_fn):
            sync_fn(proj.topology)

    def sparse_edge_cap(self, proj: ProjectionSpec) -> int | None:
        if proj.meta and "sparse_max_edges" in proj.meta:
            try:
                cap = int(proj.meta["sparse_max_edges"])
            except (TypeError, ValueError):
                return None
            return cap if cap > 0 else None
        return None

    def subset_edge_mods(
        self,
        edge_mods: Mapping[ModulatorKind, Tensor] | None,
        edges: Tensor,
    ) -> Mapping[ModulatorKind, Tensor] | None:
        if edge_mods is None:
            return None
        return {kind: value.index_select(0, edges) for kind, value in edge_mods.items()}

    def build_learning_batch(
        self,
        *,
        proj: ProjectionSpec,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        weights: Tensor,
        edge_mods: Mapping[ModulatorKind, Tensor] | None,
        device: Any,
        use_sparse: bool,
        scratch: LearningScratch | None,
    ) -> tuple[LearningBatch, Tensor | None]:
        if use_sparse:
            active_edges = self.active_edges_from_spikes(
                pre_spikes,
                proj.topology,
                device=device,
                max_edges=self.sparse_edge_cap(proj),
                scratch=scratch,
            )
            if scratch is not None:
                n_active = active_edges.numel()
                if (
                    scratch.size != n_active
                    or scratch.edge_pre_idx.device != device
                    or scratch.edge_pre.dtype != pre_spikes.dtype
                    or scratch.edge_post.dtype != post_spikes.dtype
                    or scratch.edge_weights.dtype != weights.dtype
                ):
                    torch = require_torch()
                    scratch.edge_pre_idx = torch.empty((n_active,), device=device, dtype=torch.long)
                    scratch.edge_post_idx = torch.empty((n_active,), device=device, dtype=torch.long)
                    scratch.edge_pre = torch.empty((n_active,), device=device, dtype=pre_spikes.dtype)
                    scratch.edge_post = torch.empty((n_active,), device=device, dtype=post_spikes.dtype)
                    scratch.edge_weights = torch.empty((n_active,), device=device, dtype=weights.dtype)
                    scratch.size = n_active

                edge_pre_idx = scratch.edge_pre_idx
                edge_post_idx = scratch.edge_post_idx
                edge_pre = scratch.edge_pre
                edge_post = scratch.edge_post
                edge_weights = scratch.edge_weights

                torch = require_torch()
                torch.index_select(proj.topology.pre_idx, 0, active_edges, out=edge_pre_idx)
                torch.index_select(proj.topology.post_idx, 0, active_edges, out=edge_post_idx)
                torch.index_select(pre_spikes, 0, edge_pre_idx, out=edge_pre)
                torch.index_select(post_spikes, 0, edge_post_idx, out=edge_post)
                torch.index_select(weights, 0, active_edges, out=edge_weights)
            else:
                edge_pre_idx = proj.topology.pre_idx.index_select(0, active_edges)
                edge_post_idx = proj.topology.post_idx.index_select(0, active_edges)
                edge_pre = pre_spikes.index_select(0, edge_pre_idx)
                edge_post = post_spikes.index_select(0, edge_post_idx)
                edge_weights = weights.index_select(0, active_edges)

            mods = self.subset_edge_mods(edge_mods, active_edges)
            batch = LearningBatch(
                pre_spikes=edge_pre,
                post_spikes=edge_post,
                weights=edge_weights,
                modulators=mods,
                extras={
                    "pre_idx": edge_pre_idx,
                    "post_idx": edge_post_idx,
                    "active_edges": active_edges,
                },
            )
            return batch, active_edges

        edge_pre = pre_spikes[proj.topology.pre_idx]
        edge_post = post_spikes[proj.topology.post_idx]
        batch = LearningBatch(
            pre_spikes=edge_pre,
            post_spikes=edge_post,
            weights=weights,
            modulators=edge_mods,
            extras={
                "pre_idx": proj.topology.pre_idx,
                "post_idx": proj.topology.post_idx,
            },
        )
        return batch, None

    def require_pre_adjacency(self, topology: SynapseTopology, device: Any) -> tuple[Tensor, Tensor]:
        if not topology.meta:
            raise ValueError(
                "Topology meta missing pre adjacency; compile_topology(..., build_pre_adjacency=True) "
                "must be called."
            )
        pre_ptr = topology.meta.get("pre_ptr")
        edge_idx = topology.meta.get("edge_idx")
        if pre_ptr is None or edge_idx is None:
            raise ValueError(
                "Topology meta missing pre adjacency; compile_topology(..., build_pre_adjacency=True) "
                "must be called."
            )
        if pre_ptr.device != device or edge_idx.device != device:
            raise ValueError("Adjacency tensors must be on the projection device")
        return cast(Tensor, pre_ptr), cast(Tensor, edge_idx)

    def get_arange(self, scratch: LearningScratch, needed: int, *, device: Any) -> Tensor:
        if scratch.arange_size < needed or scratch.arange_buf.device != device:
            torch = require_torch()
            scratch.arange_buf = torch.arange(needed, device=device, dtype=torch.long)
            scratch.arange_size = needed
        return scratch.arange_buf

    def gather_active_edges(
        self,
        active_pre: Tensor,
        pre_ptr: Tensor,
        edge_idx: Tensor,
        *,
        scratch: LearningScratch | None = None,
    ) -> Tensor:
        torch = require_torch()
        starts = pre_ptr.index_select(0, active_pre)
        ends = pre_ptr.index_select(0, active_pre + 1)
        counts = ends - starts
        if counts.numel() == 0:
            return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))
        base = torch.repeat_interleave(starts, counts)
        total = base.numel()
        if total == 0:
            return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))
        prefix = torch.cumsum(counts, 0)
        group_start = torch.repeat_interleave(prefix - counts, counts)
        if scratch is not None:
            intra = self.get_arange(scratch, total, device=edge_idx.device)[:total] - group_start
        else:
            intra = torch.arange(total, device=edge_idx.device) - group_start
        edge_pos = base + intra
        return cast(Tensor, edge_idx.index_select(0, edge_pos))

    def active_edges_from_spikes(
        self,
        pre_spikes: Tensor,
        topology: SynapseTopology,
        *,
        device: Any,
        max_edges: int | None = None,
        scratch: LearningScratch | None = None,
    ) -> Tensor:
        torch = require_torch()
        pre_ptr, edge_idx = self.require_pre_adjacency(topology, device)
        active_pre = pre_spikes.nonzero(as_tuple=False).flatten()
        if active_pre.numel() == 0:
            return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))
        edges = self.gather_active_edges(active_pre, pre_ptr, edge_idx, scratch=scratch)
        if max_edges is not None and edges.numel() > max_edges:
            return edges[:max_edges]
        return edges

    def apply_learning_update(
        self,
        *,
        weights: Tensor,
        result: LearningStepResult,
        active_edges: Tensor | None,
        proj_name: str,
        synapse: Any | None = None,
        topology: SynapseTopology | None = None,
    ) -> None:
        if active_edges is None and result.extras is not None:
            maybe_edges = result.extras.get("active_edges")
            if isinstance(maybe_edges, Tensor):
                active_edges = maybe_edges

        def _apply_to_weights(target: Tensor) -> None:
            if active_edges is None:
                target.add_(result.d_weights)
                return
            if result.d_weights.numel() == active_edges.numel():
                target.index_add_(0, active_edges, result.d_weights)
                return
            if result.d_weights.numel() == target.numel():
                target.add_(result.d_weights)
                return
            raise RuntimeError(
                "Sparse learning update shape mismatch for "
                f"{proj_name}: d_weights={tuple(result.d_weights.shape)} "
                f"active_edges={tuple(active_edges.shape)} "
                f"weights={tuple(target.shape)}"
            )

        if synapse is not None and topology is not None:
            apply_fn = getattr(synapse, "apply_weight_updates", None)
            if callable(apply_fn):
                apply_fn(topology, active_edges, result.d_weights)
                topo_weights = topology.weights
                if topo_weights is None or topo_weights is not weights:
                    _apply_to_weights(weights)
                return

        _apply_to_weights(weights)


__all__ = ["LearningSubsystem"]
