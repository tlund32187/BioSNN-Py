"""Buffer and scratch allocation subsystem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.network.specs import PopulationSpec, ProjectionSpec


@dataclass(slots=True)
class LearningScratch:
    edge_pre_idx: Tensor
    edge_post_idx: Tensor
    edge_pre: Tensor
    edge_post: Tensor
    edge_weights: Tensor
    size: int
    arange_buf: Tensor
    arange_size: int


class BufferSubsystem:
    """Owns allocation helpers for drive buffers, rings, and scratch tensors."""

    def collect_drive_compartments(
        self,
        pop_specs: Sequence[PopulationSpec],
        proj_specs: Sequence[ProjectionSpec],
    ) -> dict[str, tuple[Compartment, ...]]:
        comp_order = tuple(Compartment)
        comps_by_pop: dict[str, set[Compartment]] = {
            spec.name: set(spec.model.compartments) for spec in pop_specs
        }
        for proj in proj_specs:
            comps = comps_by_pop.get(proj.post)
            if comps is None:
                continue
            topology: SynapseTopology = proj.topology
            if topology.target_compartments is None:
                comps.add(topology.target_compartment)
                continue
            meta = topology.meta or {}
            comp_ids = meta.get("target_comp_ids")
            if isinstance(comp_ids, list) and comp_ids:
                for comp_id in comp_ids:
                    try:
                        idx = int(comp_id)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(comp_order):
                        comps.add(comp_order[idx])
                continue
            comps.update(comp_order)
        return {name: tuple(comps) for name, comps in comps_by_pop.items()}

    def zero_drive(
        self,
        *,
        model: Any,
        n: int,
        device: Any,
        dtype: Any,
    ) -> dict[Compartment, Tensor]:
        torch = require_torch()
        return {comp: torch.zeros((n,), device=device, dtype=dtype) for comp in model.compartments}

    def get_learning_scratch(
        self,
        *,
        scratch_by_proj: dict[str, LearningScratch],
        proj_name: str,
        device: Any,
    ) -> LearningScratch:
        scratch = scratch_by_proj.get(proj_name)
        if scratch is not None:
            return scratch
        torch = require_torch()
        scratch = LearningScratch(
            edge_pre_idx=torch.empty((0,), device=device, dtype=torch.long),
            edge_post_idx=torch.empty((0,), device=device, dtype=torch.long),
            edge_pre=torch.empty((0,), device=device, dtype=torch.bool),
            edge_post=torch.empty((0,), device=device, dtype=torch.bool),
            edge_weights=torch.empty((0,), device=device, dtype=torch.float32),
            size=0,
            arange_buf=torch.empty((0,), device=device, dtype=torch.long),
            arange_size=0,
        )
        scratch_by_proj[proj_name] = scratch
        return scratch

    def get_ring(self, *, projection_state: Any, compartment: Compartment) -> Tensor | None:
        ring = getattr(projection_state, "post_ring", None)
        if isinstance(ring, Mapping):
            value = ring.get(compartment)
            if value is not None:
                return cast(Tensor, value)
        return None


__all__ = ["BufferSubsystem", "LearningScratch"]
