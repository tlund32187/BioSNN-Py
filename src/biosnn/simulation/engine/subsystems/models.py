"""Shared subsystem dataclasses for engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biosnn.contracts.synapses import ISynapseModel, SynapseTopology
from biosnn.simulation.network.specs import ProjectionSpec


@dataclass(frozen=True, slots=True)
class EngineContext:
    """Minimal immutable runtime context shared with subsystems."""

    device: Any
    dtype: Any
    dt: float
    seed: int | None
    rng: Any | None = None


@dataclass(slots=True)
class ProjectionPlan:
    """Compiled projection execution plan."""

    name: str
    pre_name: str
    post_name: str
    synapse: ISynapseModel
    topology: SynapseTopology
    learning_enabled: bool
    needs_bucket_mapping: bool
    use_fused_sparse: bool
    fast_mode: bool
    compiled_mode: bool


@dataclass(slots=True)
class CompiledNetworkPlan:
    """Bundle of compiled projection specs + projection execution plans."""

    compiled_projections: list[ProjectionSpec]
    projection_plans: list[ProjectionPlan]
    projection_index: dict[str, ProjectionPlan]


__all__ = ["CompiledNetworkPlan", "EngineContext", "ProjectionPlan"]
