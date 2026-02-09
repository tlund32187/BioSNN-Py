"""Shared subsystem dataclasses for engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biosnn.contracts.synapses import ISynapseModel, SynapseTopology
from biosnn.simulation.network.specs import ProjectionSpec

STEP_EVENT_KEY_SPIKES = "spikes"
STEP_EVENT_KEY_V_SOMA = "v_soma"
STEP_EVENT_KEY_POPULATION_STATE = "population_state"
STEP_EVENT_KEY_PROJECTION_WEIGHTS = "projection_weights"
STEP_EVENT_KEY_PROJECTION_DRIVE = "projection_drive"
STEP_EVENT_KEY_SYNAPSE_STATE = "synapse_state"
STEP_EVENT_KEY_MODULATORS = "modulators"
STEP_EVENT_KEY_LEARNING_STATE = "learning_state"
STEP_EVENT_KEY_HOMEOSTASIS_STATE = "homeostasis_state"
STEP_EVENT_KEY_SCALARS = "scalars"
STEP_EVENT_KEY_POPULATION_SLICES = "population_slices"


@dataclass(frozen=True, slots=True)
class NetworkRequirements:
    """Combined requirements contributed by monitor, learning, and synapse subsystems."""

    needed_step_event_keys: frozenset[str] = frozenset()
    needs_bucket_edge_mapping: bool = False
    needs_by_delay_sparse: bool = False
    wants_fused_layout: str = "auto"
    ring_strategy: str = "dense"
    ring_dtype: str | None = None

    @classmethod
    def none(cls) -> NetworkRequirements:
        return cls()

    def needs_step_event(self, key: str) -> bool:
        return key in self.needed_step_event_keys

    def merge(self, other: NetworkRequirements) -> NetworkRequirements:
        merged_ring_dtype: str | None
        if self.ring_dtype == other.ring_dtype:
            merged_ring_dtype = self.ring_dtype
        elif self.ring_dtype is None:
            merged_ring_dtype = other.ring_dtype
        elif other.ring_dtype is None:
            merged_ring_dtype = self.ring_dtype
        else:
            merged_ring_dtype = other.ring_dtype

        return NetworkRequirements(
            needed_step_event_keys=self.needed_step_event_keys | other.needed_step_event_keys,
            needs_bucket_edge_mapping=self.needs_bucket_edge_mapping
            or other.needs_bucket_edge_mapping,
            needs_by_delay_sparse=self.needs_by_delay_sparse or other.needs_by_delay_sparse,
            wants_fused_layout=_merge_choice(
                self.wants_fused_layout,
                other.wants_fused_layout,
                default="auto",
            ),
            ring_strategy=_merge_choice(
                self.ring_strategy,
                other.ring_strategy,
                default="dense",
            ),
            ring_dtype=merged_ring_dtype,
        )


def _merge_choice(current: str, incoming: str, *, default: str) -> str:
    if current == incoming:
        return current
    if current == default:
        return incoming
    if incoming == default:
        return current
    return incoming


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


__all__ = [
    "CompiledNetworkPlan",
    "EngineContext",
    "NetworkRequirements",
    "ProjectionPlan",
    "STEP_EVENT_KEY_HOMEOSTASIS_STATE",
    "STEP_EVENT_KEY_LEARNING_STATE",
    "STEP_EVENT_KEY_MODULATORS",
    "STEP_EVENT_KEY_POPULATION_SLICES",
    "STEP_EVENT_KEY_POPULATION_STATE",
    "STEP_EVENT_KEY_PROJECTION_DRIVE",
    "STEP_EVENT_KEY_PROJECTION_WEIGHTS",
    "STEP_EVENT_KEY_SCALARS",
    "STEP_EVENT_KEY_SPIKES",
    "STEP_EVENT_KEY_SYNAPSE_STATE",
    "STEP_EVENT_KEY_V_SOMA",
]
