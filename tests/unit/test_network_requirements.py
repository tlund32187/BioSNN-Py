from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.simulation.engine.subsystems import (
    STEP_EVENT_KEY_PROJECTION_WEIGHTS,
    STEP_EVENT_KEY_SCALARS,
    STEP_EVENT_KEY_SPIKES,
    STEP_EVENT_KEY_SYNAPSE_STATE,
    STEP_EVENT_KEY_V_SOMA,
    LearningSubsystem,
    MonitorSubsystem,
    NetworkRequirements,
    StepEventSubsystem,
    TopologySubsystem,
)
from biosnn.simulation.network import ProjectionSpec
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


class _SpikesScalarsMonitor(IMonitor):
    name = "spikes_scalars"

    def requirements(self) -> MonitorRequirements:
        return MonitorRequirements(needs_spikes=True, needs_scalars=True)

    def on_step(self, event: StepEvent) -> None:
        _ = event

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _DummyLearning(ILearningRule):
    name = "dummy_learning"
    supports_sparse = True

    def init_state(self, e: int, *, ctx: StepContext) -> None:
        _ = e, ctx
        return None

    def step(
        self,
        state: None,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[None, LearningStepResult]:
        _ = state, dt, t, ctx
        return None, LearningStepResult(d_weights=batch.weights)

    def state_tensors(self, state: None) -> Mapping[str, Tensor]:
        _ = state
        return {}


def _projection(
    *,
    synapse: DelayedSparseMatmulSynapse,
    learning: ILearningRule | None = None,
) -> ProjectionSpec:
    topology = SynapseTopology(
        pre_idx=cast(Tensor, torch.tensor([0], dtype=torch.long)),
        post_idx=cast(Tensor, torch.tensor([0], dtype=torch.long)),
        delay_steps=cast(Tensor, torch.tensor([1], dtype=torch.int32)),
        target_compartment=Compartment.SOMA,
    )
    return ProjectionSpec(
        name="P",
        synapse=synapse,
        topology=topology,
        pre="A",
        post="B",
        learning=learning,
        sparse_learning=learning is not None,
    )


def test_monitor_subsystem_builds_network_requirements() -> None:
    subsystem = MonitorSubsystem()
    requirements = subsystem.build_network_requirements([_SpikesScalarsMonitor()])

    assert requirements.needs_step_event(STEP_EVENT_KEY_SPIKES)
    assert requirements.needs_step_event(STEP_EVENT_KEY_SCALARS)
    assert not requirements.needs_step_event(STEP_EVENT_KEY_PROJECTION_WEIGHTS)


def test_learning_subsystem_adds_bucket_mapping_requirement() -> None:
    learning_subsystem = LearningSubsystem()
    proj = _projection(
        synapse=DelayedSparseMatmulSynapse(DelayedSparseMatmulParams()),
        learning=_DummyLearning(),
    )
    requirements = learning_subsystem.projection_network_requirements(
        proj,
        base_requirements=NetworkRequirements.none(),
    )

    assert requirements.needs_bucket_edge_mapping


def test_topology_subsystem_reads_synapse_requirement_preferences() -> None:
    topology_subsystem = TopologySubsystem()
    proj = _projection(
        synapse=DelayedSparseMatmulSynapse(
            DelayedSparseMatmulParams(
                fused_layout="csr",
                store_sparse_by_delay=True,
                ring_dtype="bfloat16",
            )
        )
    )
    requirements = topology_subsystem.projection_network_requirements(
        proj,
        base_requirements=NetworkRequirements.none(),
    )

    assert requirements.wants_fused_layout == "csr"
    assert requirements.needs_by_delay_sparse
    assert requirements.ring_dtype == "bfloat16"

    flags = topology_subsystem.compile_flags_for_projection(
        proj,
        requirements=requirements,
        device="cpu",
    )
    assert flags.build_sparse_delay_mats
    assert flags.store_sparse_by_delay
    assert flags.build_fused_csr


def test_step_event_payload_plan_uses_network_requirements() -> None:
    event_subsystem = StepEventSubsystem()
    requirements = NetworkRequirements(
        needed_step_event_keys=frozenset(
            {
                STEP_EVENT_KEY_SPIKES,
                STEP_EVENT_KEY_V_SOMA,
                STEP_EVENT_KEY_SCALARS,
                STEP_EVENT_KEY_SYNAPSE_STATE,
                STEP_EVENT_KEY_PROJECTION_WEIGHTS,
            }
        )
    )
    plan = event_subsystem.payload_plan(requirements, fast_mode=False)

    assert plan.needs_event_spikes
    assert plan.needs_population_tensors
    assert plan.needs_synapse_state
    assert not plan.needs_projection_weights
