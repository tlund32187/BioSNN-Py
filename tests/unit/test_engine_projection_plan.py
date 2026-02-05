import pytest
pytestmark = pytest.mark.unit


from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.synapses import SynapseTopology
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentSynapse
from biosnn.synapses.dynamics.delayed_sparse_matmul import DelayedSparseMatmulSynapse

torch = pytest.importorskip("torch")


class DummyLearning(ILearningRule):
    name = "dummy"
    supports_sparse = False

    def init_state(self, e: int, *, ctx: StepContext):
        _ = (e, ctx)
        return None

    def reset_state(self, state, *, ctx: StepContext, edge_indices=None):
        _ = (state, ctx, edge_indices)
        return state

    def step(
        self,
        state,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[None, LearningStepResult]:
        _ = (state, dt, t, ctx)
        return None, LearningStepResult(d_weights=torch.zeros_like(batch.weights))

    def state_tensors(self, state):
        _ = state
        return {}


def _topology() -> SynapseTopology:
    pre_idx = torch.tensor([0, 1], dtype=torch.long)
    post_idx = torch.tensor([0, 1], dtype=torch.long)
    delay_steps = torch.tensor([0, 1], dtype=torch.int32)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
    )


def test_projection_plans_flags():
    pops = [
        PopulationSpec(name="A", model=GLIFModel(), n=2),
        PopulationSpec(name="B", model=GLIFModel(), n=2),
    ]
    proj_plain = ProjectionSpec(
        name="A->B",
        synapse=DelayedCurrentSynapse(),
        topology=_topology(),
        pre="A",
        post="B",
    )
    proj_learn = ProjectionSpec(
        name="B->A",
        synapse=DelayedSparseMatmulSynapse(),
        topology=_topology(),
        pre="B",
        post="A",
        learning=DummyLearning(),
    )
    engine = TorchNetworkEngine(populations=pops, projections=[proj_plain, proj_learn])
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu", dtype="float32"))

    plans = {plan.name: plan for plan in engine._proj_plan_list}
    assert set(plans) == {"A->B", "B->A"}

    plain = plans["A->B"]
    assert plain.learning_enabled is False
    assert plain.needs_bucket_mapping is False
    assert plain.use_fused_sparse is False
    assert plain.fast_mode is False
    assert plain.compiled_mode is False

    learn = plans["B->A"]
    assert learn.learning_enabled is True
    assert learn.needs_bucket_mapping is True
    assert learn.use_fused_sparse is True
