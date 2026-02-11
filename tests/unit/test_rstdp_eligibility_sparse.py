from __future__ import annotations

import pytest

from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.neurons import StepContext
from biosnn.learning.rules import RStdpEligibilityParams, RStdpEligibilityRule

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_sparse_step_no_shape_mismatch() -> None:
    rule = RStdpEligibilityRule(
        RStdpEligibilityParams(
            lr=0.05,
            tau_e=0.05,
            a_plus=1.0,
            a_minus=0.2,
            weight_decay=0.0,
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    state = rule.init_state(10, ctx=ctx)

    active_edges = torch.tensor([1, 5, 9], dtype=torch.long)
    batch = LearningBatch(
        pre_spikes=torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
        post_spikes=torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        weights=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        modulators={ModulatorKind.DOPAMINE: torch.ones((3,), dtype=torch.float32)},
        extras={
            "active_edges": active_edges,
            "pre_idx": torch.tensor([0, 1, 2], dtype=torch.long),
            "post_idx": torch.tensor([0, 1, 0], dtype=torch.long),
        },
    )

    state, res = rule.step(state, batch, dt=1e-3, t=0.0, ctx=ctx)

    assert res.d_weights.shape == (3,)
    assert res.extras is not None
    assert "active_edges" in res.extras
    torch.testing.assert_close(res.extras["active_edges"], active_edges)
    assert state.eligibility.shape == (10,)


def test_dense_requires_matching_shapes() -> None:
    rule = RStdpEligibilityRule()
    ctx = StepContext(device="cpu", dtype="float32")
    state = rule.init_state(5, ctx=ctx)
    batch = LearningBatch(
        pre_spikes=torch.ones((4,), dtype=torch.float32),
        post_spikes=torch.ones((4,), dtype=torch.float32),
        weights=torch.zeros((4,), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="Dense learning requires eligibility shape"):
        rule.step(state, batch, dt=1e-3, t=0.0, ctx=ctx)


def test_rstdp_sparse_updates_only_active_edges() -> None:
    rule = RStdpEligibilityRule(
        RStdpEligibilityParams(
            lr=0.1,
            tau_e=0.05,
            a_plus=1.0,
            a_minus=0.0,
            weight_decay=0.0,
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    state = rule.init_state(10, ctx=ctx)
    before = state.eligibility.clone()

    active_edges = torch.tensor([2, 7], dtype=torch.long)
    batch = LearningBatch(
        pre_spikes=torch.ones((2,), dtype=torch.float32),
        post_spikes=torch.ones((2,), dtype=torch.float32),
        weights=torch.zeros((2,), dtype=torch.float32),
        modulators={ModulatorKind.DOPAMINE: torch.ones((2,), dtype=torch.float32)},
        extras={
            "active_edges": active_edges,
            "pre_idx": torch.tensor([0, 1], dtype=torch.long),
            "post_idx": torch.tensor([0, 1], dtype=torch.long),
        },
    )
    state, _ = rule.step(state, batch, dt=1e-3, t=0.0, ctx=ctx)

    changed = (state.eligibility - before).abs() > 0
    changed_idx = changed.nonzero(as_tuple=False).flatten()
    torch.testing.assert_close(changed_idx, active_edges)
