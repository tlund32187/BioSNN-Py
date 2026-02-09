from __future__ import annotations

import pytest

from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.neurons import StepContext
from biosnn.learning.rules import EligibilityTraceHebbianParams, EligibilityTraceHebbianRule

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_eligibility_trace_state_shape() -> None:
    rule = EligibilityTraceHebbianRule()
    state = rule.init_state(10, ctx=StepContext(device="cpu", dtype="float32"))
    assert state.eligibility.shape == (10,)
    assert state.eligibility_scale.shape == ()


def test_eligibility_trace_sparse_updates_only_active_edges() -> None:
    rule = EligibilityTraceHebbianRule(
        EligibilityTraceHebbianParams(
            lr=0.1,
            enable_eligibility=True,
            tau_e=0.05,
            a_plus=1.0,
            a_minus=0.0,
            weight_decay=0.0,
        )
    )
    state = rule.init_state(6, ctx=StepContext(device="cpu", dtype="float32"))
    active_edges = torch.tensor([1, 4], dtype=torch.long)
    batch = LearningBatch(
        pre_spikes=torch.ones((2,), dtype=torch.float32),
        post_spikes=torch.ones((2,), dtype=torch.float32),
        weights=torch.zeros((2,), dtype=torch.float32),
        extras={
            "active_edges": active_edges,
            "pre_idx": active_edges,
            "post_idx": active_edges,
        },
    )
    state, result = rule.step(state, batch, dt=1e-3, t=0.0, ctx=StepContext())

    assert result.d_weights.shape == (2,)
    nz = state.eligibility.nonzero(as_tuple=False).flatten()
    torch.testing.assert_close(nz, active_edges)
    assert float(result.d_weights.abs().sum()) > 0.0


def test_eligibility_trace_default_off_keeps_trace_zero() -> None:
    rule = EligibilityTraceHebbianRule(
        EligibilityTraceHebbianParams(
            lr=0.2,
            enable_eligibility=False,
            weight_decay=0.0,
        )
    )
    state = rule.init_state(4, ctx=StepContext(device="cpu", dtype="float32"))
    batch = LearningBatch(
        pre_spikes=torch.ones((4,), dtype=torch.float32),
        post_spikes=torch.ones((4,), dtype=torch.float32),
        weights=torch.zeros((4,), dtype=torch.float32),
    )
    state, result = rule.step(state, batch, dt=1e-3, t=0.0, ctx=StepContext())
    assert float(state.eligibility.abs().sum()) == pytest.approx(0.0)
    assert float(result.d_weights.mean()) > 0.0
