from __future__ import annotations

import math

import pytest

from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.neurons import StepContext
from biosnn.contracts.tensor import Tensor
from biosnn.learning.rules import (
    RStdpEligibilityParams,
    RStdpEligibilityRule,
    ThreeFactorEligibilityStdpParams,
    ThreeFactorEligibilityStdpRule,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _sparse_batch(active_edges: Tensor) -> LearningBatch:
    n = int(active_edges.numel())
    return LearningBatch(
        pre_spikes=torch.zeros((n,), dtype=torch.float32),
        post_spikes=torch.zeros((n,), dtype=torch.float32),
        weights=torch.zeros((n,), dtype=torch.float32),
        extras={"active_edges": active_edges},
    )


def test_rstdp_lazy_decay_only_active_edges() -> None:
    rule = RStdpEligibilityRule(
        RStdpEligibilityParams(
            tau_e=10.0,
            lazy_decay_enabled=True,
            lazy_decay_step_dtype="int32",
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    state = rule.init_state(5, ctx=ctx)
    assert state.last_update_step is not None

    state.eligibility.fill_(1.0)
    state.last_update_step.zero_()
    state.step = 10

    active_edges = torch.tensor([2], dtype=torch.long)
    state, _ = rule.step(state, _sparse_batch(active_edges), dt=1.0, t=0.0, ctx=ctx)

    expected = math.exp(-1.0)
    assert float(state.eligibility[2].item()) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    for idx in (0, 1, 3, 4):
        assert float(state.eligibility[idx].item()) == pytest.approx(1.0, rel=0.0, abs=0.0)
    last_update_step = state.last_update_step
    assert last_update_step is not None
    assert int(last_update_step[2].item()) == 10
    for idx in (0, 1, 3, 4):
        assert int(last_update_step[idx].item()) == 0
    assert state.step == 11


def test_rstdp_lazy_decay_reapplies_on_next_activation() -> None:
    rule = RStdpEligibilityRule(
        RStdpEligibilityParams(
            tau_e=10.0,
            lazy_decay_enabled=True,
            lazy_decay_step_dtype="int32",
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    state = rule.init_state(5, ctx=ctx)
    assert state.last_update_step is not None

    state.eligibility.fill_(1.0)
    state.last_update_step.zero_()
    state.step = 10
    active_edges = torch.tensor([2], dtype=torch.long)

    state, _ = rule.step(state, _sparse_batch(active_edges), dt=1.0, t=0.0, ctx=ctx)
    state, _ = rule.step(state, _sparse_batch(active_edges), dt=1.0, t=1.0, ctx=ctx)

    expected = math.exp(-1.0) * math.exp(-0.1)
    assert float(state.eligibility[2].item()) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    last_update_step = state.last_update_step
    assert last_update_step is not None
    assert int(last_update_step[2].item()) == 11
    assert state.step == 12


def test_three_factor_lazy_decay_smoke() -> None:
    rule = ThreeFactorEligibilityStdpRule(
        ThreeFactorEligibilityStdpParams(
            tau_e=10.0,
            lazy_decay_enabled=True,
            lazy_decay_step_dtype="int32",
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    state = rule.init_state(4, ctx=ctx)
    assert state.last_update_step is not None

    state.eligibility.fill_(1.0)
    state.last_update_step.zero_()
    state.step = 10
    active_edges = torch.tensor([1], dtype=torch.long)

    state, _ = rule.step(state, _sparse_batch(active_edges), dt=1.0, t=0.0, ctx=ctx)

    assert float(state.eligibility[1].item()) == pytest.approx(math.exp(-1.0), rel=1e-6, abs=1e-6)
    assert float(state.eligibility[0].item()) == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert float(state.eligibility[2].item()) == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert float(state.eligibility[3].item()) == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_rstdp_lazy_disabled_falls_back_to_full_decay() -> None:
    rule = RStdpEligibilityRule(
        RStdpEligibilityParams(
            tau_e=10.0,
            lazy_decay_enabled=False,
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    state = rule.init_state(5, ctx=ctx)

    state.eligibility.fill_(1.0)
    state.step = 10
    active_edges = torch.tensor([2], dtype=torch.long)
    state, _ = rule.step(state, _sparse_batch(active_edges), dt=1.0, t=0.0, ctx=ctx)

    expected = torch.full((5,), math.exp(-0.1), dtype=torch.float32)
    torch.testing.assert_close(state.eligibility, expected, rtol=1e-6, atol=1e-6)
