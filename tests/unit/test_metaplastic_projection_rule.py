from __future__ import annotations

import pytest

from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.neurons import StepContext
from biosnn.learning.rules import (
    EligibilityTraceHebbianParams,
    EligibilityTraceHebbianRule,
    MetaplasticProjectionParams,
    MetaplasticProjectionRule,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _base_rule(*, lr: float = 0.5) -> EligibilityTraceHebbianRule:
    return EligibilityTraceHebbianRule(
        EligibilityTraceHebbianParams(
            lr=lr,
            enable_eligibility=False,
            weight_decay=0.0,
        )
    )


def _batch(size: int, *, reward: float | None = None) -> LearningBatch:
    meta = None if reward is None else {"reward": reward}
    return LearningBatch(
        pre_spikes=torch.ones((size,), dtype=torch.float32),
        post_spikes=torch.ones((size,), dtype=torch.float32),
        weights=torch.zeros((size,), dtype=torch.float32),
        meta=meta,
    )


def test_metaplastic_projection_default_off_passthrough() -> None:
    base = _base_rule(lr=0.25)
    rule = MetaplasticProjectionRule(
        base,
        MetaplasticProjectionParams(
            enabled=False,
            eta_init=1.0,
            eta_min=0.2,
            eta_max=1.5,
        ),
    )
    state = rule.init_state(5, ctx=StepContext(device="cpu", dtype="float32"))
    state, result = rule.step(state, _batch(5), dt=1e-3, t=0.0, ctx=StepContext())
    assert float(state.eta) == pytest.approx(1.0)
    assert float(result.d_weights.mean()) == pytest.approx(0.25)


def test_metaplastic_projection_eta_stays_bounded_under_reward_noise() -> None:
    base = _base_rule(lr=0.1)
    params = MetaplasticProjectionParams(
        enabled=True,
        reward_beta=0.1,
        variance_gain=3.0,
        eta_init=1.0,
        eta_min=0.3,
        eta_max=1.6,
    )
    rule = MetaplasticProjectionRule(base, params)
    state = rule.init_state(8, ctx=StepContext(device="cpu", dtype="float32"))

    rewards = [1.0, -1.0, 0.5, -0.5] * 32
    for idx, reward in enumerate(rewards):
        state, result = rule.step(
            state,
            _batch(8, reward=reward),
            dt=1e-3,
            t=idx * 1e-3,
            ctx=StepContext(),
        )
        assert torch.isfinite(state.eta)
        assert float(state.eta) >= params.eta_min
        assert float(state.eta) <= params.eta_max
        assert torch.isfinite(result.d_weights).all()

    assert float(state.reward_var) > 0.0
    assert float(state.eta) < 1.0
