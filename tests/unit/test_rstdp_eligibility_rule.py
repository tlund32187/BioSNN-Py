from __future__ import annotations

import pytest

from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.neurons import StepContext
from biosnn.learning.rules import RStdpEligibilityParams, RStdpEligibilityRule

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_rstdp_eligibility_state_shape() -> None:
    rule = RStdpEligibilityRule()
    state = rule.init_state(8, ctx=StepContext(device="cpu", dtype="float32"))
    assert state.eligibility.shape == (8,)


def test_rstdp_eligibility_dopamine_gates_updates() -> None:
    params = RStdpEligibilityParams(
        lr=0.1,
        tau_e=0.05,
        a_plus=1.0,
        a_minus=0.0,
        weight_decay=0.0,
        dopamine_scale=1.0,
        baseline=0.0,
    )
    rule = RStdpEligibilityRule(params)
    state = rule.init_state(4, ctx=StepContext(device="cpu", dtype="float32"))

    weights = torch.zeros((4,), dtype=torch.float32)
    pre = torch.ones((4,), dtype=torch.float32)
    post = torch.ones((4,), dtype=torch.float32)

    batch_no_dop = LearningBatch(pre_spikes=pre, post_spikes=post, weights=weights)
    state, result_no_dop = rule.step(state, batch_no_dop, dt=1e-3, t=0.0, ctx=StepContext())
    assert float(result_no_dop.d_weights.abs().sum().item()) == pytest.approx(0.0)

    dopamine = torch.full((4,), 1.0, dtype=torch.float32)
    batch_dop = LearningBatch(
        pre_spikes=pre,
        post_spikes=post,
        weights=weights,
        modulators={ModulatorKind.DOPAMINE: dopamine},
    )
    state, result_dop = rule.step(state, batch_dop, dt=1e-3, t=1e-3, ctx=StepContext())
    assert float(result_dop.d_weights.mean().item()) > 0.0
