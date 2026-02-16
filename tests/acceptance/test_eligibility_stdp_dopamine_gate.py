from __future__ import annotations

import pytest

from biosnn.contracts.simulation import SimulationConfig
from biosnn.learning import ThreeFactorEligibilityStdpParams, ThreeFactorEligibilityStdpRule
from tests.support.determinism import set_deterministic_cpu
from tests.support.scenarios import build_learning_gate_engine

pytestmark = pytest.mark.acceptance

STEPS = 15
LEARNING_STEP = 5
SEED = 123
DT = 1e-3
DTYPE = "float64"
DEVICE = "cpu"


def test_eligibility_stdp_updates_only_with_dopamine() -> None:
    pytest.importorskip("torch")

    weight_off, active_edges_off, eligibility_off = _run_learning_gate(
        dopamine_on=False,
        compiled_mode=False,
    )
    weight_on, active_edges_on, eligibility_on = _run_learning_gate(
        dopamine_on=True,
        compiled_mode=False,
    )

    assert active_edges_off > 0
    assert active_edges_on > 0
    assert eligibility_off > 0.0
    assert eligibility_on > 0.0

    assert weight_off == pytest.approx(0.0, abs=1e-12)
    assert weight_on != pytest.approx(0.0, abs=1e-12)  # dopamine gates learning
    assert weight_on > 0.0  # positive weight change with dopamine on


def test_eligibility_stdp_compiled_matches_uncompiled() -> None:
    pytest.importorskip("torch")

    weight_uncompiled, _, _ = _run_learning_gate(
        dopamine_on=True,
        compiled_mode=False,
    )
    weight_compiled, _, _ = _run_learning_gate(
        dopamine_on=True,
        compiled_mode=True,
    )

    assert weight_compiled == pytest.approx(weight_uncompiled, abs=1e-12)


def _run_learning_gate(*, dopamine_on: bool, compiled_mode: bool) -> tuple[float, int, float]:
    params = ThreeFactorEligibilityStdpParams(
        lr=0.1,
        tau_e=0.05,
        a_plus=1.0,
        a_minus=0.0,
        weight_decay=0.0,
        clamp_min=None,
        clamp_max=None,
    )
    engine, proj_name, _, _ = build_learning_gate_engine(
        dopamine_on=dopamine_on,
        compiled_mode=compiled_mode,
        learning_rule=ThreeFactorEligibilityStdpRule(params),
    )
    active_edges = -1
    eligibility_peak = 0.0

    with set_deterministic_cpu(SEED):
        engine.reset(config=SimulationConfig(dt=DT, device=DEVICE, dtype=DTYPE, seed=SEED))
        for step in range(STEPS):
            engine.step()
            learn_state = engine._proj_states[proj_name].learning_state
            assert learn_state is not None
            eligibility = learn_state.eligibility
            eligibility_peak = max(eligibility_peak, float(eligibility.abs().max().item()))
            if step == LEARNING_STEP:
                d_weights = engine.last_d_weights.get(proj_name)
                assert d_weights is not None
                active_edges = int(d_weights.numel())

    weights = engine._proj_states[proj_name].state.weights
    return float(weights[0].item()), active_edges, eligibility_peak
