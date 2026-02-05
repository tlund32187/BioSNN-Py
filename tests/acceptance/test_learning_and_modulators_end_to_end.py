from __future__ import annotations

import pytest

from biosnn.contracts.simulation import SimulationConfig
from tests.support.determinism import set_deterministic_cpu
from tests.support.scenarios import build_learning_gate_engine
from tests.support.tap_monitor import TapMonitor

pytestmark = pytest.mark.acceptance

STEPS = 15
LEARNING_STEP = 5
SEED = 123
DT = 1e-3
DTYPE = "float64"
DEVICE = "cpu"


def test_three_factor_hebbian_updates_only_with_dopamine():
    pytest.importorskip("torch")

    weight_off, active_edges_off = _run_learning_gate(
        dopamine_on=False,
        compiled_mode=False,
    )
    weight_on, active_edges_on = _run_learning_gate(
        dopamine_on=True,
        compiled_mode=False,
    )

    assert weight_off == 0.0, "Dopamine off should leave weight unchanged"
    assert weight_on == pytest.approx(0.1, abs=1e-12), (
        "Dopamine on should update weight by lr=0.1 when pre/post spike"
    )

    assert active_edges_on > 0, (
        "Expected non-empty active edge set at learning step when spikes occur"
    )
    assert active_edges_off > 0, (
        "Expected active edge set even when dopamine gate is off"
    )


def test_learning_compiled_matches_uncompiled():
    pytest.importorskip("torch")

    weight_uncompiled, _ = _run_learning_gate(
        dopamine_on=True,
        compiled_mode=False,
    )
    weight_compiled, _ = _run_learning_gate(
        dopamine_on=True,
        compiled_mode=True,
    )

    assert weight_compiled == pytest.approx(weight_uncompiled, abs=1e-12), (
        "Compiled and uncompiled learning should produce the same final weight"
    )


def _run_learning_gate(*, dopamine_on: bool, compiled_mode: bool) -> tuple[float, int]:
    engine, proj_name, _, _ = build_learning_gate_engine(
        dopamine_on=dopamine_on,
        compiled_mode=compiled_mode,
    )
    tap = TapMonitor((f"proj/{proj_name}/weights",), cpu_copy=True)
    active_edges = -1

    with set_deterministic_cpu(SEED):
        engine.reset(config=SimulationConfig(dt=DT, device=DEVICE, dtype=DTYPE, seed=SEED))
        engine.attach_monitors([tap])
        for step in range(STEPS):
            engine.step()
            if step == LEARNING_STEP:
                d_weights = engine.last_d_weights.get(proj_name)
                assert d_weights is not None, (
                    f"Missing d_weights for {proj_name} at learning step"
                )
                active_edges = int(d_weights.numel())

    event = engine._last_event
    assert event is not None and event.tensors, "Engine did not emit event tensors"
    weights = event.tensors.get(f"proj/{proj_name}/weights")
    assert weights is not None, f"Missing proj/{proj_name}/weights in event tensors"
    return float(weights[0].item()), active_edges
