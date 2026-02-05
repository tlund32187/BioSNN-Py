from __future__ import annotations

import pytest

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.simulation import SimulationConfig
from tests.support.determinism import set_deterministic_cpu
from tests.support.scenarios import build_delay_impulse_engine

pytestmark = pytest.mark.acceptance

STEPS = 20
SEED = 123
DT = 1e-3
DTYPE = "float64"
DEVICE = "cpu"

DELAYS = (0, 1, 3, 5)


def test_delay_impulse_alignment_invariants():
    pytest.importorskip("torch")
    for delay in DELAYS:
        engine, proj_name, _ = build_delay_impulse_engine(
            delay_steps=delay,
            compiled_mode=False,
        )
        drive_series = _capture_post_drive(engine, proj_name, steps=STEPS)
        expected_step = _expected_drive_step(delay)
        _assert_drive_alignment(drive_series, expected_step, delay, mode="uncompiled")


def test_delay_impulse_compiled_matches_uncompiled():
    pytest.importorskip("torch")
    for delay in DELAYS:
        engine_uncompiled, proj_name, _ = build_delay_impulse_engine(
            delay_steps=delay,
            compiled_mode=False,
        )
        engine_compiled, proj_name_compiled, _ = build_delay_impulse_engine(
            delay_steps=delay,
            compiled_mode=True,
        )
        drive_uncompiled = _capture_post_drive(engine_uncompiled, proj_name, steps=STEPS)
        drive_compiled = _capture_post_drive(engine_compiled, proj_name_compiled, steps=STEPS)

        expected_step = _expected_drive_step(delay)
        _assert_drive_alignment(drive_uncompiled, expected_step, delay, mode="uncompiled")
        _assert_drive_alignment(drive_compiled, expected_step, delay, mode="compiled")

        uncompiled_idx = _first_nonzero_step(drive_uncompiled)
        compiled_idx = _first_nonzero_step(drive_compiled)
        assert uncompiled_idx == compiled_idx, (
            "Compiled vs uncompiled first-drive mismatch "
            f"for delay={delay}: uncompiled={uncompiled_idx} compiled={compiled_idx}"
        )


def _capture_post_drive(engine, proj_name: str, *, steps: int) -> list[float]:
    values: list[float] = []
    with set_deterministic_cpu(SEED):
        engine.reset(config=SimulationConfig(dt=DT, device=DEVICE, dtype=DTYPE, seed=SEED))
        for step in range(steps):
            engine.step()
            drive_map = engine.last_projection_drive.get(proj_name)
            assert drive_map is not None, (
                f"Missing projection drive for {proj_name} at step={step}"
            )
            drive = drive_map.get(Compartment.DENDRITE)
            assert drive is not None, (
                f"Missing dendrite drive for {proj_name} at step={step}"
            )
            values.append(float(drive[0].item()))
    return values


def _expected_drive_step(delay_steps: int) -> int:
    input_spike_step = 5
    synapse_reads_step = input_spike_step + 1
    return synapse_reads_step + delay_steps


def _first_nonzero_step(values: list[float]) -> int | None:
    for idx, value in enumerate(values):
        if value != 0.0:
            return idx
    return None


def _assert_drive_alignment(
    values: list[float],
    expected_step: int,
    delay_steps: int,
    *,
    mode: str,
) -> None:
    nonzero_steps = [idx for idx, value in enumerate(values) if value != 0.0]
    assert nonzero_steps, (
        f"No nonzero drive observed for delay={delay_steps} mode={mode}. "
        f"Expected first drive at step={expected_step}."
    )
    first_nonzero = nonzero_steps[0]
    assert first_nonzero == expected_step, (
        f"Drive arrived at step={first_nonzero} for delay={delay_steps} mode={mode}; "
        f"expected step={expected_step}."
    )

    for idx in range(expected_step):
        assert values[idx] == 0.0, (
            f"Drive nonzero before expected step for delay={delay_steps} mode={mode}. "
            f"step={idx} value={values[idx]} expected_step={expected_step}"
        )

    assert values[expected_step] > 0.0, (
        f"Drive not positive at expected step for delay={delay_steps} mode={mode}. "
        f"value={values[expected_step]} expected_step={expected_step}"
    )
