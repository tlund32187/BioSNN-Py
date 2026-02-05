from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
pytestmark = [pytest.mark.acceptance, pytest.mark.cuda]


from biosnn.contracts.simulation import SimulationConfig
from tests.support.scenarios import (
    build_delay_impulse_engine,
    build_learning_gate_engine,
    build_prop_chain_engine,
)
from tests.support.tap_monitor import TapMonitor

torch = pytest.importorskip("torch")

SEED = 123
DT = 1e-3
DTYPE = "float32"
DEVICE = "cuda:0"


def has_cuda() -> bool:
    return bool(torch.cuda.is_available())


def run_scenario_on_device(
    build_fn: Callable[[], tuple[object, tuple[str, ...]]],
    *,
    device: str,
    steps: int,
    dtype: str,
) -> dict[str, np.ndarray]:
    engine, tap_keys = build_fn()
    tap = TapMonitor(tap_keys, cpu_copy=True)
    engine.reset(config=SimulationConfig(dt=DT, device=device, dtype=dtype, seed=SEED))
    engine.attach_monitors([tap])
    engine.run(steps)
    return {key: np.asarray(value) for key, value in tap.to_numpy_dict().items()}


def test_cuda_prop_chain_invariants():
    if not has_cuda():
        pytest.skip("CUDA not available")

    def _build():
        engine, tap_keys, _ = build_prop_chain_engine(
            compiled_mode=True,
            device=DEVICE,
            dtype=torch.float32,
        )
        return engine, tap_keys

    data = run_scenario_on_device(_build, device=DEVICE, steps=50, dtype=DTYPE)

    input_spikes = data["pop/Input/spikes"]
    relay_spikes = data["pop/Relay/spikes"]
    hidden_spikes = data["pop/Hidden/spikes"]
    out_spikes = data["pop/Out/spikes"]

    relay_drive = data["pop/Relay/last_drive_dendrite"]
    hidden_drive = data["pop/Hidden/last_drive_dendrite"]
    out_drive = data["pop/Out/last_drive_dendrite"]

    expected_input_count = 3 * input_spikes.shape[1]
    assert int(input_spikes.sum()) == expected_input_count, (
        "Unexpected input spike count on CUDA"
    )

    input_steps = _steps_with_true(input_spikes)
    relay_steps = _steps_with_true(relay_spikes)
    assert relay_steps, "Relay spikes never occurred on CUDA"
    first_input = min(input_steps)
    first_relay = min(relay_steps)
    assert first_relay >= first_input, "Relay spiked before input on CUDA"
    assert first_relay <= first_input + 2, "Relay latency too large on CUDA"

    hidden_drive_steps = _steps_with_nonzero(hidden_drive)
    assert hidden_drive_steps, "Hidden drive never occurred on CUDA"
    assert min(hidden_drive_steps) > first_relay, "Hidden drive did not follow relay spikes"

    hidden_steps = _steps_with_true(hidden_spikes)
    assert hidden_steps, "Hidden spikes never occurred on CUDA"

    out_steps = _steps_with_true(out_spikes)
    assert out_steps, "Output spikes never occurred on CUDA"
    assert min(out_steps) > min(hidden_steps), "Output spikes did not follow hidden spikes"

    for key in ("pop/Relay/v_soma_raw", "pop/Hidden/v_soma_raw", "pop/Out/v_soma_raw"):
        assert not np.isnan(data[key]).any(), f"NaNs detected in {key} on CUDA"


def test_cuda_delay_first_arrival():
    if not has_cuda():
        pytest.skip("CUDA not available")

    for delay in (0, 1, 3, 5):
        def _build(delay_steps: int = delay):
            engine, proj_name, _ = build_delay_impulse_engine(
                delay_steps=delay_steps,
                compiled_mode=True,
                device=DEVICE,
                dtype=torch.float32,
            )
            tap_keys = ("pop/Post/last_drive_dendrite",)
            return engine, tap_keys

        data = run_scenario_on_device(_build, device=DEVICE, steps=30, dtype=DTYPE)
        drive = np.asarray(data["pop/Post/last_drive_dendrite"])
        drive = drive.reshape(drive.shape[0], -1)
        expected_step = 6 + delay

        first_nonzero = _first_nonzero_step(drive)
        assert first_nonzero == expected_step, (
            f"CUDA delay={delay} arrived at step={first_nonzero}, "
            f"expected step={expected_step}"
        )

        assert np.allclose(drive[:expected_step], 0.0), (
            f"CUDA delay={delay} had nonzero drive before step={expected_step}"
        )
        assert drive[expected_step].max() > 0.0, (
            f"CUDA delay={delay} drive not positive at expected step={expected_step}"
        )


def test_cuda_learning_gate_invariants():
    if not has_cuda():
        pytest.skip("CUDA not available")

    def _run(dopamine_on: bool) -> float:
        engine, proj_name, _, _ = build_learning_gate_engine(
            dopamine_on=dopamine_on,
            compiled_mode=True,
            device=DEVICE,
            dtype=torch.float32,
        )
        tap_keys = (f"proj/{proj_name}/weights",)
        data = run_scenario_on_device(
            lambda: (engine, tap_keys),
            device=DEVICE,
            steps=20,
            dtype=DTYPE,
        )
        weights = np.asarray(data[f"proj/{proj_name}/weights"])
        return float(weights[-1][0])

    weight_on = _run(dopamine_on=True)
    assert weight_on > 0.0, "CUDA dopamine_on should increase weight"
    assert abs(weight_on - 0.1) < 1e-2, "CUDA dopamine_on weight outside expected tolerance"

    weight_off = _run(dopamine_on=False)
    assert abs(weight_off) < 1e-6, "CUDA dopamine_off should keep weight near zero"


def _steps_with_true(values: np.ndarray) -> set[int]:
    arr = np.asarray(values).astype(bool)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return set(np.where(arr.any(axis=1))[0].tolist())


def _steps_with_nonzero(values: np.ndarray) -> set[int]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return set(np.where(np.any(arr != 0, axis=1))[0].tolist())


def _first_nonzero_step(values: np.ndarray) -> int | None:
    for idx, row in enumerate(values):
        if np.any(row != 0):
            return idx
    return None
