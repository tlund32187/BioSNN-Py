from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biosnn.contracts.simulation import SimulationConfig
from tests.support.determinism import set_deterministic_cpu
from tests.support.scenarios import build_prop_chain_engine
from tests.support.tap_monitor import TapMonitor

pytestmark = pytest.mark.acceptance

STEPS = 20
SEED = 123
DT = 1e-3
DTYPE = "float64"
DEVICE = "cpu"

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden" / "prop_chain_v1.npz"


def test_prop_chain_matches_golden_cpu():
    if not GOLDEN_PATH.exists():
        raise AssertionError(
            "Missing golden snapshot: tests/golden/prop_chain_v1.npz. "
            "Generate it with: python -m tests.golden.generate_golden --name prop_chain --overwrite"
        )

    golden = _load_golden(GOLDEN_PATH)

    engine, tap_keys, _ = build_prop_chain_engine(compiled_mode=False)
    tap = TapMonitor(tap_keys, cpu_copy=True)
    with set_deterministic_cpu(SEED):
        engine.reset(config=SimulationConfig(dt=DT, device=DEVICE, dtype=DTYPE, seed=SEED))
        engine.attach_monitors([tap])
        engine.run(STEPS)

    series = tap.to_numpy_dict()

    spike_keys = (
        "pop/Input/spikes",
        "pop/Relay/spikes",
        "pop/Hidden/spikes",
        "pop/Out/spikes",
    )
    float_keys = (
        "pop/Relay/v_soma_raw",
        "pop/Relay/last_drive_dendrite",
        "pop/Hidden/v_soma_raw",
        "pop/Hidden/last_drive_dendrite",
        "pop/Out/v_soma_raw",
        "pop/Out/last_drive_dendrite",
    )

    for key in spike_keys:
        expected = _golden_array(golden, key)
        actual = series[key]
        assert np.array_equal(
            actual, expected
        ), f"{key} mismatch against golden snapshot"

    for key in float_keys:
        expected = _golden_array(golden, key)
        actual = series[key]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
            err_msg=f"{key} mismatch against golden snapshot",
        )

    _assert_stage_invariants(series)


def test_prop_chain_compiled_matches_uncompiled_cpu():
    uncompiled = _run_prop_chain(compiled_mode=False)
    compiled = _run_prop_chain(compiled_mode=True)

    for key in (
        "pop/Input/spikes",
        "pop/Relay/spikes",
        "pop/Hidden/spikes",
        "pop/Out/spikes",
    ):
        assert np.array_equal(
            uncompiled[key], compiled[key]
        ), f"{key} mismatch between compiled and uncompiled modes"

    for key in (
        "pop/Relay/v_soma_raw",
        "pop/Hidden/v_soma_raw",
        "pop/Out/v_soma_raw",
    ):
        np.testing.assert_allclose(
            uncompiled[key],
            compiled[key],
            rtol=0.0,
            atol=0.0,
            err_msg=f"{key} mismatch between compiled and uncompiled modes",
        )


def _run_prop_chain(*, compiled_mode: bool) -> dict[str, np.ndarray]:
    engine, tap_keys, _ = build_prop_chain_engine(compiled_mode=compiled_mode)
    tap = TapMonitor(tap_keys, cpu_copy=True)
    with set_deterministic_cpu(SEED):
        engine.reset(config=SimulationConfig(dt=DT, device=DEVICE, dtype=DTYPE, seed=SEED))
        engine.attach_monitors([tap])
        engine.run(STEPS)
    return {key: np.asarray(value) for key, value in tap.to_numpy_dict().items()}


def _load_golden(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _encode_key(key: str) -> str:
    return key.replace("/", "__")


def _golden_array(golden: dict[str, np.ndarray], key: str) -> np.ndarray:
    encoded = _encode_key(key)
    if encoded not in golden:
        raise AssertionError(f"Golden snapshot missing key: {encoded}")
    return golden[encoded]


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


def _assert_stage_invariants(series: dict[str, np.ndarray]) -> None:
    input_spikes = series["pop/Input/spikes"]
    relay_spikes = series["pop/Relay/spikes"]
    hidden_spikes = series["pop/Hidden/spikes"]
    out_spikes = series["pop/Out/spikes"]

    relay_drive = series["pop/Relay/last_drive_dendrite"]
    hidden_drive = series["pop/Hidden/last_drive_dendrite"]
    out_drive = series["pop/Out/last_drive_dendrite"]

    input_steps = _steps_with_true(input_spikes)
    assert input_steps == {5, 6, 7}, f"Input spikes missing or unexpected: {sorted(input_steps)}"

    relay_drive_steps = _steps_with_nonzero(relay_drive)
    expected_relay_drive = {step + 1 for step in input_steps}
    assert relay_drive_steps == expected_relay_drive, (
        "Relay drive does not match expected latency from input. "
        f"expected={sorted(expected_relay_drive)} got={sorted(relay_drive_steps)}"
    )

    relay_steps = _steps_with_true(relay_spikes)
    assert relay_steps == relay_drive_steps, (
        "Relay spikes do not align with relay drive. "
        f"drive={sorted(relay_drive_steps)} spikes={sorted(relay_steps)}"
    )

    hidden_drive_steps = _steps_with_nonzero(hidden_drive)
    expected_hidden_drive = {step + 1 for step in relay_steps}
    assert hidden_drive_steps == expected_hidden_drive, (
        "Hidden drive does not match expected latency from relay. "
        f"expected={sorted(expected_hidden_drive)} got={sorted(hidden_drive_steps)}"
    )

    hidden_steps = _steps_with_true(hidden_spikes)
    assert hidden_steps == hidden_drive_steps, (
        "Hidden spikes do not align with hidden drive. "
        f"drive={sorted(hidden_drive_steps)} spikes={sorted(hidden_steps)}"
    )

    out_drive_steps = _steps_with_nonzero(out_drive)
    expected_out_drive = {step + 1 for step in hidden_steps}
    assert out_drive_steps == expected_out_drive, (
        "Out drive does not match expected latency from hidden. "
        f"expected={sorted(expected_out_drive)} got={sorted(out_drive_steps)}"
    )

    out_steps = _steps_with_true(out_spikes)
    assert out_steps == out_drive_steps, (
        "Out spikes do not align with out drive. "
        f"drive={sorted(out_drive_steps)} spikes={sorted(out_steps)}"
    )

    first_input = min(input_steps)
    first_relay = min(relay_steps)
    first_hidden = min(hidden_steps)
    first_out = min(out_steps)
    assert first_input < first_relay < first_hidden < first_out, (
        "Stage latency violated. "
        f"input={first_input} relay={first_relay} hidden={first_hidden} out={first_out}"
    )
