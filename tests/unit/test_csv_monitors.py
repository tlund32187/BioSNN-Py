import csv
import uuid
from pathlib import Path

import pytest

from biosnn.contracts.monitors import StepEvent
from biosnn.monitors.csv import AdEx2CompCSVMonitor, GLIFCSVMonitor, NeuronCSVMonitor


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _artifact_dir() -> Path:
    base = Path.cwd() / ".pytest_artifacts" / "csv_monitors"
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / uuid.uuid4().hex
    run_dir.mkdir()
    return run_dir


def test_neuron_csv_monitor_writes_stats() -> None:
    path = _artifact_dir() / "neurons.csv"
    monitor = NeuronCSVMonitor(path)

    event = StepEvent(
        t=0.1,
        dt=0.01,
        spikes=[0, 1, 1],
        tensors={"v": [1.0, 2.0, 3.0], "w": [0.0, 0.5, 1.0]},
        scalars={"loss": 1.25},
    )
    monitor.on_step(event)
    monitor.close()

    rows = _read_rows(path)
    assert len(rows) == 1
    row = rows[0]

    assert float(row["t"]) == pytest.approx(0.1)
    assert float(row["dt"]) == pytest.approx(0.01)
    assert float(row["spike_count"]) == pytest.approx(2.0)
    assert float(row["spike_fraction"]) == pytest.approx(2.0 / 3.0)
    assert float(row["spike_rate_hz"]) == pytest.approx((2.0 / 3.0) / 0.01)
    assert float(row["loss"]) == pytest.approx(1.25)
    assert float(row["v_mean"]) == pytest.approx(2.0)
    assert float(row["v_min"]) == pytest.approx(1.0)
    assert float(row["v_max"]) == pytest.approx(3.0)


def test_glif_csv_monitor_preset() -> None:
    path = _artifact_dir() / "glif.csv"
    monitor = GLIFCSVMonitor(path, sample_indices=[1], stats=("mean",))

    event = StepEvent(
        t=0.0,
        dt=0.001,
        spikes=[0, 1, 0],
        tensors={
            "v_soma": [-0.07, -0.05, -0.06],
            "refrac_left": [0.0, 0.0, 0.002],
            "spike_hold_left": [0.0, 0.001, 0.0],
            "theta": [0.0, 0.003, 0.0],
        },
    )
    monitor.on_step(event)
    monitor.close()

    row = _read_rows(path)[0]
    assert "v_soma_mean" in row
    assert float(row["v_soma_i1"]) == pytest.approx(-0.05)
    assert float(row["theta_i1"]) == pytest.approx(0.003)


def test_adex_csv_monitor_preset() -> None:
    path = _artifact_dir() / "adex.csv"
    monitor = AdEx2CompCSVMonitor(path, stats=("mean",))

    event = StepEvent(
        t=0.2,
        dt=0.001,
        spikes=[1, 0],
        tensors={
            "v_soma": [-0.06, -0.07],
            "v_dend": [-0.05, -0.06],
            "w": [0.0, 0.1],
            "refrac_left": [0.0, 0.0],
            "spike_hold_left": [0.001, 0.0],
        },
    )
    monitor.on_step(event)
    monitor.close()

    row = _read_rows(path)[0]
    assert float(row["v_soma_mean"]) == pytest.approx(-0.065)
    assert float(row["v_dend_mean"]) == pytest.approx(-0.055)
    assert float(row["w_mean"]) == pytest.approx(0.05)
