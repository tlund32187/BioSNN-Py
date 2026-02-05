from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit

import csv

from biosnn.contracts.monitors import StepEvent
from biosnn.monitors.metrics.metrics_csv import MetricsCSVMonitor


def test_metrics_csv_monitor(tmp_path):
    path = tmp_path / "metrics.csv"
    monitor = MetricsCSVMonitor(str(path))

    event = StepEvent(
        t=1.0,
        dt=0.001,
        spikes=None,
        scalars={
            "step": 5.0,
            "spike_count_total": 12.0,
            "spike_fraction_total": 0.25,
            "train_accuracy": 0.8,
            "loss": 0.1,
        },
    )

    monitor.on_step(event)
    monitor.close()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["train_accuracy"] == "0.8"
    assert rows[0]["loss"] == "0.1"
