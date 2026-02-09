from __future__ import annotations

import csv

import pytest

from biosnn.contracts.monitors import StepEvent
from biosnn.monitors.metrics.homeostasis_csv import HomeostasisCSVMonitor

pytestmark = pytest.mark.unit


def test_homeostasis_csv_monitor_writes_per_population_rows(tmp_path) -> None:
    path = tmp_path / "homeostasis.csv"
    monitor = HomeostasisCSVMonitor(str(path), stride=1, flush_every=1)

    monitor.on_step(
        StepEvent(
            t=0.1,
            dt=1e-3,
            scalars={
                "step": 10.0,
                "homeostasis/rate_mean/Input0": 0.12,
                "homeostasis/target_rate/Input0": 0.10,
                "homeostasis/control_mean/Input0": 0.01,
                "homeostasis/rate_mean/Output0": 0.08,
                "homeostasis/target_rate/Output0": 0.10,
                "homeostasis/control_mean/Output0": 0.02,
            },
        )
    )
    monitor.close()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    by_pop = {row["population"]: row for row in rows}
    assert by_pop["Input0"]["rate_mean"] == "0.12"
    assert by_pop["Output0"]["control_mean"] == "0.02"

