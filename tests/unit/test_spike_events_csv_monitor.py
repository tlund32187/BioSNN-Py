from __future__ import annotations

import csv

import pytest
pytestmark = pytest.mark.unit


from biosnn.contracts.monitors import StepEvent
from biosnn.monitors.raster.spike_events_csv import SpikeEventsCSVMonitor

torch = pytest.importorskip("torch")


def test_spike_events_csv_monitor(tmp_path):
    path = tmp_path / "spikes.csv"
    monitor = SpikeEventsCSVMonitor(str(path))

    spikes = torch.tensor([1.0, 0.0, 0.0, 1.0])
    meta = {"population_slices": {"A": (0, 2), "B": (2, 4)}}
    event = StepEvent(t=0.0, dt=0.001, spikes=spikes, scalars={"step": 0.0}, meta=meta)

    monitor.on_step(event)
    monitor.close()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["pop"] == "A"
    assert rows[0]["neuron"] == "0"
    assert rows[1]["pop"] == "B"
    assert rows[1]["neuron"] == "1"
