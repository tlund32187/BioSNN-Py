from __future__ import annotations

import csv

import pytest

from biosnn.contracts.monitors import StepEvent
from biosnn.contracts.synapses import SynapseTopology
from biosnn.monitors.weights.projection_weights_csv import ProjectionWeightsCSVMonitor

torch = pytest.importorskip("torch")


def test_projection_weights_csv_monitor(tmp_path):
    path = tmp_path / "weights.csv"
    topology = SynapseTopology(
        pre_idx=torch.tensor([0, 1], dtype=torch.long),
        post_idx=torch.tensor([1, 0], dtype=torch.long),
    )
    projections = [{"name": "A_to_B", "topology": topology}]

    monitor = ProjectionWeightsCSVMonitor(
        str(path),
        projections=projections,
        max_edges_full=1,
        max_edges_sample=1,
    )

    event = StepEvent(
        t=0.0,
        dt=0.001,
        tensors={"proj/A_to_B/weights": torch.tensor([0.5, -0.25])},
        scalars={"step": 0.0},
    )

    monitor.on_step(event)
    monitor.close()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert all(row["proj"] == "A_to_B" for row in rows)
    assert len(rows) <= 1
