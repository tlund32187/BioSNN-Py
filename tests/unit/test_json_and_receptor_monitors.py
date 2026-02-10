from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from biosnn.contracts.monitors import StepEvent
from biosnn.monitors.modulators.modulator_grid_json_monitor import ModulatorGridJsonMonitor
from biosnn.monitors.synapses.receptor_summary_csv import ReceptorSummaryCsvMonitor
from biosnn.monitors.vision.vision_frame_json_monitor import VisionFrameJsonMonitor

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_modulator_grid_json_monitor_writes_downsampled_grid(tmp_path: Path) -> None:
    path = tmp_path / "modgrid.json"
    monitor = ModulatorGridJsonMonitor(path, every_n_steps=1, max_side=64, allow_cuda_sync=True)
    monitor.on_step(
        StepEvent(
            t=0.1,
            dt=1e-3,
            tensors={"mod/dopamine/grid": torch.ones((2, 128, 32), dtype=torch.float32)},
            scalars={"step": 7},
        )
    )
    monitor.close()

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["step"] == 7
    assert payload["available"] is True
    grid = payload["grids"]["dopamine"]
    assert isinstance(grid, list)
    assert len(grid) == 2
    assert len(grid[0]) <= 64
    assert len(grid[0][0]) <= 64


def test_modulator_grid_json_monitor_respects_max_elements(tmp_path: Path) -> None:
    path = tmp_path / "modgrid_limited.json"
    monitor = ModulatorGridJsonMonitor(
        path,
        every_n_steps=1,
        max_side=256,
        max_elements=256,
        allow_cuda_sync=True,
    )
    monitor.on_step(
        StepEvent(
            t=0.1,
            dt=1e-3,
            tensors={"mod/dopamine/grid": torch.ones((2, 128, 128), dtype=torch.float32)},
            scalars={"step": 8},
        )
    )
    monitor.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    grid = payload["grids"]["dopamine"]
    channels = len(grid)
    h = len(grid[0]) if channels else 0
    w = len(grid[0][0]) if h else 0
    assert channels * h * w <= 256


def test_receptor_summary_csv_monitor_writes_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "receptors.csv"
    monitor = ReceptorSummaryCsvMonitor(path, stride=1, include_max=True, allow_cuda_sync=True)
    monitor.on_step(
        StepEvent(
            t=0.2,
            dt=1e-3,
            tensors={
                "proj/P/receptors/dendrite/ampa_mean": torch.tensor([0.25], dtype=torch.float32),
                "proj/P/receptors/dendrite/nmda_mean": torch.tensor([0.10], dtype=torch.float32),
                "proj/P/receptors/dendrite/gabaa_mean": torch.tensor([0.05], dtype=torch.float32),
                "proj/P/receptors/dendrite/gabab_mean": torch.tensor([0.02], dtype=torch.float32),
                "proj/P/receptors/dendrite/ampa_max": torch.tensor([0.30], dtype=torch.float32),
                "proj/P/receptors/dendrite/nmda_max": torch.tensor([0.11], dtype=torch.float32),
                "proj/P/receptors/dendrite/gabaa_max": torch.tensor([0.06], dtype=torch.float32),
                "proj/P/receptors/dendrite/gabab_max": torch.tensor([0.03], dtype=torch.float32),
            },
            scalars={"step": 4},
        )
    )
    monitor.close()

    assert path.exists()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["proj"] == "P"
    assert row["comp"] == "dendrite"
    assert float(row["ampa_mean"]) == pytest.approx(0.25, abs=1e-6)
    assert float(row["nmda_mean"]) == pytest.approx(0.10, abs=1e-6)
    assert float(row["gabaa_mean"]) == pytest.approx(0.05, abs=1e-6)
    assert float(row["gabab_mean"]) == pytest.approx(0.02, abs=1e-6)
    assert float(row["ampa_max"]) == pytest.approx(0.30, abs=1e-6)


def test_vision_frame_json_monitor_writes_frame_payload(tmp_path: Path) -> None:
    path = tmp_path / "vision.json"
    monitor = VisionFrameJsonMonitor(
        path,
        tensor_key="vision/frame",
        every_n_steps=1,
        max_side=64,
        output_dtype="uint8",
        allow_cuda_sync=True,
    )
    monitor.on_step(
        StepEvent(
            t=0.3,
            dt=1e-3,
            tensors={"vision/frame": torch.linspace(0.0, 1.0, steps=120 * 80).reshape(1, 120, 80)},
            scalars={"step": 9},
        )
    )
    monitor.close()

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["step"] == 9
    assert payload["available"] is True
    assert payload["dtype"] == "uint8"
    shape = payload["shape"]
    assert isinstance(shape, list)
    assert len(shape) == 3
    assert shape[1] <= 64
    assert shape[2] <= 64


def test_vision_frame_json_monitor_respects_max_elements(tmp_path: Path) -> None:
    path = tmp_path / "vision_limited.json"
    monitor = VisionFrameJsonMonitor(
        path,
        tensor_key="vision/frame",
        every_n_steps=1,
        max_side=256,
        max_elements=512,
        output_dtype="uint8",
        allow_cuda_sync=True,
    )
    monitor.on_step(
        StepEvent(
            t=0.3,
            dt=1e-3,
            tensors={"vision/frame": torch.linspace(0.0, 1.0, steps=256 * 256).reshape(1, 256, 256)},
            scalars={"step": 10},
        )
    )
    monitor.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    shape = payload["shape"]
    assert shape[0] * shape[1] * shape[2] <= 512
