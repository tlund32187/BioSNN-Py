from __future__ import annotations

import csv

import pytest

from biosnn.contracts.monitors import StepEvent
from biosnn.experiments.demo_minimal import DemoMinimalConfig, run_demo_minimal
from biosnn.monitors.raster.spike_events_csv import SpikeEventsCSVMonitor

torch = pytest.importorskip("torch")


def test_default_monitors_no_cuda_sync(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist

    def _guard_item(self, *args, **kwargs):
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            raise RuntimeError("cuda item() not allowed in default monitors")
        return orig_item(self, *args, **kwargs)

    def _guard_tolist(self, *args, **kwargs):
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            raise RuntimeError("cuda tolist() not allowed in default monitors")
        return orig_tolist(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", _guard_item, raising=False)
    monkeypatch.setattr(torch.Tensor, "tolist", _guard_tolist, raising=False)

    out_dir = tmp_path / "demo_cuda"
    cfg = DemoMinimalConfig(
        out_dir=out_dir,
        mode="dashboard",
        n_neurons=8,
        p_connect=0.2,
        steps=2,
        dt=1e-3,
        device="cuda",
        allow_cuda_monitor_sync=False,
    )
    run_demo_minimal(cfg)


def test_spike_events_cuda_requires_opt_in(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    spikes = torch.tensor([True, False, True], device="cuda")
    event = StepEvent(t=0.0, dt=1e-3, spikes=spikes, scalars={"step": 0.0})

    path_disabled = tmp_path / "spikes_disabled.csv"
    monitor = SpikeEventsCSVMonitor(str(path_disabled), allow_cuda_sync=False)
    with pytest.warns(RuntimeWarning):
        monitor.on_step(event)
    monitor.close()
    with path_disabled.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []

    path_enabled = tmp_path / "spikes_enabled.csv"
    monitor_enabled = SpikeEventsCSVMonitor(str(path_enabled), allow_cuda_sync=True)
    monitor_enabled.on_step(event)
    monitor_enabled.close()

    with path_enabled.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
