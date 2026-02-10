from __future__ import annotations

import csv
from pathlib import Path

import pytest

from biosnn.runners import cli

pytestmark = pytest.mark.acceptance


def test_cli_neurogenesis_sparse_grows_with_cap(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "neurogenesis_sparse"
    args = cli._parse_args(
        [
            "--demo",
            "neurogenesis_sparse",
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "80",
            "--growth-interval-steps",
            "2",
            "--add-neurons-per-event",
            "2",
            "--max-total-neurons",
            "112",
            "--input-drive",
            "0.0",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    neuro_path = run_dir / "neurogenesis.csv"
    assert neuro_path.exists()
    assert neuro_path.stat().st_size > 0

    totals: list[float] = []
    grew_values: list[float] = []
    events_total: list[float] = []
    with neuro_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total = row.get("total_neurons", "").strip()
            if total:
                totals.append(float(total))
            grew = row.get("grew", "").strip()
            if grew:
                grew_values.append(float(grew))
            events = row.get("events_total", "").strip()
            if events:
                events_total.append(float(events))

    assert len(totals) >= 2
    assert max(totals) > min(totals)
    assert max(totals) <= 112.0
    assert max(events_total, default=0.0) >= 1.0
