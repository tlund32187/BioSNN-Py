from __future__ import annotations

import csv
from pathlib import Path

import pytest

from biosnn.runners import cli

pytestmark = pytest.mark.acceptance


def test_cli_pruning_sparse_reduces_edge_count(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "pruning_sparse"
    args = cli._parse_args(
        [
            "--demo",
            "pruning_sparse",
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "80",
            "--prune-interval-steps",
            "10",
            "--prune-w-min",
            "0.06",
            "--prune-usage-min",
            "2.0",
            "--prune-max-fraction",
            "0.20",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    edge_count_path = run_dir / "edge_count.csv"
    assert edge_count_path.exists()
    assert edge_count_path.stat().st_size > 0

    edges_total: list[float] = []
    with edge_count_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("edges_total", "").strip()
            if value:
                edges_total.append(float(value))

    assert len(edges_total) >= 2
    assert min(edges_total) > 0.0
    assert min(edges_total) < max(edges_total)
