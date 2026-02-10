from __future__ import annotations

import csv
from pathlib import Path

import pytest

from biosnn.runners import cli

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize(
    ("demo_name", "extra_args"),
    [
        ("logic_and", ["--logic-learning-mode", "none"]),
        ("logic_or", ["--logic-learning-mode", "none"]),
        ("logic_xor", ["--logic-learning-mode", "none"]),
        (
            "logic_curriculum",
            [
                "--logic-curriculum-gates",
                "or,and,xor",
                "--logic-curriculum-replay-ratio",
                "0.5",
                "--logic-learning-mode",
                "rstdp",
            ],
        ),
    ],
)
def test_cli_logic_gate_demos_smoke(
    monkeypatch,
    tmp_path: Path,
    demo_name: str,
    extra_args: list[str],
) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / demo_name
    args = cli._parse_args(
        [
            "--demo",
            demo_name,
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "40",
            "--no-open",
            *extra_args,
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    required = (
        "run_config.json",
        "run_features.json",
        "run_status.json",
        "trials.csv",
        "eval.csv",
        "metrics.csv",
        "confusion.csv",
    )
    for name in required:
        path = run_dir / name
        assert path.exists(), f"Missing artifact {name} for demo {demo_name}"
        assert path.stat().st_size > 0, f"Empty artifact {name} for demo {demo_name}"

    if demo_name == "logic_curriculum":
        with (run_dir / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        train_gates = {str(row.get("train_gate", "")).strip() for row in rows}
        assert "or" in train_gates
        assert "and" in train_gates
        assert "xor" in train_gates
        with (run_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            metrics_rows = list(csv.DictReader(handle))
        assert metrics_rows
        assert any(str(row.get("sample_accuracy_global", "")).strip() for row in metrics_rows)
        assert any(str(row.get("global_eval_accuracy", "")).strip() for row in metrics_rows)
        last_metrics = metrics_rows[-1]
        assert str(last_metrics.get("eval_accuracy", "")).strip() == str(
            last_metrics.get("global_eval_accuracy", "")
        ).strip()
