from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from biosnn.runners import cli

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize(
    ("demo_name", "logic_backend", "extra_args"),
    [
        ("logic_and", "harness", ["--logic-learning-mode", "none"]),
        ("logic_and", "engine", ["--logic-learning-mode", "none"]),
        ("logic_or", "harness", ["--logic-learning-mode", "none"]),
        ("logic_or", "engine", ["--logic-learning-mode", "none"]),
        ("logic_xor", "harness", ["--logic-learning-mode", "none"]),
        ("logic_xor", "engine", ["--logic-learning-mode", "none"]),
        (
            "logic_curriculum",
            "harness",
            [
                "--logic-curriculum-gates",
                "or,and,xor",
                "--logic-curriculum-replay-ratio",
                "0.5",
                "--logic-learning-mode",
                "rstdp",
            ],
        ),
        (
            "logic_curriculum",
            "engine",
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
    logic_backend: str,
    extra_args: list[str],
) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / demo_name
    steps = "200" if demo_name == "logic_curriculum" else "40"
    args = cli._parse_args(
        [
            "--demo",
            demo_name,
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            steps,
            "--no-open",
            "--logic-backend",
            logic_backend,
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
        "topology.json",
        "trials.csv",
        "eval.csv",
        "metrics.csv",
        "confusion.csv",
    )
    for name in required:
        path = run_dir / name
        assert path.exists(), f"Missing artifact {name} for demo {demo_name}"
        assert path.stat().st_size > 0, f"Empty artifact {name} for demo {demo_name}"
    with (run_dir / "run_features.json").open("r", encoding="utf-8") as handle:
        run_features = json.load(handle)
    assert run_features.get("logic_backend") == logic_backend
    with (run_dir / "topology.json").open("r", encoding="utf-8") as handle:
        topology_payload = json.load(handle)
    node_models = {
        str(node.get("model", "")).strip().lower()
        for node in topology_payload.get("nodes", [])
        if isinstance(node, dict)
    }
    assert "adex_3c" in node_models

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


def test_cli_logic_curriculum_engine_can_disable_learning(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "logic_curriculum_engine_no_learning"
    args = cli._parse_args(
        [
            "--demo",
            "logic_curriculum",
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "80",
            "--logic-backend",
            "engine",
            "--logic-learning-mode",
            "none",
            "--logic-curriculum-gates",
            "or,and,xor",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    with (run_dir / "run_config.json").open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    with (run_dir / "run_features.json").open("r", encoding="utf-8") as handle:
        run_features = json.load(handle)

    assert run_config.get("logic_backend") == "engine"
    assert bool(run_config.get("learning", {}).get("enabled")) is False
    assert run_config.get("logic_learning_mode") == "none"
    assert bool(run_features.get("learning", {}).get("enabled")) is False


def test_cli_logic_curriculum_harness_falls_back_to_engine_when_learning_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "logic_curriculum_harness_learning_none"
    args = cli._parse_args(
        [
            "--demo",
            "logic_curriculum",
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "80",
            "--logic-backend",
            "harness",
            "--logic-learning-mode",
            "none",
            "--logic-curriculum-gates",
            "or,and",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    with (run_dir / "run_config.json").open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    with (run_dir / "run_features.json").open("r", encoding="utf-8") as handle:
        run_features = json.load(handle)

    assert run_config.get("logic_backend") == "engine"
    assert run_features.get("logic_backend") == "engine"
    assert bool(run_config.get("learning", {}).get("enabled")) is False
