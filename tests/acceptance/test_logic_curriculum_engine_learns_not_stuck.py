from __future__ import annotations

import csv
from pathlib import Path

import pytest

from biosnn.runners import cli

pytestmark = pytest.mark.acceptance


def test_logic_curriculum_engine_not_stuck_at_single_prediction(
    monkeypatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "logic_curriculum_engine_not_stuck"
    args = cli._parse_args(
        [
            "--demo",
            "logic_curriculum",
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "400",
            "--logic-backend",
            "engine",
            "--logic-learning-mode",
            "rstdp",
            "--logic-curriculum-gates",
            "or",
            "--logic-curriculum-replay-ratio",
            "0.0",
            "--logic-exploration-enabled",
            "--logic-epsilon-start",
            "0.25",
            "--logic-epsilon-end",
            "0.05",
            "--logic-epsilon-decay-trials",
            "300",
            "--logic-reward-delivery-steps",
            "2",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    with (run_dir / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
        trials = list(csv.DictReader(handle))
    assert trials
    preds = [int((row.get("pred") or "0").strip() or 0) for row in trials]
    assert len(set(preds)) > 1

    explored_count = sum(int((row.get("explored") or "0").strip() or 0) for row in trials)
    assert explored_count > 0

    with (run_dir / "eval.csv").open("r", encoding="utf-8", newline="") as handle:
        eval_rows = list(csv.DictReader(handle))
    assert eval_rows
    or_eval = [
        float(row.get("eval_accuracy") or 0.0)
        for row in eval_rows
        if str(row.get("gate", "")).strip().lower() == "or"
    ]
    assert or_eval
    assert max(or_eval) > 0.30
