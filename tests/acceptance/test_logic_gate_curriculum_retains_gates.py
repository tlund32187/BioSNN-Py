from __future__ import annotations

import csv
from pathlib import Path

import pytest

from biosnn.tasks.logic_gates import LogicGateRunConfig, run_logic_gate_curriculum
from biosnn.tasks.logic_gates.datasets import LogicGate

pytestmark = pytest.mark.acceptance


def test_logic_curriculum_retains_or_and_xor(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    cfg = LogicGateRunConfig(
        seed=7,
        steps=800,
        sim_steps_per_trial=2,
        device="cpu",
        learning_mode="rstdp",
        export_every=200,
        out_dir=tmp_path / "logic_curriculum_or_and_xor",
    )
    summary = run_logic_gate_curriculum(
        cfg,
        gates=(LogicGate.OR, LogicGate.AND, LogicGate.XOR),
        phase_steps=800,
        replay_ratio=0.35,
    )

    final_eval = summary["final_eval_by_gate"]
    assert float(final_eval["or"]) >= 0.99
    assert float(final_eval["and"]) >= 0.99
    assert float(final_eval["xor"]) >= 0.99

    with (Path(summary["eval_csv"])).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    global_eval = [float(row["global_eval_accuracy"]) for row in rows if row.get("global_eval_accuracy")]
    assert global_eval
    assert global_eval[-1] >= 0.99

