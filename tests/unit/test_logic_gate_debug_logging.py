from __future__ import annotations

import pytest

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate

pytestmark = pytest.mark.unit


def test_logic_gate_debug_logging_and_last_trials_dump(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    out_dir = tmp_path / "logic_gate_debug"
    cfg = LogicGateRunConfig(
        gate=LogicGate.AND,
        seed=11,
        steps=12,
        sim_steps_per_trial=2,
        device="cpu",
        learning_mode="rstdp",
        export_every=6,
        debug=True,
        debug_every=5,
        debug_top_k=2,
        dump_last_trials_csv=True,
        dump_last_trials_n=5,
        out_dir=out_dir,
    )

    summary = run_logic_gate(cfg)

    captured = capsys.readouterr().out
    assert "[logic-gate][debug] trial=1/12" in captured
    assert "[logic-gate][debug] trial=5/12" in captured
    assert "[logic-gate][debug] trial=10/12" in captured
    assert "[logic-gate][debug] trial=12/12" in captured
    assert "dopamine=" in captured
    assert "hidden_topk=" in captured

    last_csv = summary["last_trials_csv"]
    assert last_csv is not None
    assert last_csv.exists()
    lines = [line for line in last_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 6  # header + last 5 rows
