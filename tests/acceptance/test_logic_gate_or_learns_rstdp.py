from __future__ import annotations

import pytest

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate

pytestmark = pytest.mark.acceptance


def test_logic_gate_or_learns_rstdp(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=23,
        steps=5000,
        sim_steps_per_trial=2,
        device="cpu",
        learning_mode="rstdp",
        export_every=100,
        out_dir=tmp_path / "logic_or_rstdp",
    )
    summary = run_logic_gate(cfg)
    assert summary["passed"] is True
    assert summary["first_pass_trial"] is not None
    assert summary["eval_accuracy"] >= 0.99
