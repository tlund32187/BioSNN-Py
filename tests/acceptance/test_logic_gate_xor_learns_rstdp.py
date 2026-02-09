from __future__ import annotations

import pytest

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate

pytestmark = pytest.mark.acceptance


def test_logic_gate_xor_learns_rstdp(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = LogicGateRunConfig(
        gate=LogicGate.XOR,
        seed=31,
        steps=20000,
        sim_steps_per_trial=2,
        device="cpu",
        learning_mode="rstdp",
        export_every=200,
        out_dir=tmp_path / "logic_xor_rstdp",
    )
    summary = run_logic_gate(cfg)
    assert summary["topology"] == "xor_ff2"
    assert summary["passed"] is True
    assert summary["first_pass_trial"] is not None
    assert summary["eval_accuracy"] >= 0.99

