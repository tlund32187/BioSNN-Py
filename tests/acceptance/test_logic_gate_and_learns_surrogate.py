from __future__ import annotations

import pytest

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate

pytestmark = pytest.mark.acceptance


def test_logic_gate_and_learns_surrogate(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = LogicGateRunConfig(
        gate=LogicGate.AND,
        seed=29,
        steps=800,
        sim_steps_per_trial=1,
        device="cpu",
        learning_mode="surrogate",
        export_every=100,
        out_dir=tmp_path / "logic_and_surrogate",
    )
    summary = run_logic_gate(cfg)

    assert summary["learning_mode"] == "surrogate"
    assert summary["passed"] is True
    assert summary["first_pass_trial"] is not None
    assert summary["eval_accuracy"] >= 0.99
    assert summary["elapsed_s"] < 1.0
    assert summary["preds"].shape == (4,)
    assert summary["sample_indices"].shape[0] <= 800
    assert (summary["out_dir"] / "trials.csv").exists()
    assert (summary["out_dir"] / "eval.csv").exists()
    assert (summary["out_dir"] / "confusion.csv").exists()
