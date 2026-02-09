from __future__ import annotations

import pytest

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate

pytestmark = pytest.mark.acceptance


def test_logic_gate_harness_smoke_cpu(tmp_path):
    pytest.importorskip("torch")
    out_dir = tmp_path / "logic_gate_smoke"
    cfg = LogicGateRunConfig(
        gate=LogicGate.AND,
        seed=123,
        steps=50,
        sim_steps_per_trial=1,
        device="cpu",
        learning_mode="none",
        export_every=10,
        out_dir=out_dir,
    )

    summary = run_logic_gate(cfg)

    assert summary["out_dir"] == out_dir
    assert summary["inputs"].shape == (4, 2)
    assert summary["targets"].shape == (4, 1)
    assert summary["preds"].shape == (4,)
    assert summary["sample_indices"].shape == (50,)
    assert (out_dir / "trials.csv").exists()
    assert (out_dir / "eval.csv").exists()
    assert (out_dir / "confusion.csv").exists()

