from __future__ import annotations

import csv

import pytest

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_curriculum_engine

pytestmark = pytest.mark.unit


def test_curriculum_engine_applies_reward_window_steps_between_trials(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=17,
        steps=4,
        sim_steps_per_trial=1,
        device="cpu",
        learning_mode="none",
        out_dir=tmp_path / "logic_reward_window",
        export_every=2,
    )
    run_spec = {
        "dtype": "float32",
        "delay_steps": 1,
        "monitors_enabled": False,
        "synapse": {
            "backend": "spmm_fused",
            "fused_layout": "auto",
            "ring_strategy": "dense",
            "ring_dtype": "none",
            "store_sparse_by_delay": None,
            "receptor_mode": "exc_only",
        },
        "learning": {"enabled": False, "rule": "none", "lr": 1e-3},
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "pulse_step": 10,
            "amount": 1.0,
            "field_type": "global_scalar",
            "grid_size": [16, 16],
            "world_extent": [1.0, 1.0],
            "diffusion": 0.0,
            "decay_tau": 1.0,
            "deposit_sigma": 0.0,
        },
        "logic_curriculum_gates": "or",
        "logic_curriculum_replay_ratio": 0.0,
        "logic": {
            "reward_delivery_steps": 2,
            "reward_delivery_clamp_input": True,
            "exploration": {
                "enabled": False,
                "mode": "epsilon_greedy",
                "epsilon_start": 0.0,
                "epsilon_end": 0.0,
                "epsilon_decay_trials": 1,
                "tie_break": "random_among_max",
                "seed": 17,
            },
        },
    }

    result = run_logic_gate_curriculum_engine(cfg, run_spec)
    trial_path = result["out_dir"] / "trials.csv"
    with trial_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    sim_step_end = [int(row["sim_step_end"]) for row in rows]
    assert sim_step_end == [3, 6, 9, 12]
    assert all(float(row.get("epsilon", "0") or 0.0) == 0.0 for row in rows)


def test_action_force_commits_exploration_and_generates_spikes_for_action_zero(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=29,
        steps=32,
        sim_steps_per_trial=1,
        device="cpu",
        learning_mode="none",
        out_dir=tmp_path / "logic_action_force",
        export_every=8,
    )
    run_spec = {
        "dtype": "float32",
        "delay_steps": 1,
        "monitors_enabled": False,
        "synapse": {
            "backend": "spmm_fused",
            "fused_layout": "auto",
            "ring_strategy": "dense",
            "ring_dtype": "none",
            "store_sparse_by_delay": None,
            "receptor_mode": "exc_only",
        },
        "learning": {"enabled": False, "rule": "none", "lr": 1e-3},
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "pulse_step": 10,
            "amount": 1.0,
            "field_type": "global_scalar",
            "grid_size": [16, 16],
            "world_extent": [1.0, 1.0],
            "diffusion": 0.0,
            "decay_tau": 1.0,
            "deposit_sigma": 0.0,
        },
        "logic_curriculum_gates": "or",
        "logic_curriculum_replay_ratio": 0.0,
        "logic": {
            "reward_delivery_steps": 2,
            "reward_delivery_clamp_input": True,
            "action_force": {
                "enabled": True,
                "window": "reward_window",
                "steps": 1,
                "amplitude": 1.0,
                "compartment": "soma",
            },
            "exploration": {
                "enabled": True,
                "mode": "epsilon_greedy",
                "epsilon_start": 1.0,
                "epsilon_end": 1.0,
                "epsilon_decay_trials": 1,
                "tie_break": "random_among_max",
                "seed": 29,
            },
        },
    }

    result = run_logic_gate_curriculum_engine(cfg, run_spec)
    trial_path = result["out_dir"] / "trials.csv"
    with trial_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    zero_action_rows = [
        row
        for row in rows
        if int((row.get("chosen_action") or "1").strip() or 1) == 0
    ]
    assert zero_action_rows, "expected at least one chosen action=0 row under epsilon=1 exploration"
    assert any(
        float((row.get("out_spikes_0") or "0").strip() or 0.0) > 0.0 for row in zero_action_rows
    )
    assert any(int((row.get("action_forced") or "0").strip() or 0) == 1 for row in zero_action_rows)
