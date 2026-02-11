from __future__ import annotations

import csv
from typing import Any, cast

import pytest

from biosnn.tasks.logic_gates import (
    LogicGate,
    LogicGateRunConfig,
    run_logic_gate_curriculum_engine,
    run_logic_gate_engine,
)

pytestmark = pytest.mark.unit


def _mean_numeric_column(path, column: str) -> float:
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            values.append(float(raw))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _base_run_spec(
    *,
    backend: str = "spmm_fused",
    learning_rule: str = "three_factor_elig_stdp",
) -> dict[str, object]:
    return {
        "dtype": "float32",
        "delay_steps": 1,
        "synapse": {
            "backend": backend,
            "fused_layout": "auto",
            "ring_strategy": "dense",
            "ring_dtype": "none",
            "store_sparse_by_delay": None,
            "receptor_mode": "exc_only",
        },
        "learning": {
            "enabled": True,
            "rule": learning_rule,
            "lr": 0.05,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "pulse_step": 10,
            "amount": 1.0,
            "field_type": "global_scalar",
            "grid_size": "16x16",
            "world_extent": [1.0, 1.0],
            "diffusion": 0.0,
            "decay_tau": 1.0,
            "deposit_sigma": 0.0,
        },
        "homeostasis": {
            "enabled": False,
            "rule": "rate_ema_threshold",
            "alpha": 0.01,
            "eta": 1e-3,
            "r_target": 0.05,
            "clamp_min": 0.0,
            "clamp_max": 0.05,
            "scope": "per_neuron",
        },
        "pruning": {
            "enabled": False,
            "prune_interval_steps": 250,
            "usage_alpha": 0.01,
            "w_min": 0.05,
            "usage_min": 0.01,
            "k_min_out": 1,
            "k_min_in": 1,
            "max_prune_fraction_per_interval": 0.1,
            "verbose": False,
        },
        "neurogenesis": {
            "enabled": False,
            "growth_interval_steps": 500,
            "add_neurons_per_event": 4,
            "newborn_plasticity_multiplier": 1.5,
            "newborn_duration_steps": 250,
            "max_total_neurons": 20000,
            "verbose": False,
        },
    }


@pytest.mark.parametrize("learning_rule", ["three_factor_elig_stdp", "rstdp_elig"])
def test_run_logic_gate_engine_writes_harness_csv_artifacts(tmp_path, learning_rule: str) -> None:
    pytest.importorskip("torch")
    cfg = LogicGateRunConfig(
        gate=LogicGate.AND,
        seed=5,
        steps=8,
        sim_steps_per_trial=1,
        device="cpu",
        learning_mode="rstdp",
        out_dir=tmp_path / "logic_engine_single",
        export_every=4,
    )
    result = run_logic_gate_engine(
        cfg,
        _base_run_spec(backend="spmm_fused", learning_rule=learning_rule),
    )

    out_dir = result["out_dir"]
    assert (out_dir / "topology.json").exists()
    assert (out_dir / "trials.csv").exists()
    assert (out_dir / "eval.csv").exists()
    assert (out_dir / "confusion.csv").exists()


def test_run_logic_gate_curriculum_engine_writes_phase_summary(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = LogicGateRunConfig(
        gate=LogicGate.AND,
        seed=9,
        steps=4,
        sim_steps_per_trial=1,
        device="cpu",
        learning_mode="rstdp",
        out_dir=tmp_path / "logic_engine_curriculum",
        export_every=2,
    )
    run_spec = _base_run_spec(backend="event_driven")
    run_spec["logic_curriculum_gates"] = "or,and"
    run_spec["logic_curriculum_replay_ratio"] = 0.25

    result = run_logic_gate_curriculum_engine(cfg, run_spec)
    out_dir = result["out_dir"]
    assert (out_dir / "topology.json").exists()
    assert (out_dir / "trials.csv").exists()
    assert (out_dir / "eval.csv").exists()
    assert (out_dir / "confusion.csv").exists()
    assert (out_dir / "phase_summary.csv").exists()


def test_run_logic_gate_curriculum_engine_bio_modulation_toggles_change_behavior(tmp_path) -> None:
    pytest.importorskip("torch")
    baseline_cfg = LogicGateRunConfig(
        gate=LogicGate.AND,
        seed=13,
        steps=8,
        sim_steps_per_trial=2,
        device="cpu",
        learning_mode="rstdp",
        out_dir=tmp_path / "logic_engine_curriculum_baseline",
        export_every=4,
    )
    baseline_spec = _base_run_spec(backend="event_driven", learning_rule="three_factor_elig_stdp")
    baseline_spec["logic_curriculum_gates"] = "or,and"
    baseline_spec["logic_curriculum_replay_ratio"] = 0.0
    baseline_mods = dict(cast(dict[str, Any], baseline_spec["modulators"]))
    baseline_spec["modulators"] = {
        **baseline_mods,
        "enabled": False,
        "kinds": [],
    }
    baseline_spec["wrapper"] = {"enabled": False}
    baseline_spec["excitability_modulation"] = {"enabled": False}
    baseline_result = run_logic_gate_curriculum_engine(baseline_cfg, baseline_spec)

    bio_cfg = LogicGateRunConfig(
        gate=LogicGate.AND,
        seed=13,
        steps=8,
        sim_steps_per_trial=2,
        device="cpu",
        learning_mode="rstdp",
        out_dir=tmp_path / "logic_engine_curriculum_bio",
        export_every=4,
    )
    bio_spec = _base_run_spec(backend="event_driven", learning_rule="three_factor_elig_stdp")
    bio_spec["logic_curriculum_gates"] = "or,and"
    bio_spec["logic_curriculum_replay_ratio"] = 0.0
    bio_learning = dict(cast(dict[str, Any], bio_spec["learning"]))
    bio_spec["learning"] = {
        **bio_learning,
        "modulator_kind": "dopamine",
    }
    bio_mods = dict(cast(dict[str, Any], bio_spec["modulators"]))
    bio_spec["modulators"] = {
        **bio_mods,
        "enabled": True,
        "kinds": ["dopamine", "acetylcholine", "noradrenaline", "serotonin"],
        "field_type": "grid_diffusion_2d",
        "grid_size": [8, 8],
        "world_extent": [1.0, 1.0],
        "diffusion": 0.15,
        "decay_tau": 0.8,
        "deposit_sigma": 0.1,
        "amount": 2.5,
    }
    bio_spec["wrapper"] = {
        "enabled": True,
        "ach_lr_gain": 0.5,
        "ne_lr_gain": 0.25,
        "ht_lr_gain": 0.1,
        "ht_extra_weight_decay": 0.02,
        "combine_mode": "exp",
    }
    bio_spec["excitability_modulation"] = {
        "enabled": True,
        "targets": ["hidden", "out"],
        "compartment": "soma",
        "ach_gain": 4.0,
        "ne_gain": 2.0,
        "ht_gain": 0.5,
        "clamp_abs": 5.0,
    }
    bio_result = run_logic_gate_curriculum_engine(bio_cfg, bio_spec)

    bio_out = bio_result["out_dir"]
    assert (bio_out / "topology.json").exists()
    assert (bio_out / "trials.csv").exists()
    assert (bio_out / "eval.csv").exists()
    assert (bio_out / "confusion.csv").exists()
    assert (bio_out / "phase_summary.csv").exists()

    baseline_trials = baseline_result["out_dir"] / "trials.csv"
    bio_trials = bio_out / "trials.csv"
    baseline_hidden = _mean_numeric_column(baseline_trials, "hidden_mean_spikes")
    bio_hidden = _mean_numeric_column(bio_trials, "hidden_mean_spikes")
    assert abs(bio_hidden - baseline_hidden) > 1e-6

    with (bio_out / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        dopamine = [float((row.get("dopamine_pulse") or "0").strip() or 0.0) for row in reader]
    assert any(abs(value) > 0.0 for value in dopamine)
