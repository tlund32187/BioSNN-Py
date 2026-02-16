"""Sweep: trace-based R-STDP with higher exploration and more trials.

Key insight: the action force only reinforces the CHOSEN output's connections.
To learn bit1→out1, the network must explore action 1 for case (0,1). With low
exploration (20%), this happens ~10% of trials for that case. More exploration
and more trials should fix this.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_one(
    *,
    seed: int,
    skip_fan_in: int,
    a_minus: float,
    af_amplitude: float,
    trials: int,
    epsilon_start: float,
    epsilon_decay_trials: int,
    label: str,
) -> None:
    gws = 5e-6
    w_max = gws * 3.0
    lr = w_max * 0.01

    spec: dict[str, Any] = {
        "dtype": "float32",
        "delay_steps": 3,
        "synapse": {
            "backend": "spmm_fused",
            "fused_layout": "auto",
            "ring_strategy": "dense",
            "ring_dtype": "none",
            "receptor_mode": "exc_only",
            "global_weight_scale": gws,
            "skip_fan_in": skip_fan_in,
            "in_to_hidden_fan_in": 2,
        },
        "learning": {
            "enabled": True,
            "rule": "rstdp_elig",
            "lr": lr,
            "w_min": 0.0,
            "w_max": w_max,
            "a_plus": 1.0,
            "a_minus": a_minus,
            "tau_e": 0.020,
            "tau_pre": 0.020,
            "tau_post": 0.020,
            "weight_decay": 0.0,
            "dopamine_scale": 1.0,
            "baseline": 0.0,
            "synaptic_scaling": False,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": 0.01,
        },
        "wrapper": {
            "enabled": True,
            "spike_window": 30,
            "decision_mode": "spike_count",
        },
        "homeostasis": {
            "enabled": True,
            "target_rate": 0.05,
            "tau": 1.0,
            "gain": 0.001,
        },
        "logic": {
            "exploration": {
                "enabled": True,
                "epsilon_start": epsilon_start,
                "epsilon_end": 0.01,
                "epsilon_decay_trials": epsilon_decay_trials,
            },
            "action_force": {
                "enabled": True,
                "mode": "always",
                "amplitude": af_amplitude,
                "window": "reward_window",
                "steps": 10,
                "compartment": "soma",
                "suppression_factor": -3.0,
            },
        },
    }

    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=seed,
        steps=trials,
        dt=1e-3,
        sim_steps_per_trial=30,
        device="cuda",
        learning_mode="rstdp",
        engine_learning_rule="rstdp_elig",
        inter_trial_reset=True,
        drive_scale=1e-9,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=10,
        reward_delivery_clamp_input=True,
        debug=False,
        dump_last_trials_csv=False,
    )

    r = run_logic_gate_engine(cfg, spec)
    print(
        f"  {label}: eval={r['eval_accuracy']:.4f} samp={r['sample_accuracy']:.4f} preds={r['preds']}"
    )


def main() -> None:
    # Test 1: High exploration (50%), 800 trials, slow decay
    print("=== High exploration (eps=0.50, decay=600), 800 trials ===")
    for seed in [42, 123, 7]:
        run_one(
            seed=seed,
            skip_fan_in=5,
            a_minus=0.6,
            af_amplitude=1.0,
            trials=800,
            epsilon_start=0.50,
            epsilon_decay_trials=600,
            label=f"seed={seed}",
        )

    # Test 2: Very high exploration (80%), 800 trials
    print("\n=== Very high exploration (eps=0.80, decay=600), 800 trials ===")
    run_one(
        seed=42,
        skip_fan_in=5,
        a_minus=0.6,
        af_amplitude=1.0,
        trials=800,
        epsilon_start=0.80,
        epsilon_decay_trials=600,
        label="seed=42",
    )

    # Test 3: Moderate exploration + stronger action force
    print("\n=== Moderate exploration (eps=0.50) + af=5 ===")
    run_one(
        seed=42,
        skip_fan_in=5,
        a_minus=0.6,
        af_amplitude=5.0,
        trials=800,
        epsilon_start=0.50,
        epsilon_decay_trials=600,
        label="af=5",
    )

    # Test 4: Same but with fan_in=2 (sparse, seed dependent)
    print("\n=== fan_in=2 + high exploration (eps=0.50, decay=600), 800 trials ===")
    for seed in [42, 123, 7]:
        run_one(
            seed=seed,
            skip_fan_in=2,
            a_minus=0.6,
            af_amplitude=1.0,
            trials=800,
            epsilon_start=0.50,
            epsilon_decay_trials=600,
            label=f"seed={seed}",
        )


if __name__ == "__main__":
    main()
