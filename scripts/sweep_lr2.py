"""Quick tests: higher lr with trace STDP, 400 trials."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_one(
    *, seed: int, lr_mult: float, weight_decay: float, skip_fan_in: int, label: str
) -> None:
    gws = 5e-6
    w_max = gws * 3.0
    lr = w_max * lr_mult

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
            "a_minus": 0.6,
            "tau_e": 0.020,
            "tau_pre": 0.020,
            "tau_post": 0.020,
            "weight_decay": weight_decay,
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
                "epsilon_start": 0.30,
                "epsilon_end": 0.05,
                "epsilon_decay_trials": 300,
            },
            "action_force": {
                "enabled": True,
                "mode": "always",
                "amplitude": 1.0,
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
        steps=400,
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
        f"  {label}: eval={r['eval_accuracy']:.4f} samp={r['sample_accuracy']:.4f} preds={r['preds']}",
        flush=True,
    )


def main() -> None:
    print("=== lr x0.1 (10x), fan_in=5, wd=0 ===", flush=True)
    run_one(seed=42, lr_mult=0.1, weight_decay=0.0, skip_fan_in=5, label="seed=42")

    print("\n=== lr x0.1, fan_in=5, wd=0.001 ===", flush=True)
    run_one(seed=42, lr_mult=0.1, weight_decay=0.001, skip_fan_in=5, label="seed=42")

    print("\n=== lr x0.5 (50x), fan_in=5, wd=0.001 ===", flush=True)
    run_one(seed=42, lr_mult=0.5, weight_decay=0.001, skip_fan_in=5, label="seed=42")

    print("\n=== lr x0.1, fan_in=2, wd=0 ===", flush=True)
    run_one(seed=42, lr_mult=0.1, weight_decay=0.0, skip_fan_in=2, label="seed=42")


if __name__ == "__main__":
    main()
