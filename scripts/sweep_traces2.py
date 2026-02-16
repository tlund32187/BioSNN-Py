"""Focused sweep: trace-based R-STDP with full connectivity.

Results from prior sweep (incomplete):
- Traces + fan_in=2 + a_minus=0.6 + seed=42 → samp=0.8383 (vs 0.76 without traces!)
- But seed=123 still fails → connectivity problem

This focused sweep tests full connectivity (fan_in=5) with traces.
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
    gws: float,
    trials: int,
    sim_steps: int,
    a_minus: float,
    tau_e: float,
    tau_pre: float,
    tau_post: float,
    af_amplitude: float,
    drive_scale: float,
    skip_fan_in: int,
    in_fan_in: int,
    label: str,
) -> dict[str, Any]:
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
            "in_to_hidden_fan_in": in_fan_in,
        },
        "learning": {
            "enabled": True,
            "rule": "rstdp_elig",
            "lr": lr,
            "w_min": 0.0,
            "w_max": w_max,
            "a_plus": 1.0,
            "a_minus": a_minus,
            "tau_e": tau_e,
            "tau_pre": tau_pre,
            "tau_post": tau_post,
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
            "spike_window": sim_steps,
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
                "epsilon_start": 0.20,
                "epsilon_end": 0.01,
                "epsilon_decay_trials": 400,
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
        sim_steps_per_trial=sim_steps,
        device="cuda",
        learning_mode="rstdp",
        engine_learning_rule="rstdp_elig",
        inter_trial_reset=True,
        drive_scale=drive_scale,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=10,
        reward_delivery_clamp_input=True,
        debug=False,
        dump_last_trials_csv=False,
    )

    result = run_logic_gate_engine(cfg, spec)
    eval_acc = result.get("eval_accuracy", 0.0)
    sample_acc = result.get("sample_accuracy", 0.0)
    preds = result.get("preds", None)
    passed = result.get("passed", False)
    print(f"  {label}: eval={eval_acc:.4f} samp={sample_acc:.4f} passed={passed} preds={preds}")
    return result


def main() -> None:
    BASE = dict(
        gws=5e-6,
        trials=400,
        sim_steps=30,
        drive_scale=1e-9,
        af_amplitude=1.0,
        in_fan_in=2,
        tau_e=0.020,
        tau_pre=0.020,
        tau_post=0.020,
    )

    # Test 1: Full skip with traces, multiple seeds
    print("=== Full skip (fan_in=5) + traces + a_minus=0.6 ===")
    for seed in [42, 123, 7]:
        label = f"seed={seed}"
        try:
            run_one(**BASE, seed=seed, a_minus=0.6, skip_fan_in=5, label=label)
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")

    # Test 2: Full skip + traces + a_minus=0
    print("\n=== Full skip + traces + a_minus=0 ===")
    for seed in [42, 123, 7]:
        label = f"seed={seed}"
        try:
            run_one(**BASE, seed=seed, a_minus=0.0, skip_fan_in=5, label=label)
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")

    # Test 3: Higher action force with traces
    print("\n=== Full skip + traces + af=5 ===")
    for seed in [42, 123]:
        label = f"seed={seed}"
        try:
            run_one(**BASE, seed=seed, a_minus=0.6, skip_fan_in=5, af_amplitude=5.0, label=label)
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
