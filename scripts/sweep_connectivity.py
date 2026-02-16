"""Sweep with increased connectivity and fixed eligibility (a_minus=0).

Key changes:
- skip_fan_in=5 (fully connected In→OutSkip)
- in_to_hidden_fan_in=4 (higher connectivity)
- a_minus=0 (no self-defeating stimulus LTD)
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
            "rule": "three_factor_elig_stdp",
            "lr": lr,
            "w_min": 0.0,
            "w_max": w_max,
            "a_plus": 1.0,
            "a_minus": a_minus,
            "tau_e": tau_e,
            "modulator_threshold": 0.5,
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
        engine_learning_rule="three_factor_elig_stdp",
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
    # Test with full skip connectivity and a_minus=0
    print("=== Full skip (fan_in=5) + a_minus=0, multiple seeds ===")
    for seed in [42, 123, 7, 99, 2024]:
        label = f"seed={seed}"
        try:
            run_one(
                seed=seed,
                gws=5e-6,
                trials=600,
                sim_steps=30,
                a_minus=0.0,
                tau_e=0.020,
                af_amplitude=1.0,
                drive_scale=1e-9,
                skip_fan_in=5,
                in_fan_in=4,
                label=label,
            )
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")

    print("\n=== Full skip + higher gws ===")
    for gws in [1e-5, 2e-5]:
        for seed in [42, 123]:
            label = f"gws={gws:.0e} seed={seed}"
            try:
                run_one(
                    seed=seed,
                    gws=gws,
                    trials=600,
                    sim_steps=30,
                    a_minus=0.0,
                    tau_e=0.020,
                    af_amplitude=1.0,
                    drive_scale=1e-9,
                    skip_fan_in=5,
                    in_fan_in=4,
                    label=label,
                )
            except Exception as e:
                print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")

    print("\n=== Full skip + larger action force ===")
    for af in [5.0, 10.0]:
        for seed in [42, 123]:
            label = f"af={af} seed={seed}"
            try:
                run_one(
                    seed=seed,
                    gws=5e-6,
                    trials=600,
                    sim_steps=30,
                    a_minus=0.0,
                    tau_e=0.020,
                    af_amplitude=af,
                    drive_scale=1e-9,
                    skip_fan_in=5,
                    in_fan_in=4,
                    label=label,
                )
            except Exception as e:
                print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
