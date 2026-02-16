"""Sweep using trace-based R-STDP (with pre/post synaptic traces).

Key insight: the previous STDP rules use instantaneous pre*post coincidence,
which nearly never fires with sparse spiking. Trace-based STDP uses decaying
pre/post traces so that temporal correlation within a ~20ms window creates
proper eligibility. This should fix the fundamental learning failure.
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
    dopamine_scale: float = 1.0,
    baseline: float = 0.0,
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
            "dopamine_scale": dopamine_scale,
            "baseline": baseline,
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


COMMON = dict(
    gws=5e-6,
    trials=600,
    sim_steps=30,
    drive_scale=1e-9,
    af_amplitude=1.0,
    in_fan_in=2,
)


def main() -> None:
    # Group 1: Trace STDP with sparse skip (fan_in=2, default)
    print("=== Group 1: Trace STDP, fan_in=2 (default), tau=20ms ===")
    for seed in [42, 123, 7, 99, 2024]:
        label = f"seed={seed}"
        try:
            run_one(
                **COMMON,
                seed=seed,
                a_minus=0.6,
                tau_e=0.020,
                tau_pre=0.020,
                tau_post=0.020,
                skip_fan_in=2,
                label=label,
            )
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")

    # Group 2: Trace STDP with full skip (fan_in=5)
    print("\n=== Group 2: Trace STDP, fan_in=5 (full), tau=20ms ===")
    for seed in [42, 123, 7, 99, 2024]:
        label = f"seed={seed}"
        try:
            run_one(
                **COMMON,
                seed=seed,
                a_minus=0.6,
                tau_e=0.020,
                tau_pre=0.020,
                tau_post=0.020,
                skip_fan_in=5,
                label=label,
            )
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")

    # Group 3: Trace STDP, a_minus=0 vs a_minus=0.6
    print("\n=== Group 3: a_minus comparison, fan_in=5, tau=20ms ===")
    for a_minus in [0.0, 0.3, 0.6]:
        label = f"a_minus={a_minus}"
        try:
            run_one(
                **COMMON,
                seed=42,
                a_minus=a_minus,
                tau_e=0.020,
                tau_pre=0.020,
                tau_post=0.020,
                skip_fan_in=5,
                label=label,
            )
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")

    # Group 4: Different tau_pre/tau_post
    print("\n=== Group 4: tau sweep, fan_in=5, a_minus=0.6 ===")
    for tau in [0.005, 0.010, 0.050]:
        label = f"tau={tau:.3f}"
        try:
            run_one(
                **COMMON,
                seed=42,
                a_minus=0.6,
                tau_e=0.020,
                tau_pre=tau,
                tau_post=tau,
                skip_fan_in=5,
                label=label,
            )
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")

    # Group 5: Stronger action force with traces
    print("\n=== Group 5: action force, fan_in=5, tau=20ms ===")
    for af in [3.0, 5.0]:
        label = f"af={af}"
        try:
            run_one(
                **COMMON,
                seed=42,
                a_minus=0.6,
                tau_e=0.020,
                tau_pre=0.020,
                tau_post=0.020,
                af_amplitude=af,
                skip_fan_in=5,
                label=label,
            )
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
