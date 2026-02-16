"""Sweep without receptor profile to avoid NMDA accumulation divergence."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_one(
    *, drive_scale: float, gws: float, trials: int, sim_steps: int, receptor_mode: str, label: str
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
            "receptor_mode": receptor_mode,
            "global_weight_scale": gws,
        },
        "learning": {
            "enabled": True,
            "rule": "three_factor_elig_stdp",
            "lr": lr,
            "w_min": 0.0,
            "w_max": w_max,
            "a_plus": 1.0,
            "a_minus": 0.6,
            "tau_e": 0.020,
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
        seed=42,
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
    print(f"  {label}: eval={eval_acc:.4f} samp={sample_acc:.4f}")
    return result


def main() -> None:
    # No receptor profile: synaptic drive is delta-function per spike
    # No NMDA accumulation divergence

    configs = [
        # receptor_mode="none": no receptor filtering
        # Need larger gws since no temporal integration
        {"drive_scale": 1e-9, "gws": 1e-5, "sim_steps": 30, "receptor_mode": "none"},
        {"drive_scale": 1e-9, "gws": 5e-5, "sim_steps": 30, "receptor_mode": "none"},
        {"drive_scale": 1e-9, "gws": 1e-4, "sim_steps": 30, "receptor_mode": "none"},
        {"drive_scale": 5e-9, "gws": 1e-5, "sim_steps": 30, "receptor_mode": "none"},
        {"drive_scale": 5e-9, "gws": 5e-6, "sim_steps": 30, "receptor_mode": "none"},
        {"drive_scale": 5e-9, "gws": 1e-6, "sim_steps": 30, "receptor_mode": "none"},
        # receptor exc_only but shorter trial
        {"drive_scale": 1e-9, "gws": 5e-7, "sim_steps": 30, "receptor_mode": "exc_only"},
        {"drive_scale": 1e-9, "gws": 1e-7, "sim_steps": 30, "receptor_mode": "exc_only"},
    ]

    for cfg in configs:
        rmode = cfg["receptor_mode"]
        ds = cfg["drive_scale"]
        gws = cfg["gws"]
        ss = cfg["sim_steps"]
        label = f"ds={ds:.0e} gws={gws:.0e} ss={ss} rec={rmode}"
        try:
            run_one(
                drive_scale=ds,
                gws=gws,
                trials=400,
                sim_steps=ss,
                receptor_mode=rmode,
                label=label,
            )
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
