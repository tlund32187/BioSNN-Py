"""Quick sweep: fine-grained gws to find stable regime for OR gate."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_one(
    *, drive_scale: float, gws: float, trials: int, receptor_mode: str, label: str
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
            "spike_window": 10,
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
        sim_steps_per_trial=30,
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
    spk = result.get("eval_spike_counts", [0, 0])
    tie_pct = result.get("tie_count", 0) / max(1, result.get("eval_trials", 1)) * 100.0
    print(f"  {label}: eval={eval_acc:.4f}  spk={[round(s, 2) for s in spk]}  ties={tie_pct:.0f}%")
    return result


def main() -> None:
    drive_scale = 1e-9

    configs = [
        # Fine-grained sweep around the working range
        (1e-7, "exc_only"),
        (5e-7, "exc_only"),
        (1e-6, "exc_only"),
        (2e-6, "exc_only"),
        (5e-6, "exc_only"),
        # Without NMDA (AMPA-only has faster decay, less accumulation)
        (1e-5, "none"),
        (5e-6, "none"),
        (1e-6, "none"),
    ]

    for gws, rmode in configs:
        w_max = gws * 3.0
        lr = w_max * 0.01
        label = f"gws={gws:.0e} rec={rmode}"
        try:
            run_one(drive_scale=drive_scale, gws=gws, trials=400, receptor_mode=rmode, label=label)
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
