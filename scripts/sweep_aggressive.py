"""Sweep with more aggressive params now that voltage clamping prevents divergence."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_one(
    *,
    drive_scale: float,
    gws: float,
    trials: int,
    sim_steps: int,
    af_amplitude: float,
    lr_mult: float,
    label: str,
) -> dict[str, Any]:
    w_max = gws * 3.0
    lr = w_max * 0.01 * lr_mult

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
    preds = result.get("preds", None)
    passed = result.get("passed", False)
    first_pass = result.get("first_pass_trial", None)
    print(
        f"  {label}: eval={eval_acc:.4f} samp={sample_acc:.4f} passed={passed} fp={first_pass} preds={preds}"
    )
    return result


def main() -> None:
    configs = [
        # Higher gws → stronger synaptic drives → more output spikes
        {"ds": 1e-9, "gws": 1e-5, "ss": 30, "af": 1.0, "lr_mult": 1.0},
        {"ds": 1e-9, "gws": 2e-5, "ss": 30, "af": 1.0, "lr_mult": 1.0},
        {"ds": 1e-9, "gws": 5e-5, "ss": 30, "af": 1.0, "lr_mult": 1.0},
        # Higher action force → stronger teaching signal
        {"ds": 1e-9, "gws": 5e-6, "ss": 30, "af": 5.0, "lr_mult": 1.0},
        {"ds": 1e-9, "gws": 1e-5, "ss": 30, "af": 5.0, "lr_mult": 1.0},
        {"ds": 1e-9, "gws": 5e-6, "ss": 30, "af": 10.0, "lr_mult": 1.0},
        # Longer trial → more integration time
        {"ds": 1e-9, "gws": 5e-6, "ss": 50, "af": 1.0, "lr_mult": 1.0},
        {"ds": 1e-9, "gws": 1e-5, "ss": 50, "af": 1.0, "lr_mult": 1.0},
        # Higher drive_scale → input fires early → more propagation time
        {"ds": 2e-9, "gws": 5e-6, "ss": 30, "af": 1.0, "lr_mult": 1.0},
        {"ds": 5e-9, "gws": 5e-6, "ss": 30, "af": 1.0, "lr_mult": 1.0},
        # Higher learning rate
        {"ds": 1e-9, "gws": 5e-6, "ss": 30, "af": 1.0, "lr_mult": 5.0},
        {"ds": 1e-9, "gws": 5e-6, "ss": 30, "af": 1.0, "lr_mult": 10.0},
    ]

    for cfg in configs:
        label = (
            f"ds={cfg['ds']:.0e} gws={cfg['gws']:.0e} ss={cfg['ss']} "
            f"af={cfg['af']} lr×{cfg['lr_mult']}"
        )
        try:
            run_one(
                drive_scale=cfg["ds"],
                gws=cfg["gws"],
                trials=600,
                sim_steps=cfg["ss"],
                af_amplitude=cfg["af"],
                lr_mult=cfg["lr_mult"],
                label=label,
            )
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
