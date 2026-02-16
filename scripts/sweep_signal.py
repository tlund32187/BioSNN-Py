"""Sweep: fix learning signal strength.

Key findings from diagnostic:
- max |dw| = 1.6e-9 (0.01% of w_max) → need 100-1000x higher
- DA and eligibility rarely coincide temporally
- Need: higher lr, longer tau_e, baseline > 0, slower DA decay

Strategy: test parameter combinations that amplify the dw signal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_one(
    *,
    label: str,
    lr_mult: float,
    tau_e: float,
    baseline: float,
    da_decay_tau: float,
    seed: int = 42,
    steps: int = 400,
) -> None:
    gws = 5e-6
    w_max = gws * 3.0
    lr = w_max * lr_mult  # lr_mult replaces 0.01

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
            "skip_fan_in": 5,
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
            "tau_e": tau_e,
            "tau_pre": 0.020,
            "tau_post": 0.020,
            "weight_decay": 0.0,
            "dopamine_scale": 1.0,
            "baseline": baseline,
            "synaptic_scaling": False,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": da_decay_tau,
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
                "epsilon_start": 0.20,
                "epsilon_end": 0.01,
                "epsilon_decay_trials": 400,
            },
            "action_force": {
                "enabled": True,
                "mode": "supervised",
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
        steps=steps,
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
        f"  {label}: eval={r['eval_accuracy']:.4f} samp={r['sample_accuracy']:.4f} "
        f"preds={r['preds']} dw={r.get('mean_abs_dw', 0):.2e}",
        flush=True,
    )


def main() -> None:
    # Grid: lr_mult × tau_e × baseline × da_decay_tau
    # Focus on amplifying the effective dw

    configs = [
        # 1. Baseline experiment: just increase lr 100x
        {"label": "lr100x", "lr_mult": 1.0, "tau_e": 0.020, "baseline": 0.0, "da_decay_tau": 0.01},
        # 2. Increase lr 100x + longer eligibility (100ms)
        {
            "label": "lr100x+te100",
            "lr_mult": 1.0,
            "tau_e": 0.100,
            "baseline": 0.0,
            "da_decay_tau": 0.01,
        },
        # 3. lr 100x + baseline 0.5 (unsupervised Hebbian drive)
        {
            "label": "lr100x+b0.5",
            "lr_mult": 1.0,
            "tau_e": 0.020,
            "baseline": 0.5,
            "da_decay_tau": 0.01,
        },
        # 4. lr 100x + longer tau_e + baseline
        {
            "label": "lr100x+te100+b0.5",
            "lr_mult": 1.0,
            "tau_e": 0.100,
            "baseline": 0.5,
            "da_decay_tau": 0.01,
        },
        # 5. lr 100x + slower DA decay
        {
            "label": "lr100x+da0.1",
            "lr_mult": 1.0,
            "tau_e": 0.020,
            "baseline": 0.0,
            "da_decay_tau": 0.10,
        },
        # 6. Kitchen sink: everything boosted
        {
            "label": "all_boost",
            "lr_mult": 1.0,
            "tau_e": 0.100,
            "baseline": 0.5,
            "da_decay_tau": 0.10,
        },
        # 7. Higher lr: 1000x
        {
            "label": "lr1000x",
            "lr_mult": 10.0,
            "tau_e": 0.020,
            "baseline": 0.0,
            "da_decay_tau": 0.01,
        },
        # 8. lr 1000x + baseline
        {
            "label": "lr1000x+b0.5",
            "lr_mult": 10.0,
            "tau_e": 0.020,
            "baseline": 0.5,
            "da_decay_tau": 0.01,
        },
        # 9. lr 1000x + all boosts
        {
            "label": "lr1000x+all",
            "lr_mult": 10.0,
            "tau_e": 0.100,
            "baseline": 0.5,
            "da_decay_tau": 0.10,
        },
    ]

    print("=== Learning Signal Strength Sweep ===", flush=True)
    for c in configs:
        try:
            run_one(**c)
        except Exception as e:
            print(f"  {c['label']}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
