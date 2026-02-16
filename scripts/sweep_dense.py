"""Sweep: dense learning mode with trace STDP (fixes sparse/trace bug).

The key fix: supports_sparse=False when traces enabled, so ALL edges
get updated every step and trace decay properly carries temporal info.

Strategy:
- Dense mode (automatic via traces)
- RL mode (not supervised) for differential DA signals (+correct, -wrong)
- No baseline (let DA be the only weight driver)
- Try varying tau_e, DA decay, lr
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
    af_mode: str,
    seed: int = 42,
    steps: int = 600,
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
                "mode": af_mode,
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
    configs = [
        # RL mode (differential DA) + dense traces + different lr/tau combos
        # 1. Base RL config (dense traces fix only)
        {
            "label": "RL+dense lr=1x tau_e=20ms",
            "lr_mult": 1.0,
            "tau_e": 0.020,
            "baseline": 0.0,
            "da_decay_tau": 0.01,
            "af_mode": "always",
        },
        # 2. Longer eligibility persistence
        {
            "label": "RL+dense lr=1x tau_e=50ms",
            "lr_mult": 1.0,
            "tau_e": 0.050,
            "baseline": 0.0,
            "da_decay_tau": 0.01,
            "af_mode": "always",
        },
        # 3. Even longer eligibility
        {
            "label": "RL+dense lr=1x tau_e=100ms",
            "lr_mult": 1.0,
            "tau_e": 0.100,
            "baseline": 0.0,
            "da_decay_tau": 0.01,
            "af_mode": "always",
        },
        # 4. Slower DA decay so more reward steps get DA
        {
            "label": "RL+dense lr=1x tau_e=50ms da=0.1",
            "lr_mult": 1.0,
            "tau_e": 0.050,
            "baseline": 0.0,
            "da_decay_tau": 0.10,
            "af_mode": "always",
        },
        # 5. Much higher lr
        {
            "label": "RL+dense lr=10x tau_e=50ms",
            "lr_mult": 10.0,
            "tau_e": 0.050,
            "baseline": 0.0,
            "da_decay_tau": 0.01,
            "af_mode": "always",
        },
        # 6. Kitchen sink: lr=10x, tau_e=100ms, slow DA
        {
            "label": "RL+dense all boost",
            "lr_mult": 10.0,
            "tau_e": 0.100,
            "baseline": 0.0,
            "da_decay_tau": 0.10,
            "af_mode": "always",
        },
        # 7. Supervised for comparison
        {
            "label": "SV+dense lr=1x tau_e=50ms",
            "lr_mult": 1.0,
            "tau_e": 0.050,
            "baseline": 0.0,
            "da_decay_tau": 0.01,
            "af_mode": "supervised",
        },
        # 8. Multi-seed check on best config (try tau_e=50ms)
        {
            "label": "RL seed=123",
            "lr_mult": 1.0,
            "tau_e": 0.050,
            "baseline": 0.0,
            "da_decay_tau": 0.01,
            "af_mode": "always",
            "seed": 123,
        },
    ]

    print("=== Dense Learning + Trace STDP Sweep ===", flush=True)
    for c in configs:
        try:
            run_one(**c)
        except Exception as e:
            print(f"  {c['label']}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
