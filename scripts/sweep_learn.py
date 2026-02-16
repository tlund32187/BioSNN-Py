"""Test learning with skip_fan_in=2 (correct default) — sweep configs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_config(
    *,
    label: str,
    gws: float = 5e-6,
    lr_mult: float = 1.0,
    tau_e: float = 0.020,
    tau_pre: float = 0.020,
    tau_post: float = 0.020,
    da_baseline: float = 0.0,
    da_decay: float = 0.01,
    da_scale: float = 1.0,
    af_mode: str = "supervised",
    explore: bool = True,
    eps_start: float = 0.3,
    eps_end: float = 0.05,
    eps_decay: int = 200,
    steps: int = 400,
    seed: int = 42,
    sim_steps: int = 30,
    skip_fi: int = 2,
) -> dict[str, Any]:
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
            "skip_fan_in": skip_fi,
            "in_to_hidden_fan_in": 2,
        },
        "learning": {
            "enabled": True,
            "rule": "rstdp_eligibility",
            "lr": lr,
            "w_min": 0.0,
            "w_max": w_max,
            "tau_e": tau_e,
            "tau_pre": tau_pre,
            "tau_post": tau_post,
            "dopamine_scale": da_scale,
            "baseline": da_baseline,
            "synaptic_scaling": False,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": da_decay,
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
                "enabled": explore,
                "epsilon_start": eps_start,
                "epsilon_end": eps_end,
                "epsilon_decay_trials": eps_decay,
                "tie_break": "random_among_max",
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
        sim_steps_per_trial=sim_steps,
        device="cuda",
        learning_mode="rstdp",
        engine_learning_rule="rstdp_elig",
        inter_trial_reset=True,
        drive_scale=1e-9,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=10,
        reward_delivery_clamp_input=True,
        debug=True,
    )

    r = run_logic_gate_engine(cfg, spec)
    return r


def main() -> None:
    configs = [
        # (label, kwargs)
        (
            "baseline: no learning, no explore",
            dict(
                label="no_learn",
                steps=20,
                explore=False,
                # Override learning to off
            ),
        ),
        (
            "supervised, lr=1x, explore",
            dict(
                label="sup_1x",
                af_mode="supervised",
                lr_mult=1.0,
                explore=True,
            ),
        ),
        (
            "supervised, lr=10x, explore",
            dict(
                label="sup_10x",
                af_mode="supervised",
                lr_mult=10.0,
                explore=True,
            ),
        ),
        (
            "supervised, lr=100x, explore",
            dict(
                label="sup_100x",
                af_mode="supervised",
                lr_mult=100.0,
                explore=True,
            ),
        ),
        (
            "always, lr=10x, explore",
            dict(
                label="always_10x",
                af_mode="always",
                lr_mult=10.0,
                explore=True,
            ),
        ),
        (
            "supervised, lr=10x, tau_e=50ms",
            dict(
                label="sup_t50",
                af_mode="supervised",
                lr_mult=10.0,
                tau_e=0.050,
            ),
        ),
        (
            "supervised, lr=10x, da_decay=0.1",
            dict(
                label="sup_dd01",
                af_mode="supervised",
                lr_mult=10.0,
                da_decay=0.10,
            ),
        ),
        (
            "sup, lr=10x, tau_e=50, da_decay=0.1",
            dict(
                label="sup_combo",
                af_mode="supervised",
                lr_mult=10.0,
                tau_e=0.050,
                da_decay=0.10,
            ),
        ),
    ]

    print(f"{'Config':<45s}  {'eval':>5s}  {'preds':>20s}")
    print("-" * 80)

    for label, kwargs in configs:
        try:
            # Special case: no-learning baseline
            if kwargs.get("label") == "no_learn":
                no_learn_spec: dict[str, Any] = {
                    "dtype": "float32",
                    "delay_steps": 3,
                    "synapse": {
                        "backend": "spmm_fused",
                        "fused_layout": "auto",
                        "ring_strategy": "dense",
                        "ring_dtype": "none",
                        "receptor_mode": "exc_only",
                        "global_weight_scale": 5e-6,
                        "skip_fan_in": 2,
                        "in_to_hidden_fan_in": 2,
                    },
                    "learning": {"enabled": False, "synaptic_scaling": False},
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
                        "exploration": {"enabled": False},
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
                no_learn_cfg = LogicGateRunConfig(
                    gate=LogicGate.OR,
                    seed=42,
                    steps=20,
                    dt=1e-3,
                    sim_steps_per_trial=30,
                    device="cuda",
                    learning_mode="none",
                    engine_learning_rule="none",
                    inter_trial_reset=True,
                    drive_scale=1e-9,
                    curriculum_gate_context={"enabled": True, "amplitude": 0.30},
                    reward_delivery_steps=10,
                    reward_delivery_clamp_input=True,
                    debug=True,
                )
                r = run_logic_gate_engine(no_learn_cfg, no_learn_spec)
            else:
                r = run_config(**kwargs)  # type: ignore[arg-type]
            print(f"{label:<45s}  {r['eval_accuracy']:>5.2f}  {r['preds']}")
        except Exception as e:
            print(f"{label:<45s}  ERROR: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
