"""Focused sweep: track actual spike-based decisions to find working OR config.

Strategy:
1. drive_scale controls input neuron speed (higher → faster spiking)
2. gws controls synaptic strength (must balance signal vs stability)
3. sim_steps_per_trial must be long enough for full chain propagation
4. Track real spike differentiation between output neurons
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_one(
    *, drive_scale: float, gws: float, trials: int, sim_steps: int, label: str
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
    total_spikes = result.get("total_output_spikes", "?")
    print(f"  {label}: eval={eval_acc:.4f} samp={sample_acc:.4f} tot_spk={total_spikes}")
    return result


def main() -> None:
    configs = [
        # Vary drive_scale: higher = faster input spiking
        # Vary gws: synaptic strength
        # Vary sim_steps: propagation time
        # Baseline: 30 steps, gws=1e-6
        {"drive_scale": 1e-9, "gws": 1e-6, "sim_steps": 30},
        # More time for propagation
        {"drive_scale": 1e-9, "gws": 1e-6, "sim_steps": 50},
        {"drive_scale": 1e-9, "gws": 1e-6, "sim_steps": 80},
        # Faster input + more time
        {"drive_scale": 5e-9, "gws": 1e-6, "sim_steps": 30},
        {"drive_scale": 5e-9, "gws": 1e-6, "sim_steps": 50},
        # Larger weights for stronger drive
        {"drive_scale": 5e-9, "gws": 5e-6, "sim_steps": 50},
        # Smaller weights but fast input
        {"drive_scale": 1e-8, "gws": 1e-7, "sim_steps": 30},
        {"drive_scale": 1e-8, "gws": 5e-7, "sim_steps": 30},
    ]

    for cfg in configs:
        label = f"ds={cfg['drive_scale']:.0e} gws={cfg['gws']:.0e} steps={cfg['sim_steps']}"
        try:
            run_one(
                drive_scale=cfg["drive_scale"],
                gws=cfg["gws"],
                trials=400,
                sim_steps=cfg["sim_steps"],
                label=label,
            )
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
