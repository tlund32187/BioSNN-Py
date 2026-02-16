"""Systematic sweep with drive_scale=1.0 (the fix), varying sim_steps and delay."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_config(
    *,
    seed: int = 42,
    trials: int = 800,
    gws: float = 5e-7,
    lr_mult: float = 0.08,
    sim_steps: int = 20,
    delay: int = 1,
    af_amp: float = 1.0,
    drive_scale: float = 1.0,
    af_mode: str = "always",
    suppression: float = -0.25,
    label: str = "",
) -> dict[str, Any]:
    w_max = gws * 3.0
    spec: dict[str, Any] = {
        "dtype": "float32",
        "delay_steps": delay,
        "synapse": {
            "backend": "spmm_fused",
            "fused_layout": "auto",
            "ring_strategy": "dense",
            "ring_dtype": "none",
            "receptor_mode": "exc_only",
            "global_weight_scale": gws,
            "skip_fan_in": 2,
            "in_to_hidden_fan_in": 2,
            "excit_target_compartment": "soma",
        },
        "learning": {
            "enabled": True,
            "rule": "rstdp_eligibility",
            "lr": w_max * lr_mult,
            "w_min": -w_max,
            "w_max": w_max,
            "tau_e": 0.01,
            "tau_pre": 0.020,
            "tau_post": 0.020,
            "a_plus": 1.0,
            "a_minus": 0.0,
            "dopamine_scale": 1.0,
            "baseline": 0.0,
            "weight_decay": 0.0,
            "synaptic_scaling": False,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": 0.05,
        },
        "wrapper": {"enabled": True, "spike_window": sim_steps, "decision_mode": "spike_count"},
        "homeostasis": {"enabled": True, "alpha": 0.01, "eta": 1e-3, "r_target": 0.05},
        "logic": {
            "exploration": {
                "enabled": True,
                "epsilon_start": 0.3,
                "epsilon_end": 0.01,
                "epsilon_decay_trials": 400,
                "tie_break": "random_among_max",
            },
            "action_force": {
                "enabled": True,
                "mode": af_mode,
                "amplitude": af_amp,
                "window": "reward_window",
                "steps": 5,
                "compartment": "soma",
                "suppression_factor": suppression,
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
        reward_delivery_steps=5,
        reward_delivery_clamp_input=True,
        debug=True,
    )

    r = run_logic_gate_engine(cfg, spec)
    eval_acc = r["eval_accuracy"]
    preds = r["preds"]

    # Quick stats from CSV
    out_dir = r["out_dir"]
    trial_csv = out_dir / "trials.csv"
    with open(trial_csv) as f:
        rows = list(csv.DictReader(f))

    n_ties = sum(1 for row in rows if row.get("tie_wta", "0") == "1")
    first_spikes = [0.0, 0.0]
    last_spikes = [0.0, 0.0]
    first_hid = last_hid = 0.0
    n = min(50, len(rows))
    for row in rows[:n]:
        first_spikes[0] += float(row["out_spikes_0"])
        first_spikes[1] += float(row["out_spikes_1"])
        first_hid += float(row.get("hidden_mean_spikes", 0))
    for row in rows[-n:]:
        last_spikes[0] += float(row["out_spikes_0"])
        last_spikes[1] += float(row["out_spikes_1"])
        last_hid += float(row.get("hidden_mean_spikes", 0))

    print(f"  {label}")
    print(f"    eval={eval_acc:.2f}  preds={preds}  tie%={100 * n_ties / len(rows):.0f}%")
    print(
        f"    first50: spk0={first_spikes[0] / n:.1f} spk1={first_spikes[1] / n:.1f} hid={first_hid / n:.3f}"
    )
    print(
        f"    last50:  spk0={last_spikes[0] / n:.1f} spk1={last_spikes[1] / n:.1f} hid={last_hid / n:.3f}"
    )
    sys.stdout.flush()
    return {"eval": eval_acc, "preds": preds, "label": label}


def main():
    results = []
    print("=" * 70)
    print("Phase 1: sim_steps and delay sweep (seed=42, 800 trials)")
    print("=" * 70)

    # Baseline configs with different sim_steps and delays
    for sim_steps in [15, 20, 30]:
        for delay in [1, 2, 3]:
            label = f"sim={sim_steps} delay={delay}"
            r = run_config(sim_steps=sim_steps, delay=delay, label=label)
            results.append(r)

    print("\n" + "=" * 70)
    print("Phase 2: lr_mult and af_mode sweep")
    print("=" * 70)

    # Best sim_steps/delay from phase 1 + different learning params
    for lr_mult in [0.01, 0.05, 0.15]:
        r = run_config(sim_steps=20, delay=1, lr_mult=lr_mult, label=f"sim=20 delay=1 lr={lr_mult}")
        results.append(r)

    # Supervised mode
    r = run_config(sim_steps=20, delay=1, af_mode="supervised", label="sim=20 delay=1 supervised")
    results.append(r)

    print("\n" + "=" * 70)
    print("Phase 3: seed sweep (best config)")
    print("=" * 70)
    for seed in [1, 7, 13, 42, 99]:
        r = run_config(seed=seed, sim_steps=20, delay=1, label=f"sim=20 delay=1 seed={seed}")
        results.append(r)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['label']}: eval={r['eval']:.2f}  preds={r['preds']}")

    passing = [r for r in results if r["eval"] >= 1.0]
    print(f"\nPassing configs: {len(passing)}/{len(results)}")
    if passing:
        for r in passing:
            print(f"  PASS: {r['label']}")
    print("Done.")


if __name__ == "__main__":
    main()
