"""Test with learning restricted to output-targeting projections only."""

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
    targets: list[str] | None = None,
    af_mode: str = "always",
    label: str = "",
) -> dict[str, Any]:
    w_max = gws * 3.0
    learn_cfg: dict[str, Any] = {
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
    }
    if targets is not None:
        learn_cfg["targets"] = targets

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
        "learning": learn_cfg,
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
                "amplitude": 1.0,
                "window": "reward_window",
                "steps": 5,
                "compartment": "soma",
                "suppression_factor": -0.25,
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
        drive_scale=1.0,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=5,
        reward_delivery_clamp_input=True,
        debug=True,
    )

    r = run_logic_gate_engine(cfg, spec)
    eval_acc = r["eval_accuracy"]
    preds = r["preds"]

    out_dir = r["out_dir"]
    trial_csv = out_dir / "trials.csv"
    with open(trial_csv) as f:
        rows = list(csv.DictReader(f))

    n_ties = sum(1 for row in rows if row.get("tie_wta", "0") == "1")
    n = min(50, len(rows))
    first_hid = sum(float(r.get("hidden_mean_spikes", 0)) for r in rows[:n]) / n
    last_hid = sum(float(r.get("hidden_mean_spikes", 0)) for r in rows[-n:]) / n
    first_s0 = sum(float(r["out_spikes_0"]) for r in rows[:n]) / n
    last_s0 = sum(float(r["out_spikes_0"]) for r in rows[-n:]) / n
    first_s1 = sum(float(r["out_spikes_1"]) for r in rows[:n]) / n
    last_s1 = sum(float(r["out_spikes_1"]) for r in rows[-n:]) / n

    print(f"  {label}")
    print(f"    eval={eval_acc:.2f}  preds={preds}  tie%={100 * n_ties / len(rows):.0f}%")
    print(f"    first50: s0={first_s0:.1f} s1={first_s1:.1f} hid={first_hid:.3f}")
    print(f"    last50:  s0={last_s0:.1f} s1={last_s1:.1f} hid={last_hid:.3f}")
    sys.stdout.flush()
    return {"eval": eval_acc, "preds": preds, "label": label}


def main():
    results = []
    print("=" * 70)
    print("PHASE 1: Learning targets comparison")
    print("=" * 70)

    # A: Default targets (all non-inhibitory) — should fail
    r = run_config(
        targets=None, sim_steps=20, delay=1, label="default targets (In->H + HE->Out + Skip)"
    )
    results.append(r)

    # B: Only output-targeting projections — should preserve hidden layer
    r = run_config(
        targets=["HiddenExcit->Out", "In->OutSkip"],
        sim_steps=20,
        delay=1,
        label="targets=[HE->Out, Skip] (output only)",
    )
    results.append(r)

    # C: Only HE->Out
    r = run_config(
        targets=["HiddenExcit->Out"], sim_steps=20, delay=1, label="targets=[HE->Out] only"
    )
    results.append(r)

    print("\n" + "=" * 70)
    print("PHASE 2: Parameter variations with output-only targets")
    print("=" * 70)

    for lr in [0.01, 0.05, 0.15, 0.30]:
        r = run_config(
            targets=["HiddenExcit->Out", "In->OutSkip"],
            sim_steps=20,
            delay=1,
            lr_mult=lr,
            label=f"output-targets lr={lr}",
        )
        results.append(r)

    # More sim_steps
    r = run_config(
        targets=["HiddenExcit->Out", "In->OutSkip"],
        sim_steps=30,
        delay=1,
        label="output-targets sim=30",
    )
    results.append(r)

    # Supervised mode
    r = run_config(
        targets=["HiddenExcit->Out", "In->OutSkip"],
        sim_steps=20,
        delay=1,
        af_mode="supervised",
        label="output-targets supervised",
    )
    results.append(r)

    print("\n" + "=" * 70)
    print("PHASE 3: Seed sweep with best config")
    print("=" * 70)
    for seed in [1, 7, 13, 42, 99]:
        r = run_config(
            seed=seed,
            targets=["HiddenExcit->Out", "In->OutSkip"],
            sim_steps=20,
            delay=1,
            label=f"output-targets seed={seed}",
        )
        results.append(r)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["eval"] >= 1.0 else "FAIL"
        print(f"  [{status}] {r['label']}: eval={r['eval']:.2f}")

    passing = [r for r in results if r["eval"] >= 1.0]
    print(f"\nPassing: {len(passing)}/{len(results)}")
    print("Done.")


if __name__ == "__main__":
    main()
