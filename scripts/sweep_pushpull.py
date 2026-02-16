"""Test push-pull mechanisms: synaptic scaling, weight decay, a_minus."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
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
    a_minus: float = 0.0,
    synaptic_scaling: bool = False,
    weight_decay: float = 0.0,
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
        "a_minus": a_minus,
        "dopamine_scale": 1.0,
        "baseline": 0.0,
        "weight_decay": weight_decay,
        "synaptic_scaling": synaptic_scaling,
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
    last_s0 = sum(float(r["out_spikes_0"]) for r in rows[-n:]) / n
    last_s1 = sum(float(r["out_spikes_1"]) for r in rows[-n:]) / n
    last_hid = sum(float(r.get("hidden_mean_spikes", 0)) for r in rows[-n:]) / n

    # Check per-case predictions in last 100 trials
    by_case: dict[int, list] = defaultdict(list)
    for row in rows[-100:]:
        ci = int(row["case_idx"])
        by_case[ci].append(int(float(row["pred"])))
    case_acc = {}
    targets_or = {0: 0, 1: 1, 2: 1, 3: 1}
    for ci in range(4):
        if by_case[ci]:
            correct = sum(1 for p in by_case[ci] if p == targets_or[ci])
            case_acc[ci] = correct / len(by_case[ci])
        else:
            case_acc[ci] = 0.0

    print(f"  {label}")
    print(f"    eval={eval_acc:.2f}  preds={preds}  tie%={100 * n_ties / len(rows):.0f}%")
    print(f"    last50: s0={last_s0:.1f} s1={last_s1:.1f} hid={last_hid:.3f}")
    print(
        f"    case_acc: c0={case_acc[0]:.2f} c1={case_acc[1]:.2f} "
        f"c2={case_acc[2]:.2f} c3={case_acc[3]:.2f}"
    )
    sys.stdout.flush()
    return {"eval": eval_acc, "preds": preds, "label": label}


def main():
    results = []
    tgt = ["HiddenExcit->Out", "In->OutSkip"]

    print("=" * 70)
    print("PHASE 1: Push-pull mechanisms (seed=42)")
    print("=" * 70)

    # A: Synaptic scaling
    r = run_config(targets=tgt, synaptic_scaling=True, label="synaptic_scaling=True")
    results.append(r)

    # B: Weight decay
    for wd in [1e-4, 1e-3, 1e-2]:
        r = run_config(targets=tgt, weight_decay=wd, label=f"weight_decay={wd}")
        results.append(r)

    # C: a_minus > 0
    for am in [0.3, 0.6, 1.0]:
        r = run_config(targets=tgt, a_minus=am, label=f"a_minus={am}")
        results.append(r)

    # D: Combined: synaptic_scaling + a_minus
    r = run_config(targets=tgt, synaptic_scaling=True, a_minus=0.6, label="scaling + a_minus=0.6")
    results.append(r)

    # E: Combined: weight_decay + a_minus
    r = run_config(targets=tgt, weight_decay=1e-3, a_minus=0.6, label="wd=1e-3 + a_minus=0.6")
    results.append(r)

    # F: Combined all three
    r = run_config(
        targets=tgt,
        synaptic_scaling=True,
        weight_decay=1e-3,
        a_minus=0.6,
        label="all three combined",
    )
    results.append(r)

    print("\n" + "=" * 70)
    print("PHASE 2: Supervised mode with push-pull")
    print("=" * 70)

    r = run_config(
        targets=tgt, synaptic_scaling=True, af_mode="supervised", label="supervised + scaling"
    )
    results.append(r)

    r = run_config(targets=tgt, a_minus=0.6, af_mode="supervised", label="supervised + a_minus=0.6")
    results.append(r)

    r = run_config(
        targets=tgt, weight_decay=1e-3, af_mode="supervised", label="supervised + wd=1e-3"
    )
    results.append(r)

    print("\n" + "=" * 70)
    print("PHASE 3: Best config across seeds")
    print("=" * 70)
    # Identify best from Phase 1/2 and sweep seeds
    best = max(results, key=lambda x: x["eval"])
    print(f"  Best so far: {best['label']} eval={best['eval']:.2f}")

    # Run best params across seeds
    # (Hardcode the best combo - adjust based on Phase 1/2 results)
    for seed in [1, 7, 13, 42, 99]:
        r = run_config(
            seed=seed,
            targets=tgt,
            synaptic_scaling=True,
            a_minus=0.6,
            label=f"scaling+am0.6 seed={seed}",
        )
        results.append(r)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["eval"] >= 1.0 else "FAIL"
        print(f"  [{status}] {r['label']}: eval={r['eval']:.2f}  preds={r['preds']}")

    passing = [r for r in results if r["eval"] >= 1.0]
    print(f"\nPassing: {len(passing)}/{len(results)}")
    print("Done.")


if __name__ == "__main__":
    main()
