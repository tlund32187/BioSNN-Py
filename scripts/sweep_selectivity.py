"""Sweep In→Hidden weight scale to improve hidden-layer selectivity.

Problem: With in_to_hidden_weight_scale=0.03 and gws=5e-7, a single presynaptic
spike drives 75 mV voltage change (threshold is 20 mV).  ALL hidden neurons
fire for ANY input → no discrimination → output converges to majority class.

Solution: Reduce in_to_hidden_weight_scale so hidden neurons need BOTH of their
fan_in=2 inputs to fire simultaneously (AND-like).  Target: single-spike gives
~10-15 mV (sub-threshold), two coincident give ~20-30 mV (supra-threshold).
Target weight_scale ≈ 0.004–0.008 instead of 0.03.
"""

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
    baseline: float = 0.0,
    synaptic_scaling: bool = False,
    weight_decay: float = 0.0,
    targets: list[str] | None = None,
    af_mode: str = "always",
    suppression: float = -0.25,
    reset_traces: bool = False,
    in_to_hidden_ws: float = 0.03,
    he_out_fan_in: int = 3,
    he_out_ws: float = 0.03,
    skip_fan_in: int = 2,
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
        "baseline": baseline,
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
            "skip_fan_in": skip_fan_in,
            "in_to_hidden_fan_in": 2,
            "hidden_excit_to_out_fan_in": he_out_fan_in,
            "in_to_hidden_weight_scale": in_to_hidden_ws,
            "hidden_excit_to_out_weight_scale": he_out_ws,
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
        drive_scale=1.0,
        reset_traces_between_trials=reset_traces,
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

    # Hidden spikes
    hid_key = "hidden_mean_spikes"
    if rows and hid_key in rows[0]:
        last_hid = sum(float(r[hid_key]) for r in rows[-n:]) / n
    else:
        last_hid = -1.0

    by_case: dict[int, list] = defaultdict(list)
    for row in rows[-100:]:
        ci = int(row["case_idx"])
        by_case[ci].append(int(float(row["pred"])))
    targets_or = {0: 0, 1: 1, 2: 1, 3: 1}
    case_strs = []
    for ci in range(4):
        if by_case[ci]:
            acc = sum(1 for p in by_case[ci] if p == targets_or[ci]) / len(by_case[ci])
            case_strs.append(f"c{ci}={acc:.0%}")
        else:
            case_strs.append(f"c{ci}=?")

    status = "PASS" if eval_acc >= 1.0 else "fail"
    print(f"  [{status}] {label}")
    print(
        f"    eval={eval_acc:.2f}  preds={preds}  tie%={100 * n_ties / len(rows):.0f}%  "
        f"s0={last_s0:.1f} s1={last_s1:.1f}  hid={last_hid:.3f}  {' '.join(case_strs)}"
    )
    sys.stdout.flush()
    return {"eval": eval_acc, "preds": preds, "label": label}


def main():
    results = []
    tgt = ["HiddenExcit->Out", "In->OutSkip"]
    tgt_he_only = ["HiddenExcit->Out"]

    # ─── PHASE 1: In→Hidden weight scale sweep ──────────────────────
    print("=" * 70)
    print("PHASE 1: In→Hidden weight_scale sweep (hidden selectivity)")
    print("  Current: 0.03 → 75mV/spike. Target: 0.004-0.008 → ~10-20mV")
    print("=" * 70)
    for ws in [0.03, 0.015, 0.008, 0.006, 0.004, 0.002]:
        r = run_config(targets=tgt, in_to_hidden_ws=ws, label=f"ih_ws={ws}")
        results.append(r)

    # ─── PHASE 2: Best ws + baseline ─────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Reduced weight_scale + baseline correction")
    print("=" * 70)
    for ws in [0.008, 0.006, 0.004]:
        r = run_config(targets=tgt, in_to_hidden_ws=ws, baseline=-0.5, label=f"ih_ws={ws} bl=-0.5")
        results.append(r)
        r = run_config(
            targets=tgt,
            in_to_hidden_ws=ws,
            baseline=-0.5,
            a_minus=0.3,
            label=f"ih_ws={ws} bl=-0.5 am=0.3",
        )
        results.append(r)

    # ─── PHASE 3: HE-only learning (no skip) ────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: HE→Out only learning (no skip connections learn)")
    print("=" * 70)
    for ws in [0.008, 0.006]:
        r = run_config(
            targets=tgt_he_only,
            in_to_hidden_ws=ws,
            baseline=-0.5,
            label=f"he_only ih_ws={ws} bl=-0.5",
        )
        results.append(r)
        r = run_config(
            targets=tgt_he_only,
            in_to_hidden_ws=ws,
            baseline=-0.5,
            a_minus=0.3,
            label=f"he_only ih_ws={ws} bl=-0.5 am=0.3",
        )
        results.append(r)

    # ─── PHASE 4: Dense HE→Out + reduced ih ──────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: Dense HE→Out (fi=8) + reduced In→Hidden")
    print("=" * 70)
    for ws in [0.008, 0.006, 0.004]:
        r = run_config(
            targets=tgt_he_only,
            in_to_hidden_ws=ws,
            he_out_fan_in=8,
            baseline=-0.5,
            label=f"he_fi=8 ih_ws={ws} bl=-0.5",
        )
        results.append(r)

    # ─── PHASE 5: Supervised + reduced ih ────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 5: Supervised mode + reduced In→Hidden")
    print("=" * 70)
    for ws in [0.008, 0.006, 0.004]:
        r = run_config(
            targets=tgt_he_only,
            in_to_hidden_ws=ws,
            af_mode="supervised",
            he_out_fan_in=8,
            label=f"supervised he_fi=8 ih_ws={ws}",
        )
        results.append(r)

    # ─── PHASE 6: Multiple seeds with best config ────────────────────
    print("\n" + "=" * 70)
    print("PHASE 6: If any passed, test across seeds")
    print("=" * 70)
    # Find best config
    passing = [r for r in results if r["eval"] >= 1.0]
    if passing:
        best_label = passing[0]["label"]
        print(f"  Testing best config '{best_label}' across seeds...")
        # Re-run with different seeds using parameters from best
        # (manual extraction - we'll know after phase 1-5 which params to use)
    else:
        print("  No passing config found in phases 1-5.")

    # ─── SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["eval"] >= 1.0 else "FAIL"
        print(f"  [{status}] {r['label']}: eval={r['eval']:.2f}  preds={r['preds']}")

    passing = [r for r in results if r["eval"] >= 1.0]
    print(f"\nPassing: {len(passing)}/{len(results)}")
    if passing:
        print("WINNING configs:")
        for r in passing:
            print(f"  >>> {r['label']}")
    print("Done.")


if __name__ == "__main__":
    main()
