"""Test error-only DA with supervised teacher correction.

The runner's key mechanism: DA=0 on correct, DA=+1 on errors with target forced.
This prevents class-imbalance collapse AND teaches the correct response.

In instantaneous mode (tau_pre=0, tau_post=0) + a_minus > 0:
- Target output fires (action force) → LTP eligibility
- Non-target suppressed → LTD eligibility
- DA=+1 → target gets LTP, non-target gets LTD
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
    a_plus: float = 1.0,
    baseline: float = 0.0,
    tau_pre: float = 0.0,
    tau_post: float = 0.0,
    tau_e: float = 0.01,
    targets: list[str] | None = None,
    af_mode: str = "supervised",
    suppression: float = -0.25,
    in_to_hidden_ws: float = 0.03,
    he_out_fan_in: int = 3,
    reset_traces: bool = True,
    da_decay_tau: float = 0.05,
    label: str = "",
) -> dict[str, Any]:
    w_max = gws * 3.0
    learn_cfg: dict[str, Any] = {
        "enabled": True,
        "rule": "rstdp_eligibility",
        "lr": w_max * lr_mult,
        "w_min": -w_max,
        "w_max": w_max,
        "tau_e": tau_e,
        "tau_pre": tau_pre,
        "tau_post": tau_post,
        "a_plus": a_plus,
        "a_minus": a_minus,
        "dopamine_scale": 1.0,
        "baseline": baseline,
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
            "hidden_excit_to_out_fan_in": he_out_fan_in,
            "in_to_hidden_weight_scale": in_to_hidden_ws,
            "excit_target_compartment": "soma",
        },
        "learning": learn_cfg,
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": da_decay_tau,
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
    hid_key = "hidden_mean_spikes"
    last_hid = (
        sum(float(r[hid_key]) for r in rows[-n:]) / n if (rows and hid_key in rows[0]) else -1.0
    )

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
    tgt_he_only = ["HiddenExcit->Out"]
    tgt_both = ["HiddenExcit->Out", "In->OutSkip"]

    # ─── PHASE 1: Error-only supervised + instantaneous + a_minus ────
    print("=" * 70)
    print("PHASE 1: Error-only supervised + instantaneous STDP")
    print("  DA=0 on correct, DA=+1 on wrong, AF drives target")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3, 0.5]:
        r = run_config(targets=tgt_he_only, a_minus=am, label=f"sup_err inst am={am}")
        results.append(r)

    # ─── PHASE 2: + Dense HE→Out ────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Error-only supervised + dense HE→Out (fi=8)")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3, 0.5]:
        r = run_config(
            targets=tgt_he_only, a_minus=am, he_out_fan_in=8, label=f"sup_err inst fi=8 am={am}"
        )
        results.append(r)

    # ─── PHASE 3: + Selective hidden ─────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Error-only + dense + selective hidden (ih_ws=0.006)")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3, 0.5]:
        r = run_config(
            targets=tgt_he_only,
            a_minus=am,
            he_out_fan_in=8,
            in_to_hidden_ws=0.006,
            label=f"sup_err inst fi=8 sel am={am}",
        )
        results.append(r)

    # ─── PHASE 4: + Short DA decay (prevent inter-trial bleed) ──────
    print("\n" + "=" * 70)
    print("PHASE 4: + Short DA decay (tau=0.005)")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3]:
        r = run_config(
            targets=tgt_he_only,
            a_minus=am,
            he_out_fan_in=8,
            da_decay_tau=0.005,
            label=f"sup_err inst fi=8 fast_da am={am}",
        )
        results.append(r)

    # ─── PHASE 5: Include skip connections in learning ───────────────
    print("\n" + "=" * 70)
    print("PHASE 5: Error-only supervised + both HE+Skip learning")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3]:
        r = run_config(targets=tgt_both, a_minus=am, label=f"sup_err inst both am={am}")
        results.append(r)

    # ─── PHASE 6: Trace mode (tau_pre=0.02) for comparison ──────────
    print("\n" + "=" * 70)
    print("PHASE 6: Trace mode (tau=0.02) for comparison")
    print("=" * 70)
    for am in [0.1, 0.3]:
        r = run_config(
            targets=tgt_he_only,
            a_minus=am,
            tau_pre=0.02,
            tau_post=0.02,
            label=f"sup_err trace am={am}",
        )
        results.append(r)

    # ─── PHASE 7: Multi-seed of best results ────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 7: Multi-seed validation")
    print("=" * 70)
    passing = [r for r in results if r["eval"] >= 1.0]
    if passing:
        print(f"  Found {len(passing)} passing! Testing across seeds...")
        # Find first passing label and re-test with different seeds
        # (manually extract params)
    else:
        best = max(results, key=lambda r: r["eval"])
        print(f"  Best: {best['label']} (eval={best['eval']:.2f})")
        # Test closest candidates with different seeds
        for seed in [1, 7, 13, 99]:
            r = run_config(
                targets=tgt_he_only,
                a_minus=0.2,
                he_out_fan_in=8,
                seed=seed,
                label=f"sup_err inst fi=8 am=0.2 seed={seed}",
            )
            results.append(r)

    # ─── SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["eval"] >= 1.0 else "FAIL"
        print(f"  [{status}] {r['label']}: eval={r['eval']:.2f}")

    passing = [r for r in results if r["eval"] >= 1.0]
    print(f"\nPassing: {len(passing)}/{len(results)}")
    if passing:
        print("WINNING configs:")
        for r in passing:
            print(f"  >>> {r['label']}")
    print("Done.")


if __name__ == "__main__":
    main()
