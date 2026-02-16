"""Test instantaneous STDP mode (tau_pre=0, tau_post=0) with a_minus > 0.

In instantaneous mode: inc = a_plus × (pre × post) - a_minus × pre × (1-post)
- When post fires (target output, action force): inc = +a_plus × pre (LTP)
- When post doesn't fire (non-target, suppressed): inc = -a_minus × pre (LTD!)

This provides anti-Hebbian LTD for the non-target output, which is the missing
piece for discriminative learning. In trace mode, the LTD term requires recent
post spikes (post_trace > 0) which the suppressed output never has.
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
    targets: list[str] | None = None,
    af_mode: str = "always",
    suppression: float = -0.25,
    in_to_hidden_ws: float = 0.03,
    he_out_fan_in: int = 3,
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
    tgt = ["HiddenExcit->Out", "In->OutSkip"]
    tgt_he_only = ["HiddenExcit->Out"]

    # ─── PHASE 1: Instantaneous mode (tau=0) + a_minus sweep ─────────
    print("=" * 70)
    print("PHASE 1: Instantaneous STDP + a_minus (a_minus × pre × (1-post))")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3, 0.5]:
        r = run_config(targets=tgt, a_minus=am, tau_pre=0.0, tau_post=0.0, label=f"inst am={am}")
        results.append(r)

    # ─── PHASE 2: Inst + supervised (both LTP target + LTD non-target) ─
    print("\n" + "=" * 70)
    print("PHASE 2: Instantaneous + supervised (target gets LTP, non-target LTD)")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3, 0.5]:
        r = run_config(
            targets=tgt_he_only,
            a_minus=am,
            af_mode="supervised",
            tau_pre=0.0,
            tau_post=0.0,
            label=f"inst supervised he am={am}",
        )
        results.append(r)

    # ─── PHASE 3: Inst + supervised + dense HE→Out ───────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Inst + supervised + dense HE→Out (fi=8)")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3, 0.5]:
        r = run_config(
            targets=tgt_he_only,
            a_minus=am,
            af_mode="supervised",
            tau_pre=0.0,
            tau_post=0.0,
            he_out_fan_in=8,
            label=f"inst sup fi=8 am={am}",
        )
        results.append(r)

    # ─── PHASE 4: Inst + supervised + dense + selective hidden ───────
    print("\n" + "=" * 70)
    print("PHASE 4: Inst + supervised + dense + selective hidden (ih_ws=0.006)")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3, 0.5]:
        r = run_config(
            targets=tgt_he_only,
            a_minus=am,
            af_mode="supervised",
            tau_pre=0.0,
            tau_post=0.0,
            he_out_fan_in=8,
            in_to_hidden_ws=0.006,
            label=f"inst sup fi=8 sel am={am}",
        )
        results.append(r)

    # ─── PHASE 5: Inst + always mode + baseline ──────────────────────
    print("\n" + "=" * 70)
    print("PHASE 5: Inst + always mode + baseline + a_minus")
    print("=" * 70)
    for am in [0.1, 0.2, 0.3]:
        r = run_config(
            targets=tgt,
            a_minus=am,
            baseline=-0.5,
            tau_pre=0.0,
            tau_post=0.0,
            label=f"inst bl=-0.5 am={am}",
        )
        results.append(r)
    # Also with HE-only + dense
    for am in [0.1, 0.2, 0.3]:
        r = run_config(
            targets=tgt_he_only,
            a_minus=am,
            baseline=-0.5,
            tau_pre=0.0,
            tau_post=0.0,
            he_out_fan_in=8,
            label=f"inst bl=-0.5 fi=8 am={am}",
        )
        results.append(r)

    # ─── PHASE 6: Multi-seed validation of best ─────────────────────
    print("\n" + "=" * 70)
    print("PHASE 6: Multi-seed validation")
    print("=" * 70)
    passing = [r for r in results if r["eval"] >= 1.0]
    if passing:
        print(f"  Found {len(passing)} passing config(s). Testing best across seeds...")
    else:
        # Try the configs closest to passing with multiple seeds
        print("  No passing configs. Testing closest candidates with multiple seeds.")
        # Try inst supervised he am=0.2 with multiple seeds as it's most promising
        for seed in [1, 7, 13, 99]:
            r = run_config(
                targets=tgt_he_only,
                a_minus=0.2,
                af_mode="supervised",
                tau_pre=0.0,
                tau_post=0.0,
                he_out_fan_in=8,
                seed=seed,
                label=f"inst sup fi=8 am=0.2 seed={seed}",
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
