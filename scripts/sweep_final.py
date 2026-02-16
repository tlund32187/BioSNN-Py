"""Fine-grained a_minus sweep + denser connectivity."""

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
    skip_fan_in: int = 2,
    hidden_fan_in: int = 2,
    he_out_fan_in: int = 3,
    suppression: float = -0.25,
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
            "skip_fan_in": skip_fan_in,
            "in_to_hidden_fan_in": hidden_fan_in,
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
        f"s0={last_s0:.1f} s1={last_s1:.1f}  {' '.join(case_strs)}"
    )
    sys.stdout.flush()
    return {"eval": eval_acc, "preds": preds, "label": label}


def main():
    results = []
    tgt = ["HiddenExcit->Out", "In->OutSkip"]

    print("=" * 70)
    print("PHASE 1: Fine a_minus sweep (0.7 to 0.95)")
    print("=" * 70)
    for am in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        r = run_config(targets=tgt, a_minus=am, label=f"a_minus={am}")
        results.append(r)

    print("\n" + "=" * 70)
    print("PHASE 2: Supervised + a_minus + scaling")
    print("=" * 70)
    for am in [0.5, 0.7, 0.9, 1.0]:
        r = run_config(
            targets=tgt,
            a_minus=am,
            af_mode="supervised",
            synaptic_scaling=True,
            label=f"supervised am={am} scaling",
        )
        results.append(r)

    print("\n" + "=" * 70)
    print("PHASE 3: Dense connectivity (fan_in=5/8)")
    print("=" * 70)
    # More skip connections (all inputs to each output)
    r = run_config(targets=tgt, skip_fan_in=5, hidden_fan_in=3, label="skip_fi=5 hid_fi=3")
    results.append(r)

    # More hidden->out connections
    r = run_config(
        targets=tgt, skip_fan_in=5, hidden_fan_in=3, a_minus=0.8, label="skip_fi=5 hid_fi=3 am=0.8"
    )
    results.append(r)

    # Full excitatory connectivity
    r = run_config(targets=tgt, skip_fan_in=5, hidden_fan_in=5, label="skip_fi=5 hid_fi=5")
    results.append(r)

    r = run_config(
        targets=tgt,
        skip_fan_in=5,
        hidden_fan_in=5,
        a_minus=0.8,
        synaptic_scaling=True,
        label="full_fi am=0.8 scaling",
    )
    results.append(r)

    r = run_config(
        targets=tgt,
        skip_fan_in=5,
        hidden_fan_in=5,
        a_minus=0.8,
        af_mode="supervised",
        label="full_fi am=0.8 supervised",
    )
    results.append(r)

    print("\n" + "=" * 70)
    print("PHASE 4: Higher suppression factor")
    print("=" * 70)
    for sup in [-1.0, -3.0, -5.0]:
        r = run_config(targets=tgt, suppression=sup, a_minus=0.8, label=f"sup={sup} am=0.8")
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
