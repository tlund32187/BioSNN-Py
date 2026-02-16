"""Test learning with correct drive_scale=1.0 (fixing the zero-spike bug)."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_trial(
    *,
    seed: int = 42,
    trials: int = 400,
    gws: float = 5e-7,
    lr_mult: float = 0.08,
    drive_scale: float = 1.0,
    sim_steps: int = 10,
    af_amplitude: float = 1.0,
    label: str = "",
) -> dict[str, Any]:
    w_max = gws * 3.0
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
                "epsilon_decay_trials": 200,
                "tie_break": "random_among_max",
            },
            "action_force": {
                "enabled": True,
                "mode": "always",
                "amplitude": af_amplitude,
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
        drive_scale=drive_scale,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=5,
        reward_delivery_clamp_input=True,
        debug=True,
    )

    r = run_logic_gate_engine(cfg, spec)
    eval_acc = r["eval_accuracy"]
    preds = r["preds"]

    # Read last 4 trials from CSV to check spike counts
    out_dir = r["out_dir"]
    trial_csv = out_dir / "trials.csv"
    with open(trial_csv) as f:
        rows = list(csv.DictReader(f))

    by_case: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[int(row["case_idx"])].append(row)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  eval={eval_acc:.2f}  preds={preds}")
    for ci in range(4):
        last = by_case[ci][-1]
        s0 = float(last["out_spikes_0"])
        s1 = float(last["out_spikes_1"])
        hid = float(last.get("hidden_mean_spikes", 0))
        t = int(last["trial"])
        forced = int(float(last.get("action_forced", 0)))
        tie = last.get("tie_wta", "?")
        print(
            f"  Case {ci} (x0={last['x0']}, x1={last['x1']}): "
            f"spk0={s0:.0f} spk1={s1:.0f} hid={hid:.3f} "
            f"tie={tie} forced={forced} trial={t}"
        )

    # Check spike count progression: first 50 trials vs last 50
    first_spikes = [0.0, 0.0]
    last_spikes = [0.0, 0.0]
    n_first = n_last = 0
    for row in rows[:50]:
        first_spikes[0] += float(row["out_spikes_0"])
        first_spikes[1] += float(row["out_spikes_1"])
        n_first += 1
    for row in rows[-50:]:
        last_spikes[0] += float(row["out_spikes_0"])
        last_spikes[1] += float(row["out_spikes_1"])
        n_last += 1
    print(
        f"  First 50 avg spikes: spk0={first_spikes[0] / n_first:.2f} spk1={first_spikes[1] / n_first:.2f}"
    )
    print(
        f"  Last  50 avg spikes: spk0={last_spikes[0] / n_last:.2f} spk1={last_spikes[1] / n_last:.2f}"
    )

    # Check tie percentage
    n_ties = sum(1 for r in rows if r.get("tie_wta", "0") == "1")
    print(f"  Tie %: {100 * n_ties / len(rows):.1f}%")
    sys.stdout.flush()
    return {"eval": eval_acc, "preds": preds}


def main():
    print("=== Testing with drive_scale=1.0 (default, fixing zero-spike bug) ===")

    # Config 1: default drive_scale, original params
    r1 = run_trial(
        seed=42,
        trials=400,
        gws=5e-7,
        lr_mult=0.08,
        drive_scale=1.0,
        sim_steps=10,
        label="Config 1: drive_scale=1.0, gws=5e-7, lr=0.08",
    )

    # Config 2: different seed
    r2 = run_trial(
        seed=1,
        trials=400,
        gws=5e-7,
        lr_mult=0.08,
        drive_scale=1.0,
        sim_steps=10,
        label="Config 2: seed=1",
    )

    # Config 3: more sim_steps
    r3 = run_trial(
        seed=42,
        trials=400,
        gws=5e-7,
        lr_mult=0.08,
        drive_scale=1.0,
        sim_steps=15,
        label="Config 3: sim_steps=15",
    )

    # Config 4: smaller lr
    r4 = run_trial(
        seed=42,
        trials=400,
        gws=5e-7,
        lr_mult=0.01,
        drive_scale=1.0,
        sim_steps=10,
        label="Config 4: lr_mult=0.01",
    )

    # Config 5: supervised mode
    r5 = run_trial(
        seed=42,
        trials=400,
        gws=5e-7,
        lr_mult=0.08,
        drive_scale=1.0,
        sim_steps=10,
        label="Config 5: supervised af",
    )

    print("\n" + "=" * 70)
    print("SUMMARY:")
    for label, r in [
        ("C1 default", r1),
        ("C2 seed=1", r2),
        ("C3 steps=15", r3),
        ("C4 lr=0.01", r4),
        ("C5 supervised", r5),
    ]:
        print(f"  {label}: eval={r['eval']:.2f}  preds={r['preds']}")
    print("Done.")


if __name__ == "__main__":
    main()
