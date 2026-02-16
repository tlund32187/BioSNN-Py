"""Run the best OR gate config for 800 trials and dump trial CSV for analysis."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def main() -> None:
    gws = 5e-6
    w_max = gws * 3.0
    lr = w_max * 0.01
    sim_steps = 30
    trials = 800

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
        drive_scale=1e-9,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=10,
        reward_delivery_clamp_input=True,
        debug=False,
        dump_last_trials_csv=True,
    )

    print(
        f"Running OR gate: gws={gws}, lr={lr:.2e}, w_max={w_max:.2e}, "
        f"sim_steps={sim_steps}, trials={trials}, drive_scale=1e-9"
    )
    print("  receptor_mode=exc_only, action_force amp=1.0 supp=-3.0")

    result = run_logic_gate_engine(cfg, spec)

    eval_acc = result.get("eval_accuracy", 0.0)
    sample_acc = result.get("sample_accuracy", 0.0)
    preds = result.get("preds", None)
    passed = result.get("passed", False)
    first_pass = result.get("first_pass_trial", None)
    mean_dw = result.get("mean_abs_dw", None)

    print("\n=== RESULTS ===")
    print(f"  eval_accuracy:    {eval_acc:.4f}")
    print(f"  sample_accuracy:  {sample_acc:.4f}")
    print(f"  predictions:      {preds}")
    print(f"  passed:           {passed}")
    print(f"  first_pass_trial: {first_pass}")
    print(f"  mean_abs_dw:      {mean_dw}")

    # Check for trials CSV
    trials_csv = result.get("trials_csv", None)
    if trials_csv and os.path.exists(trials_csv):
        print("\n=== TRIAL CSV ANALYSIS ===")
        print(f"  CSV at: {trials_csv}")
        with open(trials_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"  Total rows: {len(rows)}")

        # Show first and last 10 trials
        cols = ["trial", "case_idx", "target", "action", "correct", "out_spikes_0", "out_spikes_1"]
        available_cols = [c for c in cols if c in rows[0]]
        print(f"\n  Available columns: {list(rows[0].keys())[:15]}...")

        print("\n  First 10 trials:")
        for row in rows[:10]:
            print("    " + " | ".join(f"{c}={row.get(c, '?')}" for c in available_cols))

        print("\n  Last 10 trials:")
        for row in rows[-10:]:
            print("    " + " | ".join(f"{c}={row.get(c, '?')}" for c in available_cols))

        # Compute rolling accuracy over windows
        window = 100
        if len(rows) >= window:
            correct_vals = [float(row.get("correct", "0")) for row in rows]
            print(f"\n  Rolling {window}-trial accuracy:")
            for start in range(0, len(rows) - window + 1, window):
                end = start + window
                acc = sum(correct_vals[start:end]) / window
                print(f"    Trials {start:4d}-{end:4d}: {acc:.3f}")

        # Spike count summary per case
        print("\n  Spike counts by case (last 200 trials):")
        last_rows = rows[-200:]
        for case_idx in ["0", "1", "2", "3"]:
            case_rows = [r for r in last_rows if r.get("case_idx") == case_idx]
            if not case_rows:
                continue
            spk0 = [float(r.get("out_spikes_0", "0")) for r in case_rows]
            spk1 = [float(r.get("out_spikes_1", "0")) for r in case_rows]
            avg0 = sum(spk0) / len(spk0) if spk0 else 0
            avg1 = sum(spk1) / len(spk1) if spk1 else 0
            target = case_rows[0].get("target", "?")
            print(
                f"    case={case_idx} target={target}: avg_spk0={avg0:.1f} avg_spk1={avg1:.1f} ({len(case_rows)} trials)"
            )
    else:
        print(f"  No trials CSV found: {trials_csv}")


if __name__ == "__main__":
    main()
