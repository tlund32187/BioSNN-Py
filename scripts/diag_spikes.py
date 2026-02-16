"""Quick check: are output spike counts always tied?"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_and_check(
    *,
    seed: int = 42,
    trials: int = 100,
    learning: bool = True,
    sim_steps: int = 10,
    gws: float = 5e-7,
    label: str = "",
) -> None:
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
            "enabled": learning,
            "rule": "rstdp_eligibility",
            "lr": w_max * 0.08,
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
        "homeostasis": {"enabled": True, "target_rate": 0.05, "tau": 1.0, "gain": 0.001},
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
                "amplitude": 1.0,
                "window": "reward_window",
                "steps": 5,
                "compartment": "soma",
                "suppression_factor": -3.0,
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
        drive_scale=1e-9,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=5,
        reward_delivery_clamp_input=True,
        debug=True,
    )

    r = run_logic_gate_engine(cfg, spec)
    out_dir = r["out_dir"]
    trial_csv = out_dir / "trials.csv"

    print(f"\n{'=' * 70}")
    print(f"  {label}  eval={r['eval_accuracy']:.2f}  preds={r['preds']}")
    print("  Last trials spike counts:")

    # Read all trial rows
    with open(trial_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by case, show last 3 per case
    from collections import defaultdict

    by_case: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[int(row["case_idx"])].append(row)

    for ci in range(4):
        trials_for_case = by_case[ci]
        print(f"  Case {ci} (x0={trials_for_case[0]['x0']}, x1={trials_for_case[0]['x1']}):")
        for row in trials_for_case[-3:]:
            s0 = float(row["out_spikes_0"])
            s1 = float(row["out_spikes_1"])
            pred = int(float(row["pred"]))
            corr = int(float(row["correct"]))
            expl = int(float(row.get("explored", 0)))
            tie = row.get("tie_wta", "")
            eps = float(row.get("epsilon", 0))
            trial = int(row["trial"])
            print(
                f"    t={trial:4d}: spk0={s0:5.1f}  spk1={s1:5.1f}  "
                f"pred={pred}  correct={corr}  explored={expl}  "
                f"tie={tie}  eps={eps:.3f}"
            )

    print()
    sys.stdout.flush()


def main() -> None:
    # Baseline: no learning
    run_and_check(seed=42, trials=100, learning=False, label="BASELINE seed=42")
    run_and_check(seed=42, trials=100, learning=True, label="LEARNED  seed=42")
    run_and_check(seed=1, trials=100, learning=False, label="BASELINE seed=1")
    run_and_check(seed=1, trials=100, learning=True, label="LEARNED  seed=1")

    # Try different gws
    run_and_check(seed=42, trials=100, learning=False, gws=2e-7, label="BASELINE seed=42 gws=2e-7")
    run_and_check(seed=42, trials=100, learning=True, gws=2e-7, label="LEARNED  seed=42 gws=2e-7")

    print("Done.")


if __name__ == "__main__":
    main()
