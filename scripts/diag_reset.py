"""Diagnostic: detailed per-trial analysis at gws=-8 with inter-trial reset."""

from __future__ import annotations

import csv as csvmod
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def main() -> None:
    gws = 1e-8
    w_max = gws * 3.0
    lr = w_max * 0.01

    spec: dict[str, Any] = {
        "dtype": "float32",
        "delay_steps": 3,
        "synapse": {
            "backend": "spmm_fused",
            "fused_layout": "auto",
            "ring_strategy": "dense",
            "ring_dtype": "none",
            "store_sparse_by_delay": None,
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
            "tau_e": 0.001,
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
        "wrapper": {"enabled": False},
        "homeostasis": {"enabled": False},
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
            "gate_context": {
                "enabled": True,
                "amplitude": 0.30,
                "compartment": "dendrite",
                "targets": ["hidden"],
            },
        },
    }

    out = Path(tempfile.mkdtemp(prefix="biosnn_diag_"))
    print(f"Output: {out}")
    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=42,
        steps=400,
        sim_steps_per_trial=15,
        device="cuda",
        learning_mode="rstdp",
        engine_learning_rule="three_factor_elig_stdp",
        debug=True,
        out_dir=out,
        artifacts_root=out,
        reward_delivery_steps=10,
        reward_delivery_clamp_input=True,
        inter_trial_reset=True,
    )
    result = run_logic_gate_engine(config=cfg, run_spec=spec)
    print(f"eval={result.get('eval_accuracy', 0)}, sample={result.get('sample_accuracy', 0)}")

    # Analyze per-trial data
    csv_path = out / "trials.csv"
    with open(csv_path) as f:
        rows = list(csvmod.DictReader(f))

    # Show first 40 trials
    print("\n=== FIRST 40 TRIALS ===")
    print(
        f"{'trial':>5} {'case':>4} {'s0':>5} {'s1':>5} {'pred':>4} {'tgt':>3} {'corr':>4} "
        f"{'DA':>6} {'dW':>10} {'w_mean':>10} {'elig':>10} {'explored':>3} {'tied':>4}"
    )
    for row in rows[:40]:
        print(
            f"{int(row['trial']):5d} {int(row['case_idx']):4d} "
            f"{float(row['out_spikes_0']):5.1f} {float(row['out_spikes_1']):5.1f} "
            f"{int(row['pred']):4d} {int(row['target']):3d} {int(row['correct']):4d} "
            f"{float(row['dopamine_pulse']):6.2f} "
            f"{float(row['mean_abs_dw']):10.2e} "
            f"{float(row['weights_mean']):10.2e} "
            f"{float(row['mean_eligibility_abs']):10.2e} "
            f"{int(row['explored']):3d} "
            f"{int(row['tie_wta']):4d}"
        )

    # Per-case accuracy over time (windows of 20 cycles = 80 trials)
    print("\n=== PER-CASE ACCURACY OVER TIME (20-cycle windows) ===")
    window = 80  # 20 complete cycles
    for start in range(0, len(rows), window):
        chunk = rows[start : start + window]
        if len(chunk) < 4:
            break
        per_case = {0: [], 1: [], 2: [], 3: []}
        for r in chunk:
            per_case[int(r["case_idx"])].append(int(r["correct"]))
        accs = {ci: sum(lst) / max(1, len(lst)) for ci, lst in per_case.items()}
        print(
            f"  trials {start + 1:4d}-{start + len(chunk):4d}: "
            f"case0={accs[0]:.2f}  case1={accs[1]:.2f}  "
            f"case2={accs[2]:.2f}  case3={accs[3]:.2f}"
        )

    # Show last 20 trials
    print("\n=== LAST 20 TRIALS ===")
    for row in rows[-20:]:
        print(
            f"{int(row['trial']):5d} {int(row['case_idx']):4d} "
            f"{float(row['out_spikes_0']):5.1f} {float(row['out_spikes_1']):5.1f} "
            f"{int(row['pred']):4d} {int(row['target']):3d} {int(row['correct']):4d} "
            f"{float(row['dopamine_pulse']):6.2f}"
        )

    # Weight evolution
    print("\n=== WEIGHT EVOLUTION ===")
    for i in [0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4, len(rows) - 1]:
        r = rows[i]
        # Check for per-output weight columns
        w_cols = [k for k in r.keys() if "w_out" in k]
        w_info = "  ".join(f"{c}={float(r[c]):.2e}" for c in w_cols[:6])
        print(
            f"  trial {int(r['trial']):4d}: w_mean={float(r['weights_mean']):.2e} "
            f"w_min={float(r['weights_min']):.2e} w_max={float(r['weights_max']):.2e} "
            f"{w_info}"
        )


if __name__ == "__main__":
    main()
