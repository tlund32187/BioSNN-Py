"""Quick single OR gate test with optimized parameters."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def main() -> None:
    n_trials = 800
    device = "cuda"
    rds = 10  # reward delivery steps

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
        },
        "learning": {
            "enabled": True,
            "rule": "three_factor_elig_stdp",
            "lr": 0.01,
            "w_min": 0.0,
            "w_max": 1.0,
            "a_plus": 1.0,
            "a_minus": 1.0,
            "tau_e": 0.001,  # very fast eligibility decay — must match dt
            "modulator_threshold": 0.5,  # gate learning to reward window only
            "synaptic_scaling": False,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": 0.01,  # DA drops below 0.5 in ~7 steps
        },
        "wrapper": {"enabled": False},
        "homeostasis": {
            "enabled": True,
            "alpha": 0.01,
            "eta": 1e-3,
            "r_target": 0.05,
            "clamp_min": 0.0,
            "clamp_max": 0.05,
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
                "steps": rds,  # force for ENTIRE reward window
                "compartment": "soma",
                "suppression_factor": -3.0,  # strong suppression
            },
            "gate_context": {
                "enabled": True,
                "amplitude": 0.30,
                "compartment": "dendrite",
                "targets": ["hidden"],
            },
        },
    }

    out = Path(tempfile.mkdtemp(prefix="biosnn_ortarget_"))

    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=42,
        steps=n_trials,
        sim_steps_per_trial=15,
        device=device,
        learning_mode="rstdp",
        engine_learning_rule="three_factor_elig_stdp",
        debug=False,
        out_dir=out,
        artifacts_root=out,
        reward_delivery_steps=rds,
        reward_delivery_clamp_input=True,  # keep input alive => hidden fires => pre spikes
    )

    print(f"=== OR gate targeted test: {n_trials} trials, rds={rds}, device={device} ===")
    print("  tau_e=0.001, a_minus=1.0, threshold=0.5, decay_tau=0.01")
    print(f"  action_force.steps={rds}, clamp_input=True, suppression=-3.0")
    print(f"  out_dir={out}")

    t0 = time.perf_counter()
    result = run_logic_gate_engine(cfg, spec)
    elapsed = time.perf_counter() - t0

    acc = result.get("eval_accuracy", -1)
    passed = result.get("passed", False)
    fp = result.get("first_pass_trial", -1)
    dw = result.get("mean_abs_dw", 0.0)
    fp_str = str(fp) if fp and fp > 0 else "-"

    print(
        f"\n  acc={acc:.4f}  passed={passed}  first_pass={fp_str}  dw={dw:.4e}  time={elapsed:.1f}s"
    )

    # Read the trial CSV and show last 20 lines
    csv_path = out / "trials.csv"
    if csv_path.exists():
        import csv

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        print(f"\n  Last 20 trials (of {len(rows)}):")
        for row in rows[-20:]:
            t = int(row["trial"])
            corr = row.get("correct", "?")
            eval_acc = row.get("eval_accuracy", "?")
            w0 = row.get("w_out0_mean", "?")
            w1 = row.get("w_out1_mean", "?")
            e0 = row.get("elig_out0_mean", "?")
            e1 = row.get("elig_out1_mean", "?")
            dw_v = row.get("mean_abs_dw", "?")
            print(
                f"    t={t:4d} corr={corr} acc={eval_acc} w0={w0} w1={w1} e0={e0} e1={e1} dw={dw_v}"
            )


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    main()
