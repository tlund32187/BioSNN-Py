"""Test OR gate with skip-connection-only learning targets."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_test(label: str, targets: list[str], lr: float, sf: float, seed: int = 42) -> None:
    rds = 10
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
            "lr": lr,
            "w_min": 0.0,
            "w_max": 1.0,
            "a_plus": 1.0,
            "a_minus": 1.0,
            "tau_e": 0.005,
            "modulator_threshold": 0.5,
            "synaptic_scaling": False,
            "targets": targets,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": 0.01,
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
                "steps": rds,
                "compartment": "soma",
                "suppression_factor": sf,
            },
            "gate_context": {
                "enabled": True,
                "amplitude": 0.30,
                "compartment": "dendrite",
                "targets": ["hidden"],
            },
        },
    }

    out = Path(tempfile.mkdtemp(prefix="biosnn_orskip_"))
    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=seed,
        steps=800,
        sim_steps_per_trial=15,
        device="cuda",
        learning_mode="rstdp",
        engine_learning_rule="three_factor_elig_stdp",
        debug=False,
        out_dir=out,
        artifacts_root=out,
        reward_delivery_steps=rds,
        reward_delivery_clamp_input=True,
    )

    t0 = time.perf_counter()
    result = run_logic_gate_engine(cfg, spec)
    elapsed = time.perf_counter() - t0

    acc = result.get("eval_accuracy", -1)
    passed = result.get("passed", False)
    fp = result.get("first_pass_trial", -1)
    dw = result.get("mean_abs_dw", 0.0)
    fp_str = str(fp) if fp and fp > 0 else "-"
    print(
        f"  {label:50s} acc={acc:.2f} pass={str(passed):5s} "
        f"1st={fp_str:6s} dw={dw:.2e} ({elapsed:.1f}s)"
    )

    # Show last 10 trials
    csv_path = out / "trials.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        for row in rows[-5:]:
            t = int(row["trial"])
            corr = row.get("correct", "?")
            w0 = row.get("w_out0_mean", "?")
            w1 = row.get("w_out1_mean", "?")
            print(f"    t={t:4d} corr={corr} w0={w0} w1={w1}")


def main() -> None:
    print("=== OR gate: skip-connection learning ===\n")

    # Test various combinations
    experiments = [
        ("skip-only lr=0.001 sf=-3.0", ["In->OutSkip"], 0.001, -3.0, 42),
        ("skip-only lr=0.005 sf=-3.0", ["In->OutSkip"], 0.005, -3.0, 42),
        ("skip-only lr=0.01 sf=-3.0", ["In->OutSkip"], 0.01, -3.0, 42),
        ("skip+hidden lr=0.001 sf=-3.0", ["In->OutSkip", "HiddenExcit->Out"], 0.001, -3.0, 42),
        ("skip-only lr=0.001 sf=-5.0", ["In->OutSkip"], 0.001, -5.0, 42),
        ("skip-only lr=0.001 sf=-3.0 seed=7", ["In->OutSkip"], 0.001, -3.0, 7),
        ("all-default lr=0.001 sf=-3.0", [], 0.001, -3.0, 42),
    ]

    for label, targets, lr, sf, seed in experiments:
        try:
            run_test(label, targets, lr, sf, seed)
        except Exception as exc:
            print(f"  {label:50s} ERROR: {exc}")
        print()


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    main()
