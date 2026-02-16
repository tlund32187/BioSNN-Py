"""Minimal debug: trace weight changes across 5 OR trials."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from biosnn.tasks.logic_gates import (
    LogicGate,
    LogicGateRunConfig,
    run_logic_gate_engine,
)


def main() -> None:
    out_dir = Path(tempfile.mkdtemp(prefix="biosnn_debug_or_"))
    print(f"Output: {out_dir}")

    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=42,
        steps=5,
        sim_steps_per_trial=15,
        device="cpu",  # CPU for easier debugging
        learning_mode="rstdp",
        engine_learning_rule="three_factor_elig_stdp",
        debug=True,
        out_dir=out_dir,
        artifacts_root=out_dir,
        export_every=1,
    )
    spec = {
        "dtype": "float64",
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
            "lr": 0.1,  # High lr to make changes visible
            "w_min": 0.0,
            "w_max": 1.0,
            "synaptic_scaling": False,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": 1.0,
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
            "exploration": {"enabled": False},
            "action_force": {
                "enabled": True,
                "mode": "always",
                "amplitude": 1.0,
                "window": "reward_window",
                "steps": 1,
                "compartment": "soma",
            },
            "gate_context": {
                "enabled": True,
                "amplitude": 0.30,
                "compartment": "dendrite",
                "targets": ["hidden"],
            },
        },
    }

    result = run_logic_gate_engine(cfg, spec)

    # Read and display the trial CSV
    import csv

    trials_csv = out_dir / "trials.csv"
    with open(trials_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["trial"]
            case = row["case_idx"]
            target = row["target"]
            pred = row["pred"]
            correct = row["correct"]
            dw = float(row["mean_abs_dw"])
            w0 = float(row["w_out0_mean"])
            w1 = float(row["w_out1_mean"])
            wmin = float(row["weights_min"])
            wmax = float(row["weights_max"])
            elig0 = float(row["elig_out0_mean"])
            elig1 = float(row["elig_out1_mean"])
            dopa0 = float(row["dopa_out0_mean"])
            dopa1 = float(row["dopa_out1_mean"])
            accum = float(row["dw_accum_abs_sum"])
            af = row["action_forced"]

            print(
                f"T{t} case={case} tgt={target} pred={pred} "
                f"{'OK' if correct == '1' else 'NO':>2} af={af} "
                f"w0={w0:.6f} w1={w1:.6f} "
                f"e0={elig0:.4f} e1={elig1:.4f} "
                f"d0={dopa0:.4f} d1={dopa1:.4f} "
                f"dw={dw:.6f} accum={accum:.6f} "
                f"wmin={wmin:.6f} wmax={wmax:.6f}"
            )

    # Also check weights CSV
    weights_csv = out_dir / "weights.csv"
    if weights_csv.exists():
        print("\n--- Weight snapshots ---")
        with open(weights_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                proj = row["proj"]
                if "Out" in proj:
                    print(
                        f"  step={row['step']} {proj} pre={row['pre']} "
                        f"post={row['post']} w={float(row['w']):.8f}"
                    )


if __name__ == "__main__":
    main()
