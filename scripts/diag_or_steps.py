"""Per-step diagnostic for the first trial of OR gate to trace exact learning dynamics.

Instruments the engine to print per-step: DA level, output spikes, elig, and dW.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def main() -> None:
    """Run 2 trials and inspect all available CSV metrics."""
    device = "cuda"
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
            "lr": 0.01,
            "w_min": 0.0,
            "w_max": 1.0,
            "a_plus": 1.0,
            "a_minus": 1.0,
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
                "enabled": False,
            },
            "action_force": {
                "enabled": True,
                "mode": "always",
                "amplitude": 1.0,
                "window": "reward_window",
                "steps": rds,
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
    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=42,
        steps=2,
        sim_steps_per_trial=15,
        device=device,
        learning_mode="rstdp",
        engine_learning_rule="three_factor_elig_stdp",
        debug=False,
        out_dir=out,
        artifacts_root=out,
        reward_delivery_steps=rds,
        reward_delivery_clamp_input=True,
    )

    # We'll run it normally but hook into the engine to extract data
    # Actually, let's just run with 2 trials and capture the trial CSV
    result = run_logic_gate_engine(config=cfg, run_spec=spec)

    # Now load the trials CSV to see the weights
    import csv

    trials_csv = out / "trials.csv"
    if trials_csv.exists():
        with open(trials_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"\nTotal trials: {len(rows)}")
        print(f"\nCSV columns: {list(rows[0].keys()) if rows else 'N/A'}")

        for row in rows:
            trial = int(row["trial"])
            print(f"\n--- Trial {trial} ---")
            for key in sorted(row.keys()):
                val = row[key]
                try:
                    val_f = float(val)
                    if val_f != 0.0:
                        print(f"  {key:30s} = {val_f:.6f}")
                except (ValueError, TypeError):
                    if val and val != "0" and val != "0.0":
                        print(f"  {key:30s} = {val}")

    print(f"\nFinal result: acc={result.get('eval_accuracy', 0)}")


if __name__ == "__main__":
    main()
