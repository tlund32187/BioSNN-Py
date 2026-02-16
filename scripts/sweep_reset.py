"""Sweep: inter_trial_reset + global_weight_scale.

With inter-trial reset, NMDA/adaptation carry-over is eliminated.
Output spike counts should now depend on actual synaptic weights.
"""

from __future__ import annotations

import csv as csvmod
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def make_spec(
    *, gws: float, lr: float, w_max: float, rds: int, fs: int, a_minus: float = 0.0
) -> dict[str, Any]:
    return {
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
            "a_minus": a_minus,
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
                "steps": fs,
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


def analyze_csv(csv_path: Path, last_n: int = 20) -> dict[str, float]:
    with open(csv_path) as f:
        rows = list(csvmod.DictReader(f))
    tail = rows[-last_n:]
    tie_count = sum(1 for r in tail if float(r["out_spikes_0"]) == float(r["out_spikes_1"]))
    avg_s0 = sum(float(r["out_spikes_0"]) for r in tail) / len(tail)
    avg_s1 = sum(float(r["out_spikes_1"]) for r in tail) / len(tail)
    return {"tie_pct": tie_count / len(tail), "avg_s0": avg_s0, "avg_s1": avg_s1}


def main() -> None:
    device = "cuda"
    n_trials = 400

    # (label, gws, lr, w_max, rds, fs, a_minus)
    configs: list[tuple[str, float, float, float, int, int, float]] = []

    # Phase 1: Find the right gws range with inter-trial reset
    for gws_exp in range(-4, -12, -1):
        gws = 10.0**gws_exp
        lr = 1e-3  # keep lr at default, let gws control weight magnitudes
        w_max = 1.0  # also default
        configs.append((f"gws_{gws_exp}", gws, lr, w_max, 10, 10, 0.0))

    print(f"Running {len(configs)} experiments, {n_trials} trials each, inter_trial_reset=True")
    print("=" * 100)

    results: list[tuple[str, float, float, float, float, float, float]] = []
    for label, gws_val, lr, w_max, rds, fs, a_minus in configs:
        spec = make_spec(gws=gws_val, lr=lr, w_max=w_max, rds=rds, fs=fs, a_minus=a_minus)
        out = Path(tempfile.mkdtemp(prefix=f"biosnn_reset_{label}_"))
        cfg = LogicGateRunConfig(
            gate=LogicGate.OR,
            seed=42,
            steps=n_trials,
            sim_steps_per_trial=15,
            device=device,
            learning_mode="rstdp",
            engine_learning_rule="three_factor_elig_stdp",
            debug=True,
            out_dir=out,
            artifacts_root=out,
            reward_delivery_steps=rds,
            reward_delivery_clamp_input=True,
            inter_trial_reset=True,
        )
        t0 = time.perf_counter()
        result = run_logic_gate_engine(config=cfg, run_spec=spec)
        elapsed = time.perf_counter() - t0
        acc = float(result.get("eval_accuracy", 0.0))
        sa = float(result.get("sample_accuracy", 0.0))

        csv_stats = analyze_csv(out / "trials.csv")
        tie_pct = csv_stats["tie_pct"]
        avg_s0 = csv_stats["avg_s0"]
        avg_s1 = csv_stats["avg_s1"]

        results.append((label, acc, sa, tie_pct, avg_s0, avg_s1, elapsed))
        tag = "***" if acc > 0.75 else "== " if acc == 0.75 else "   "
        print(
            f"{tag} {label:15s}  eval={acc:.4f}  sample={sa:.4f}  "
            f"ties={tie_pct:.0%}  spk_avg=[{avg_s0:.1f},{avg_s1:.1f}]  ({elapsed:.1f}s)"
        )

    print("\n" + "=" * 100)
    print("SORTED BY EVAL ACCURACY:")
    for label, acc, sa, tie_pct, avg_s0, avg_s1, elapsed in sorted(
        results, key=lambda x: (-x[1], x[3])
    ):
        tag = "***" if acc > 0.75 else "== " if acc == 0.75 else "   "
        print(
            f"{tag} {label:15s}  eval={acc:.4f}  sample={sa:.4f}  "
            f"ties={tie_pct:.0%}  spk_avg=[{avg_s0:.1f},{avg_s1:.1f}]"
        )


if __name__ == "__main__":
    main()
