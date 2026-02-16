"""Targeted experiment: gws=-8, scaled lr/w_max, inter-trial reset, 800 trials.

Focus on OR gate learning with calibrated biophysical parameters.
Try multiple seeds and a_minus values.
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
    *,
    gws: float,
    lr: float,
    w_max: float,
    rds: int,
    fs: int,
    a_minus: float = 0.0,
    action_amp: float = 1.0,
    suppress: float = -3.0,
    weight_decay: float = 0.0,
    tau_e: float = 0.001,
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
            "tau_e": tau_e,
            "modulator_threshold": 0.5,
            "weight_decay": weight_decay,
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
                "amplitude": action_amp,
                "window": "reward_window",
                "steps": fs,
                "compartment": "soma",
                "suppression_factor": suppress,
            },
            "gate_context": {
                "enabled": True,
                "amplitude": 0.30,
                "compartment": "dendrite",
                "targets": ["hidden"],
            },
        },
    }


def analyze_csv(csv_path: Path) -> dict[str, Any]:
    with open(csv_path) as f:
        rows = list(csvmod.DictReader(f))
    n = len(rows)
    tail = rows[-20:]
    tie_count = sum(1 for r in tail if float(r["out_spikes_0"]) == float(r["out_spikes_1"]))
    avg_s0 = sum(float(r["out_spikes_0"]) for r in tail) / len(tail)
    avg_s1 = sum(float(r["out_spikes_1"]) for r in tail) / len(tail)

    # Per-case accuracy in last 100 trials
    per_case: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for row in rows[-min(100, n) :]:
        ci = int(row["case_idx"])
        per_case[ci].append(int(row["correct"]))
    per_case_acc = {ci: sum(lst) / max(1, len(lst)) for ci, lst in per_case.items()}

    # Weight trajectory: sample at 25% and 75%
    w_25 = rows[n // 4] if n > 4 else rows[-1]
    w_75 = rows[3 * n // 4] if n > 4 else rows[-1]
    w_end = rows[-1]

    return {
        "tie_pct": tie_count / len(tail),
        "avg_s0": avg_s0,
        "avg_s1": avg_s1,
        "per_case_acc": per_case_acc,
        "w_25": float(w_end.get("weights_mean", 0)),
        "w_end_min": float(w_end.get("weights_min", 0)),
        "w_end_max": float(w_end.get("weights_max", 0)),
        "w_end_mean": float(w_end.get("weights_mean", 0)),
    }


def main() -> None:
    device = "cuda"
    n_trials = 800

    # (label, gws, lr, w_max, rds, fs, a_minus, action_amp, suppress, wd, tau_e, seed)
    configs: list[tuple[str, dict[str, Any], int]] = []

    gws = 1e-8
    w_max_base = gws * 3.0  # 3e-8

    # Vary a_minus and lr
    for a_minus in [0.0, 0.3, 0.6]:
        for lr_mult in [0.5, 1.0, 3.0]:
            lr = w_max_base * 0.01 * lr_mult
            label = f"am{a_minus}_lr{lr_mult}"
            spec = make_spec(gws=gws, lr=lr, w_max=w_max_base, rds=10, fs=10, a_minus=a_minus)
            configs.append((label, spec, 42))

    # Best from above with different seeds
    for seed in [123, 456, 789]:
        for a_minus in [0.0, 0.6]:
            lr = w_max_base * 0.01
            spec = make_spec(gws=gws, lr=lr, w_max=w_max_base, rds=10, fs=10, a_minus=a_minus)
            configs.append((f"am{a_minus}_s{seed}", spec, seed))

    # Try with weight decay to prevent weight saturation
    for wd_mult in [0.001, 0.01, 0.1]:
        lr = w_max_base * 0.01
        wd = lr * wd_mult
        spec = make_spec(
            gws=gws, lr=lr, w_max=w_max_base, rds=10, fs=10, a_minus=0.6, weight_decay=wd
        )
        configs.append((f"am0.6_wd{wd_mult}", spec, 42))

    # Try longer tau_e for more temporal credit assignment
    for te in [0.005, 0.010, 0.050]:
        lr = w_max_base * 0.01
        spec = make_spec(gws=gws, lr=lr, w_max=w_max_base, rds=10, fs=10, a_minus=0.6, tau_e=te)
        configs.append((f"am0.6_te{te}", spec, 42))

    print(f"Running {len(configs)} experiments, {n_trials} trials each")
    print("=" * 120)

    results = []
    for label, spec, seed in configs:
        out = Path(tempfile.mkdtemp(prefix=f"biosnn_tgt_{label}_"))
        cfg = LogicGateRunConfig(
            gate=LogicGate.OR,
            seed=seed,
            steps=n_trials,
            sim_steps_per_trial=15,
            device=device,
            learning_mode="rstdp",
            engine_learning_rule="three_factor_elig_stdp",
            debug=True,
            out_dir=out,
            artifacts_root=out,
            reward_delivery_steps=10,
            reward_delivery_clamp_input=True,
            inter_trial_reset=True,
        )
        t0 = time.perf_counter()
        result = run_logic_gate_engine(config=cfg, run_spec=spec)
        elapsed = time.perf_counter() - t0
        acc = float(result.get("eval_accuracy", 0.0))
        sa = float(result.get("sample_accuracy", 0.0))
        stats = analyze_csv(out / "trials.csv")

        results.append((label, acc, sa, stats, elapsed, seed))
        tag = "***" if acc > 0.75 else "== " if acc == 0.75 else "   "
        pc = stats["per_case_acc"]
        print(
            f"{tag} {label:20s} s={seed}  eval={acc:.4f}  sa={sa:.4f}  "
            f"spk=[{stats['avg_s0']:.1f},{stats['avg_s1']:.1f}]  "
            f"case=[{pc[0]:.2f},{pc[1]:.2f},{pc[2]:.2f},{pc[3]:.2f}]  "
            f"w=[{stats['w_end_min']:.2e},{stats['w_end_max']:.2e}]  "
            f"({elapsed:.0f}s)"
        )

    print("\n" + "=" * 120)
    print("TOP 10 BY EVAL ACCURACY:")
    for label, acc, sa, stats, elapsed, seed in sorted(results, key=lambda x: -x[1])[:10]:
        tag = "***" if acc > 0.75 else "== " if acc == 0.75 else "   "
        pc = stats["per_case_acc"]
        print(
            f"{tag} {label:20s} s={seed}  eval={acc:.4f}  sa={sa:.4f}  "
            f"case=[{pc[0]:.2f},{pc[1]:.2f},{pc[2]:.2f},{pc[3]:.2f}]  "
            f"w=[{stats['w_end_min']:.2e},{stats['w_end_max']:.2e}]"
        )


if __name__ == "__main__":
    main()
