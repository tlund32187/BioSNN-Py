"""Sweep: weight_decay × force_steps × seeds.

Root cause: both outputs saturate at max fire rate (~5 spikes/trial) making
WTA decisions random.  Weight decay should keep weights sub-saturation.
Fewer action-force steps reduce per-trial weight growth.
Multiple seeds probe topology sensitivity.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def make_spec(*, lr: float, wd: float, rds: int, fs: int, sf: float) -> dict[str, Any]:
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
        },
        "learning": {
            "enabled": True,
            "rule": "three_factor_elig_stdp",
            "lr": lr,
            "w_min": 0.0,
            "w_max": 1.0,
            "a_plus": 1.0,
            "a_minus": 0.0,
            "tau_e": 0.001,
            "modulator_threshold": 0.5,
            "weight_decay": wd,
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


def main() -> None:
    device = "cuda"
    n_trials = 400

    configs: list[tuple[str, int, float, float, int, int, float]] = []
    # (label, seed, lr, wd, rds, force_steps, sf)

    # Dimension 1: weight_decay with baseline params
    for wd in [0.0, 0.001, 0.005, 0.01, 0.05]:
        configs.append((f"wd{wd}", 42, 0.01, wd, 10, 10, -3.0))

    # Dimension 2: fewer force steps with moderate weight decay
    for fs in [1, 3, 5]:
        configs.append((f"fs{fs}_wd001", 42, 0.01, 0.001, fs, fs, -3.0))

    # Dimension 3: multiple seeds with best-bet config
    for seed in [123, 456, 789, 1337]:
        configs.append((f"s{seed}_wd001", seed, 0.01, 0.001, 10, 10, -3.0))

    print(f"Running {len(configs)} experiments, {n_trials} trials each")
    print("=" * 80)

    results: list[tuple[str, float, float]] = []
    for label, seed, lr, wd, rds, fs, sf in configs:
        spec = make_spec(lr=lr, wd=wd, rds=rds, fs=fs, sf=sf)
        out = Path(tempfile.mkdtemp(prefix=f"biosnn_v4_{label}_"))
        cfg = LogicGateRunConfig(
            gate=LogicGate.OR,
            seed=seed,
            steps=n_trials,
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
        t0 = time.perf_counter()
        result = run_logic_gate_engine(config=cfg, run_spec=spec)
        elapsed = time.perf_counter() - t0
        acc = float(result.get("eval_accuracy", 0.0))
        sa = float(result.get("sample_accuracy", 0.0))
        results.append((label, acc, sa))
        tag = "***" if acc > 0.75 else "== " if acc == 0.75 else "   "
        print(f"{tag} {label:20s}  eval={acc:.4f}  sample={sa:.4f}  ({elapsed:.1f}s)")

    print("\n" + "=" * 80)
    print("SORTED BY EVAL ACCURACY:")
    for label, acc, sa in sorted(results, key=lambda x: -x[1]):
        tag = "***" if acc > 0.75 else "== " if acc == 0.75 else "   "
        print(f"{tag} {label:20s}  eval={acc:.4f}  sample={sa:.4f}")


if __name__ == "__main__":
    main()
