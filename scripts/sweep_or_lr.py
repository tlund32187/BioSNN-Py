"""Quick sweep: learning rate + suppression factor tuning."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine

BASE_SPEC: dict[str, Any] = {
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
        "lr": 0.001,
        "w_min": 0.0,
        "w_max": 1.0,
        "a_plus": 1.0,
        "a_minus": 1.0,
        "tau_e": 0.005,
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

EXPERIMENTS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    # Baseline: lr=0.001 suppression=-3.0
    ("lr=0.001 sf=-3.0", {}, {}),
    ("lr=0.001 sf=-3.0 s=7", {}, {"seed": 7}),
    ("lr=0.001 sf=-3.0 s=123", {}, {"seed": 123}),
    # Lower lr
    ("lr=0.0005 sf=-3.0", {"learning": {"lr": 0.0005}}, {}),
    ("lr=0.0003 sf=-3.0", {"learning": {"lr": 0.0003}}, {}),
    # Higher a_minus
    ("lr=0.001 a-=2.0 sf=-3.0", {"learning": {"a_minus": 2.0}}, {}),
    ("lr=0.001 a-=3.0 sf=-3.0", {"learning": {"a_minus": 3.0}}, {}),
    # Even stronger suppression
    ("lr=0.001 sf=-5.0", {"logic": {"action_force": {"suppression_factor": -5.0}}}, {}),
    ("lr=0.001 sf=-10.0", {"logic": {"action_force": {"suppression_factor": -10.0}}}, {}),
    # Longer reward window
    (
        "lr=0.001 sf=-3.0 rds=15",
        {"logic": {"action_force": {"steps": 15}}},
        {"reward_delivery_steps": 15},
    ),
    # Combination: low lr + strong suppression + high a_minus
    (
        "lr=0.0005 a-=2.0 sf=-5.0",
        {
            "learning": {"lr": 0.0005, "a_minus": 2.0},
            "logic": {"action_force": {"suppression_factor": -5.0}},
        },
        {},
    ),
    # Weight decay
    ("lr=0.001 sf=-3.0 wd=0.0005", {"learning": {"weight_decay": 0.0005}}, {}),
]


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def main() -> None:
    n_trials = 800
    device = "cuda"
    print(f"=== LR+suppression sweep: {len(EXPERIMENTS)} x {n_trials} trials ===\n")

    for i, (label, spec_ov, cfg_kw) in enumerate(EXPERIMENTS, 1):
        spec = _deep_merge(BASE_SPEC, spec_ov)
        rds = cfg_kw.get("reward_delivery_steps", 10)
        out = Path(tempfile.mkdtemp(prefix=f"biosnn_orlr_{i:02d}_"))

        cfg = LogicGateRunConfig(
            gate=LogicGate.OR,
            seed=cfg_kw.get("seed", 42),
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

        try:
            t0 = time.perf_counter()
            result = run_logic_gate_engine(cfg, spec)
            elapsed = time.perf_counter() - t0
            acc = result.get("eval_accuracy", -1)
            passed = result.get("passed", False)
            fp = result.get("first_pass_trial", -1)
            dw = result.get("mean_abs_dw", 0.0)
            fp_str = str(fp) if fp and fp > 0 else "-"
            print(
                f"[{i:2d}/{len(EXPERIMENTS)}] {label:40s}  "
                f"acc={acc:.2f} pass={str(passed):5s} 1st={fp_str:6s} "
                f"dw={dw:.2e}  ({elapsed:.1f}s)"
            )
        except Exception as exc:
            print(f"[{i:2d}/{len(EXPERIMENTS)}] {label:40s}  ERROR: {exc}")


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    main()
