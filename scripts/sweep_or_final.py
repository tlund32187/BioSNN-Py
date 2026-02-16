"""Final OR gate sweep: modulator_threshold + full action force + tuning."""

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
        "lr": 0.01,
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
            "steps": 10,  # Match rds
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

# Each experiment: (label, spec_overrides, cfg_overrides)
EXPERIMENTS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    # --- Baseline (the 0.75 config that works) ---
    ("baseline seed=42", {}, {}),
    ("baseline seed=7", {}, {"seed": 7}),
    ("baseline seed=123", {}, {"seed": 123}),
    # --- Learning rate ---
    ("lr=0.005", {"learning": {"lr": 0.005}}, {}),
    ("lr=0.02", {"learning": {"lr": 0.02}}, {}),
    ("lr=0.05", {"learning": {"lr": 0.05}}, {}),
    # --- a_minus (depression strength) ---
    ("a-=1.5", {"learning": {"a_minus": 1.5}}, {}),
    ("a-=2.0", {"learning": {"a_minus": 2.0}}, {}),
    ("a-=0.6", {"learning": {"a_minus": 0.6}}, {}),
    # --- Longer reward window ---
    ("rds=15", {"logic": {"action_force": {"steps": 15}}}, {"reward_delivery_steps": 15}),
    ("rds=20", {"logic": {"action_force": {"steps": 20}}}, {"reward_delivery_steps": 20}),
    # --- tau_e variations ---
    ("tau_e=0.003", {"learning": {"tau_e": 0.003}}, {}),
    ("tau_e=0.01", {"learning": {"tau_e": 0.01}}, {}),
    # --- Combined: longer reward + higher a_minus ---
    (
        "rds=15 a-=2.0",
        {"learning": {"a_minus": 2.0}, "logic": {"action_force": {"steps": 15}}},
        {"reward_delivery_steps": 15},
    ),
    (
        "rds=15 a-=1.5 lr=0.005",
        {"learning": {"a_minus": 1.5, "lr": 0.005}, "logic": {"action_force": {"steps": 15}}},
        {"reward_delivery_steps": 15},
    ),
    # --- DA decay_tau ---
    ("tau_DA=0.005", {"modulators": {"decay_tau": 0.005}}, {}),
    ("tau_DA=0.02", {"modulators": {"decay_tau": 0.02}}, {}),
    # --- Action force amplitude ---
    ("amp=2.0", {"logic": {"action_force": {"amplitude": 2.0}}}, {}),
    ("amp=0.5", {"logic": {"action_force": {"amplitude": 0.5}}}, {}),
    # --- Weight decay to prevent saturation ---
    ("wd=0.001", {"learning": {"weight_decay": 0.001}}, {}),
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
    default_rds = 10
    print(f"=== OR gate final sweep: {len(EXPERIMENTS)} experiments x {n_trials} trials ===\n")

    for i, (label, spec_ov, cfg_kw) in enumerate(EXPERIMENTS, 1):
        spec = _deep_merge(BASE_SPEC, spec_ov)
        rds = cfg_kw.get("reward_delivery_steps", default_rds)
        out = Path(tempfile.mkdtemp(prefix=f"biosnn_orfin_{i:02d}_"))

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
