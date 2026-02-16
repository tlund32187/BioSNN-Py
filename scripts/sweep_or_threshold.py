"""OR gate sweep: modulator_threshold + tau_e + unclamped reward."""

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

# Each experiment: (label, spec_overrides, cfg_overrides)
EXPERIMENTS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    # --- Baseline with threshold only (clamped input) ---
    ("thr=0.5 clamp", {}, {}),
    # --- Unclamped input during reward (key differentiator) ---
    ("thr=0.5 unclamp", {}, {"reward_delivery_clamp_input": False}),
    # --- Vary tau_e (eligibility decay) ---
    (
        "thr=0.5 unclamp tau_e=0.003",
        {"learning": {"tau_e": 0.003}},
        {"reward_delivery_clamp_input": False},
    ),
    (
        "thr=0.5 unclamp tau_e=0.01",
        {"learning": {"tau_e": 0.01}},
        {"reward_delivery_clamp_input": False},
    ),
    (
        "thr=0.5 unclamp tau_e=0.05",
        {"learning": {"tau_e": 0.05}},
        {"reward_delivery_clamp_input": False},
    ),
    # --- Vary a_minus (depression strength) ---
    (
        "thr=0.5 unclamp a-=1.5",
        {"learning": {"a_minus": 1.5}},
        {"reward_delivery_clamp_input": False},
    ),
    (
        "thr=0.5 unclamp a-=2.0",
        {"learning": {"a_minus": 2.0}},
        {"reward_delivery_clamp_input": False},
    ),
    (
        "thr=0.5 unclamp a-=3.0",
        {"learning": {"a_minus": 3.0}},
        {"reward_delivery_clamp_input": False},
    ),
    # --- Longer reward window ---
    (
        "thr=0.5 unclamp rds=10",
        {},
        {"reward_delivery_clamp_input": False, "reward_delivery_steps": 10},
    ),
    (
        "thr=0.5 unclamp rds=15",
        {},
        {"reward_delivery_clamp_input": False, "reward_delivery_steps": 15},
    ),
    # --- Combined: long reward + high a_minus ---
    (
        "thr=0.5 unclamp rds=10 a-=2.0",
        {"learning": {"a_minus": 2.0}},
        {"reward_delivery_clamp_input": False, "reward_delivery_steps": 10},
    ),
    (
        "thr=0.5 unclamp rds=15 a-=2.0",
        {"learning": {"a_minus": 2.0}},
        {"reward_delivery_clamp_input": False, "reward_delivery_steps": 15},
    ),
    # --- Higher DA decay_tau (so DA stays above threshold longer in reward) ---
    (
        "thr=0.5 unclamp tau=0.02",
        {"modulators": {"decay_tau": 0.02}},
        {"reward_delivery_clamp_input": False},
    ),
    (
        "thr=0.5 unclamp tau=0.005",
        {"modulators": {"decay_tau": 0.005}},
        {"reward_delivery_clamp_input": False},
    ),
    # --- Lower threshold (more permissive) ---
    (
        "thr=0.2 unclamp",
        {"learning": {"modulator_threshold": 0.2}},
        {"reward_delivery_clamp_input": False},
    ),
    # --- Higher lr ---
    ("thr=0.5 unclamp lr=0.05", {"learning": {"lr": 0.05}}, {"reward_delivery_clamp_input": False}),
    (
        "thr=0.5 unclamp lr=0.001",
        {"learning": {"lr": 0.001}},
        {"reward_delivery_clamp_input": False},
    ),
    # --- No threshold (control) ---
    (
        "no_thr unclamp tau=0.01",
        {"learning": {"modulator_threshold": 0.0}},
        {"reward_delivery_clamp_input": False},
    ),
    # --- Short tau_e + high a_minus + long reward (best combo?) ---
    (
        "tau_e=0.003 a-=2.0 rds=15 unclamp",
        {"learning": {"tau_e": 0.003, "a_minus": 2.0}},
        {"reward_delivery_clamp_input": False, "reward_delivery_steps": 15},
    ),
    # --- Weight decay to prevent saturation ---
    (
        "thr=0.5 unclamp wd=0.001",
        {"learning": {"weight_decay": 0.001}},
        {"reward_delivery_clamp_input": False},
    ),
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
    n_trials = 400
    device = "cuda"
    print(f"=== OR gate threshold sweep: {len(EXPERIMENTS)} experiments × {n_trials} trials ===\n")

    for i, (label, spec_ov, cfg_kw) in enumerate(EXPERIMENTS, 1):
        spec = _deep_merge(BASE_SPEC, spec_ov)
        out = Path(tempfile.mkdtemp(prefix=f"biosnn_orthr_{i:02d}_"))

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
            reward_delivery_steps=cfg_kw.get("reward_delivery_steps", 5),
            reward_delivery_clamp_input=cfg_kw.get("reward_delivery_clamp_input", True),
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
                f"[{i:2d}/{len(EXPERIMENTS)}] {label:45s}  "
                f"acc={acc:.2f} pass={str(passed):5s} 1st={fp_str:6s} "
                f"dw={dw:.2e}  ({elapsed:.1f}s)"
            )
        except Exception as exc:
            print(f"[{i:2d}/{len(EXPERIMENTS)}] {label:45s}  ERROR: {exc}")


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    main()
