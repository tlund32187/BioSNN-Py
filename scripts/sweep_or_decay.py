"""Targeted OR gate experiments: decay_tau, w_max, weight_decay."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from biosnn.tasks.logic_gates import (
    LogicGate,
    LogicGateRunConfig,
    run_logic_gate_engine,
)

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


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


EXPERIMENTS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    # --- DA decay_tau variations (prevent inter-trial leakage) ---
    ("tau=0.003", {"modulators": {"decay_tau": 0.003}}, {}),
    ("tau=0.005", {"modulators": {"decay_tau": 0.005}}, {}),
    # --- w_max (prevent saturation) ---
    ("wmax=0.1 tau=0.003", {"learning": {"w_max": 0.1}, "modulators": {"decay_tau": 0.003}}, {}),
    ("wmax=0.2 tau=0.003", {"learning": {"w_max": 0.2}, "modulators": {"decay_tau": 0.003}}, {}),
    ("wmax=0.05 tau=0.003", {"learning": {"w_max": 0.05}, "modulators": {"decay_tau": 0.003}}, {}),
    # --- weight_decay (prevent saturation) ---
    (
        "wd=0.001 tau=0.003",
        {"learning": {"weight_decay": 0.001}, "modulators": {"decay_tau": 0.003}},
        {},
    ),
    (
        "wd=0.01 tau=0.003",
        {"learning": {"weight_decay": 0.01}, "modulators": {"decay_tau": 0.003}},
        {},
    ),
    # --- Higher reward steps ---
    ("tau=0.003 rds=10", {"modulators": {"decay_tau": 0.003}}, {"reward_delivery_steps": 10}),
    ("tau=0.003 rds=15", {"modulators": {"decay_tau": 0.003}}, {"reward_delivery_steps": 15}),
    # --- Higher DA amount with short tau ---
    ("tau=0.003 da=3.0", {"modulators": {"decay_tau": 0.003, "amount": 3.0}}, {}),
    ("tau=0.003 da=5.0", {"modulators": {"decay_tau": 0.003, "amount": 5.0}}, {}),
    # --- Combos: best candidates ---
    (
        "wmax=0.1 tau=0.003 rds=10",
        {"learning": {"w_max": 0.1}, "modulators": {"decay_tau": 0.003}},
        {"reward_delivery_steps": 10},
    ),
    (
        "wmax=0.1 tau=0.003 da=3",
        {"learning": {"w_max": 0.1}, "modulators": {"decay_tau": 0.003, "amount": 3.0}},
        {},
    ),
    (
        "wmax=0.2 tau=0.003 rds=10",
        {"learning": {"w_max": 0.2}, "modulators": {"decay_tau": 0.003}},
        {"reward_delivery_steps": 10},
    ),
    (
        "wmax=0.2 wd=0.001 tau=0.003",
        {"learning": {"w_max": 0.2, "weight_decay": 0.001}, "modulators": {"decay_tau": 0.003}},
        {},
    ),
    # --- a_minus variations ---
    ("aminus=3 tau=0.003", {"learning": {"a_minus": 3.0}, "modulators": {"decay_tau": 0.003}}, {}),
    ("aminus=5 tau=0.003", {"learning": {"a_minus": 5.0}, "modulators": {"decay_tau": 0.003}}, {}),
    # --- Higher lr with protections ---
    (
        "lr=0.05 wmax=0.2 tau=0.003",
        {"learning": {"lr": 0.05, "w_max": 0.2}, "modulators": {"decay_tau": 0.003}},
        {},
    ),
    (
        "lr=0.02 wmax=0.2 tau=0.003",
        {"learning": {"lr": 0.02, "w_max": 0.2}, "modulators": {"decay_tau": 0.003}},
        {},
    ),
]


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="biosnn_or_decay_"))
    print(f"Output: {tmp_root}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print()

    results: list[dict[str, Any]] = []
    for i, (label, spec_patch, cfg_kw) in enumerate(EXPERIMENTS, 1):
        safe_label = label.replace("=", "_").replace(" ", "_").replace("/", "_").replace(">", "_")
        run_dir = tmp_root / safe_label
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = _deep_merge(BASE_SPEC, spec_patch)
        cfg = LogicGateRunConfig(
            gate=LogicGate.OR,
            seed=42,
            steps=400,
            sim_steps_per_trial=15,
            device="cuda",
            learning_mode="rstdp",
            engine_learning_rule="three_factor_elig_stdp",
            debug=False,
            out_dir=run_dir,
            artifacts_root=run_dir,
            reward_delivery_steps=cfg_kw.get("reward_delivery_steps", 5),
        )
        t0 = time.perf_counter()
        try:
            result = run_logic_gate_engine(cfg, spec)
            elapsed = time.perf_counter() - t0
            ea = result["eval_accuracy"]
            pa = result["passed"]
            fp = result.get("first_pass_trial")
            dw = result["mean_abs_dw"]
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            ea = -1.0
            pa = False
            fp = None
            dw = -1.0
            print(f"  ERROR: {exc!s:.100}")

        results.append(
            {
                "label": label,
                "eval_accuracy": ea,
                "passed": pa,
                "first_pass_trial": fp,
                "mean_abs_dw": dw,
                "elapsed_s": round(elapsed, 1),
            }
        )
        fp_str = str(fp) if fp else "-"
        dw_str = f"{dw:.2e}" if isinstance(dw, float) and dw >= 0 else "-"
        print(
            f"[{i}/{len(EXPERIMENTS)}] {label:<40} acc={ea:.2f} pass={pa!s:<5} 1st={fp_str:<6} dw={dw_str}  ({elapsed:.1f}s)"
        )

    # Summary
    print(f"\n{'=' * 90}")
    sorted_r = sorted(results, key=lambda r: (-int(r["passed"]), -r["eval_accuracy"]))
    print(f"{'Label':<45} {'Acc':>5} {'Pass':>5} {'1stP':>6} {'dW':>10}")
    print("-" * 90)
    for r in sorted_r:
        fp_str = str(r["first_pass_trial"]) if r["first_pass_trial"] else "-"
        dw_str = (
            f"{r['mean_abs_dw']:.2e}"
            if isinstance(r["mean_abs_dw"], float) and r["mean_abs_dw"] >= 0
            else "-"
        )
        print(
            f"{r['label']:<45} {r['eval_accuracy']:>5.2f} {'YES' if r['passed'] else 'no':>5} {fp_str:>6} {dw_str:>10}"
        )


if __name__ == "__main__":
    main()
