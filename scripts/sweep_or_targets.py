"""Quick experiments varying learning targets, lr, and reward settings."""

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
    # --- Learning targets ---
    ("tgt=HiddenExc->Out", {"learning": {"targets": ["HiddenExcit->Out"]}}, {}),
    ("tgt=OutSkip", {"learning": {"targets": ["In->OutSkip"]}}, {}),
    ("tgt=Out+Skip", {"learning": {"targets": ["HiddenExcit->Out", "In->OutSkip"]}}, {}),
    ("tgt=all(default)", {}, {}),
    # --- Learning targets + higher lr ---
    ("tgt=HiddenExc lr=0.05", {"learning": {"targets": ["HiddenExcit->Out"], "lr": 0.05}}, {}),
    ("tgt=OutSkip lr=0.05", {"learning": {"targets": ["In->OutSkip"], "lr": 0.05}}, {}),
    (
        "tgt=Out+Skip lr=0.05",
        {"learning": {"targets": ["HiddenExcit->Out", "In->OutSkip"], "lr": 0.05}},
        {},
    ),
    # --- Reward unclamped ---
    ("unclamp rds=5", {}, {"reward_delivery_clamp_input": False, "reward_delivery_steps": 5}),
    ("unclamp rds=10", {}, {"reward_delivery_clamp_input": False, "reward_delivery_steps": 10}),
    # --- Combo: targets + unclamped ---
    (
        "tgt=HiddenExc unclamp rds=5",
        {"learning": {"targets": ["HiddenExcit->Out"]}},
        {"reward_delivery_clamp_input": False, "reward_delivery_steps": 5},
    ),
    (
        "tgt=Out+Skip unclamp rds=5",
        {"learning": {"targets": ["HiddenExcit->Out", "In->OutSkip"]}},
        {"reward_delivery_clamp_input": False, "reward_delivery_steps": 5},
    ),
    # --- DA decay_tau variations ---
    ("tau=0.01", {"modulators": {"decay_tau": 0.01}}, {}),
    ("tau=0.1", {"modulators": {"decay_tau": 0.1}}, {}),
    # --- High rds ---
    ("rds=10 clamp", {}, {"reward_delivery_steps": 10}),
    ("rds=15 clamp", {}, {"reward_delivery_steps": 15}),
    # --- Action force only on explore ---
    ("af=explore_or_silent", {"logic": {"action_force": {"mode": "explore_or_silent"}}}, {}),
]


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="biosnn_targets_"))
    print(f"Output: {tmp_root}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print()

    results: list[dict[str, Any]] = []
    for i, (label, spec_patch, cfg_kw) in enumerate(EXPERIMENTS, 1):
        run_dir = tmp_root / label.replace("=", "_").replace("+", "_").replace(" ", "_").replace(
            "/", "_"
        ).replace(">", "_").replace("<", "_").replace("(", "").replace(")", "")
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
            reward_delivery_clamp_input=cfg_kw.get("reward_delivery_clamp_input", True),
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
