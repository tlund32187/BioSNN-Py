#!/usr/bin/env python
"""Quick OR-gate parameter search — focused probes after bug fixes.

Now that learning actually works (dense eligibility + weight clamping),
run targeted experiments to find the convergence sweet spot.
"""

from __future__ import annotations

import copy
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from biosnn.tasks.logic_gates import (  # noqa: E402
    LogicGate,
    LogicGateRunConfig,
    run_logic_gate_engine,
)

DEVICE = "cuda"
SEED = 42

# ── Base spec: all required subsystems enabled ───────────────────────
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
        "synaptic_scaling": True,
    },
    "modulators": {
        "enabled": True,
        "kinds": ["dopamine"],
        "amount": 1.0,
        "field_type": "global_scalar",
        "decay_tau": 1.0,
    },
    "wrapper": {
        "enabled": False,
        "ach_lr_gain": 0.0,
        "ne_lr_gain": 0.0,
    },
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


def run_one(
    label: str,
    spec_patch: dict[str, Any],
    steps: int,
    tmp_root: Path,
    *,
    seed: int = SEED,
    sim_steps: int = 15,
    reward_delivery_steps: int = 2,
) -> dict[str, Any]:
    run_dir = tmp_root / label.replace("=", "_").replace("+", "_").replace("/", "_").replace(
        " ", "_"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = _deep_merge(BASE_SPEC, spec_patch)
    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=seed,
        steps=steps,
        sim_steps_per_trial=sim_steps,
        device=DEVICE,
        learning_mode="rstdp",
        engine_learning_rule="three_factor_elig_stdp",
        debug=False,
        out_dir=run_dir,
        artifacts_root=run_dir,
        reward_delivery_steps=reward_delivery_steps,
    )
    t0 = time.perf_counter()
    try:
        result = run_logic_gate_engine(cfg, spec)
        elapsed = time.perf_counter() - t0
        return {
            "label": label,
            "eval_accuracy": result["eval_accuracy"],
            "sample_accuracy": result["sample_accuracy"],
            "passed": result["passed"],
            "first_pass_trial": result.get("first_pass_trial"),
            "trial_acc_rolling": result["trial_acc_rolling"],
            "mean_abs_dw": result["mean_abs_dw"],
            "elapsed_s": round(elapsed, 1),
            "error": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        traceback.print_exc()
        return {
            "label": label,
            "eval_accuracy": -1,
            "sample_accuracy": -1,
            "passed": False,
            "first_pass_trial": -1,
            "trial_acc_rolling": -1,
            "mean_abs_dw": -1,
            "elapsed_s": round(elapsed, 1),
            "error": str(exc)[:200],
        }


def print_results(results: list[dict[str, Any]], header: str) -> None:
    print(f"\n{'=' * 90}")
    print(f"  {header}")
    print(f"{'=' * 90}")
    results_sorted = sorted(results, key=lambda x: (-int(x["passed"]), -x["eval_accuracy"]))
    print(f"{'Label':<48} {'Acc':>5} {'Pass':>5} {'1stP':>6} {'dW':>10} {'Time':>6}")
    print("-" * 90)
    for r in results_sorted:
        fp = (
            str(r["first_pass_trial"])
            if r["first_pass_trial"] and r["first_pass_trial"] > 0
            else "-"
        )
        dw = (
            f"{r['mean_abs_dw']:.2e}"
            if isinstance(r["mean_abs_dw"], float) and r["mean_abs_dw"] >= 0
            else "-"
        )
        err_tag = " ERR" if r["error"] else ""
        print(
            f"{r['label']:<48} {r['eval_accuracy']:>5.2f} "
            f"{'YES' if r['passed'] else 'no':>5} {fp:>6} "
            f"{dw:>10} {r['elapsed_s']:>5.1f}s{err_tag}"
        )


# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENTS                                                      ║
# ╚════════════════════════════════════════════════════════════════════╝

EXPERIMENTS: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
# Format: (label, spec_patch, cfg_overrides)

# ── A. Learning rate sweep at 800 trials ─────────────────────────────
for lr in [0.002, 0.005, 0.008, 0.01, 0.015, 0.02]:
    EXPERIMENTS.append((f"lr={lr}", {"learning": {"lr": lr}}, {}))

# ── B. Dopamine amount (reward signal strength) ─────────────────────
for da in [0.5, 1.0, 2.0, 4.0]:
    EXPERIMENTS.append((f"da={da}", {"modulators": {"amount": da}}, {}))

# ── C. Reward delivery steps (more time for DA to act) ──────────────
for rds in [1, 2, 3, 5]:
    EXPERIMENTS.append((f"rds={rds}", {}, {"reward_delivery_steps": rds}))

# ── D. Synaptic scaling on/off ──────────────────────────────────────
EXPERIMENTS.append(("scaling=off", {"learning": {"synaptic_scaling": False}}, {}))

# ── E. Action force mode/amplitude combos ───────────────────────────
EXPERIMENTS.append(
    (
        "af=explore_only",
        {"logic": {"action_force": {"mode": "explore_or_silent"}}},
        {},
    )
)
EXPERIMENTS.append(
    (
        "af=silent_only",
        {"logic": {"action_force": {"mode": "silent_only"}}},
        {},
    )
)
EXPERIMENTS.append(
    (
        "af_amp=2.0",
        {"logic": {"action_force": {"amplitude": 2.0}}},
        {},
    )
)

# ── F. Exploration settings ─────────────────────────────────────────
EXPERIMENTS.append(
    (
        "eps=0",
        {"logic": {"exploration": {"enabled": False, "epsilon_start": 0.0}}},
        {},
    )
)
EXPERIMENTS.append(
    (
        "eps=0.10",
        {"logic": {"exploration": {"epsilon_start": 0.10}}},
        {},
    )
)

# ── G. Combined: best from sanity test + tweaks ─────────────────────
EXPERIMENTS.append(
    (
        "combo_lr005_rds3",
        {"learning": {"lr": 0.005}},
        {"reward_delivery_steps": 3},
    )
)
EXPERIMENTS.append(
    (
        "combo_lr01_da2_rds3",
        {"learning": {"lr": 0.01}, "modulators": {"amount": 2.0}},
        {"reward_delivery_steps": 3},
    )
)
EXPERIMENTS.append(
    (
        "combo_lr01_noscale",
        {"learning": {"lr": 0.01, "synaptic_scaling": False}},
        {},
    )
)
EXPERIMENTS.append(
    (
        "combo_lr01_noscale_rds3",
        {"learning": {"lr": 0.01, "synaptic_scaling": False}},
        {"reward_delivery_steps": 3},
    )
)
EXPERIMENTS.append(
    (
        "combo_lr01_da2_noscale",
        {"learning": {"lr": 0.01, "synaptic_scaling": False}, "modulators": {"amount": 2.0}},
        {},
    )
)
EXPERIMENTS.append(
    (
        "combo_lr005_noscale",
        {"learning": {"lr": 0.005, "synaptic_scaling": False}},
        {},
    )
)


def main() -> None:
    import tempfile

    tmp_root = Path(tempfile.mkdtemp(prefix="biosnn_sweep_or_quick_"))
    print(f"Output root: {tmp_root}")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Each run: 800 trials × 15 sim steps on {DEVICE}")
    print()

    results: list[dict[str, Any]] = []
    for i, (label, spec_patch, cfg_kw) in enumerate(EXPERIMENTS, 1):
        rds = cfg_kw.get("reward_delivery_steps", 2)
        r = run_one(
            label,
            spec_patch,
            steps=800,
            tmp_root=tmp_root,
            reward_delivery_steps=rds,
        )
        results.append(r)
        fp = r.get("first_pass_trial") or "-"
        dw = (
            f"{r['mean_abs_dw']:.2e}"
            if isinstance(r["mean_abs_dw"], float) and r["mean_abs_dw"] >= 0
            else "-"
        )
        print(
            f"[{i}/{len(EXPERIMENTS)}] {label:<40} acc={r['eval_accuracy']:.2f} "
            f"pass={r['passed']}  1st={fp!s:<6} dw={dw}  ({r['elapsed_s']:.1f}s)"
        )

    print_results(results, "FINAL RESULTS — OR gate 800-trial runs")

    # Print best config
    passed = [r for r in results if r["passed"]]
    if passed:
        best = min(passed, key=lambda r: r.get("first_pass_trial") or 9999)
        print(f"\n*** BEST: {best['label']} — passed at trial {best['first_pass_trial']}")
    else:
        # Show top 3 by accuracy
        top3 = sorted(results, key=lambda r: -r["eval_accuracy"])[:3]
        print("\nNo runs passed. Top 3 by eval_accuracy:")
        for r in top3:
            print(f"  {r['label']}: eval_acc={r['eval_accuracy']:.2f}")


if __name__ == "__main__":
    main()
