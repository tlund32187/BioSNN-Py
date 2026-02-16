#!/usr/bin/env python
"""Focused OR-gate parameter search.

Two-phase strategy:
  Phase 1 — Quick 400-trial probes to rank parameter axes.
  Phase 2 — 800-trial validation of the best combination.

All runs use: engine backend, CUDA, learning, delay_steps,
exploration, action_force, gate_context, dopamine modulation,
wrapper, and homeostasis.
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
# This is the starting point — every run uses all required features.
BASE_SPEC: dict = {
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


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def make_cfg(steps: int, sim_steps: int = 15, out_dir: Path | None = None) -> LogicGateRunConfig:
    return LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=SEED,
        steps=steps,
        sim_steps_per_trial=sim_steps,
        device=DEVICE,
        learning_mode="rstdp",
        engine_learning_rule="three_factor_elig_stdp",
        debug=False,
        out_dir=out_dir,
        artifacts_root=out_dir,
    )


def run_one(label: str, spec_patch: dict, steps: int, tmp_root: Path) -> dict:
    run_dir = tmp_root / label.replace("=", "_").replace("+", "_").replace("/", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = _deep_merge(BASE_SPEC, spec_patch)
    cfg = make_cfg(steps=steps, out_dir=run_dir)
    t0 = time.perf_counter()
    try:
        result = run_logic_gate_engine(cfg, spec)
        elapsed = time.perf_counter() - t0
        return {
            "label": label,
            "eval_accuracy": result["eval_accuracy"],
            "sample_accuracy": result["sample_accuracy"],
            "passed": result["passed"],
            "first_pass_trial": result["first_pass_trial"],
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


def print_results(results: list[dict], header: str) -> None:
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
# ║  PHASE 1 — Quick 400-trial probes                                ║
# ╚════════════════════════════════════════════════════════════════════╝

PHASE1_STEPS = 400

PHASE1_EXPERIMENTS: list[tuple[str, dict]] = []

# ── A. Learning rate (most impactful for convergence speed) ──────────
for lr in [0.005, 0.01, 0.02, 0.05, 0.10]:
    PHASE1_EXPERIMENTS.append((f"lr={lr}", {"learning": {"lr": lr}}))

# ── B. Dopamine amount (reward signal strength) ─────────────────────
for da in [0.1, 0.5, 1.0, 2.0]:
    PHASE1_EXPERIMENTS.append((f"da_amt={da}", {"modulators": {"amount": da}}))

# ── C. Action force amplitude (output reinforcement) ────────────────
for amp in [0.35, 0.75, 1.0, 2.0, 4.0]:
    PHASE1_EXPERIMENTS.append((f"af_amp={amp}", {"logic": {"action_force": {"amplitude": amp}}}))

# ── D. Action force mode ────────────────────────────────────────────
for mode in ["always", "explore_or_silent", "silent_only"]:
    PHASE1_EXPERIMENTS.append((f"af_mode={mode}", {"logic": {"action_force": {"mode": mode}}}))

# ── E. Exploration epsilon_start ─────────────────────────────────────
for eps in [0.0, 0.10, 0.20, 0.40]:
    PHASE1_EXPERIMENTS.append(
        (
            f"eps_start={eps}",
            {
                "logic": {
                    "exploration": {
                        "enabled": eps > 0,
                        "epsilon_start": eps,
                        "epsilon_end": 0.01 if eps > 0 else 0.0,
                        "epsilon_decay_trials": 250,
                    }
                }
            },
        )
    )

# ── F. Sim steps per trial ───────────────────────────────────────────
# More steps = more spikes = stronger signal, but also more time
for ss in [10, 15, 20]:
    PHASE1_EXPERIMENTS.append((f"sim_steps={ss}", {}))
    # We override sim_steps in LogicGateRunConfig, not in spec
    # Handle this specially in the runner (see below)

# ── G. Delay steps ──────────────────────────────────────────────────
for ds in [1, 3, 5]:
    PHASE1_EXPERIMENTS.append((f"delay={ds}", {"delay_steps": ds}))

# ── H. Reward delivery steps ────────────────────────────────────────
for rds in [2, 5, 8]:
    PHASE1_EXPERIMENTS.append((f"reward_steps={rds}", {}))

# ── I. Dopamine decay_tau ────────────────────────────────────────────
for tau in [0.05, 0.5, 1.0, 5.0]:
    PHASE1_EXPERIMENTS.append((f"da_tau={tau}", {"modulators": {"decay_tau": tau}}))

# ── J. Homeostasis on/off and eta ───────────────────────────────────
PHASE1_EXPERIMENTS.append(("homeo=off", {"homeostasis": {"enabled": False}}))
for eta in [1e-4, 1e-3, 5e-3]:
    PHASE1_EXPERIMENTS.append(
        (
            f"homeo_eta={eta}",
            {"homeostasis": {"enabled": True, "eta": eta}},
        )
    )

# ── K. Gate context amplitude ────────────────────────────────────────
for gc_amp in [0.0, 0.15, 0.30, 0.60]:
    PHASE1_EXPERIMENTS.append(
        (
            f"gc_amp={gc_amp}",
            {
                "logic": {
                    "gate_context": {
                        "enabled": gc_amp > 0,
                        "amplitude": gc_amp,
                    }
                }
            },
        )
    )


def main() -> None:
    tmp_root = Path(os.environ.get("TEMP", "/tmp")) / "biosnn_sweep_or_focused"
    tmp_root.mkdir(parents=True, exist_ok=True)

    # ── Phase 1 ──────────────────────────────────────────────────────
    phase1_results: list[dict] = []
    total = len(PHASE1_EXPERIMENTS)
    print(f"\n{'#' * 90}")
    print(f"  PHASE 1: {total} quick probes at {PHASE1_STEPS} trials each")
    print(f"  Output: {tmp_root}")
    print(f"{'#' * 90}\n")

    for i, (label, patch) in enumerate(PHASE1_EXPERIMENTS, 1):
        # Handle sim_steps override
        sim_steps = 15
        if label.startswith("sim_steps="):
            sim_steps = int(label.split("=")[1])

        # Handle reward_delivery_steps override
        rds = 2
        if label.startswith("reward_steps="):
            rds = int(label.split("=")[1])

        print(f"[{i}/{total}] {label} ... ", end="", flush=True)

        run_dir = tmp_root / label.replace("=", "_").replace("+", "_").replace("/", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = _deep_merge(BASE_SPEC, patch)
        cfg = LogicGateRunConfig(
            gate=LogicGate.OR,
            seed=SEED,
            steps=PHASE1_STEPS,
            sim_steps_per_trial=sim_steps,
            device=DEVICE,
            learning_mode="rstdp",
            engine_learning_rule="three_factor_elig_stdp",
            debug=False,
            out_dir=run_dir,
            artifacts_root=run_dir,
            reward_delivery_steps=rds,
        )
        t0 = time.perf_counter()
        try:
            result = run_logic_gate_engine(cfg, spec)
            elapsed = time.perf_counter() - t0
            r = {
                "label": label,
                "eval_accuracy": result["eval_accuracy"],
                "sample_accuracy": result["sample_accuracy"],
                "passed": result["passed"],
                "first_pass_trial": result["first_pass_trial"],
                "trial_acc_rolling": result["trial_acc_rolling"],
                "mean_abs_dw": result["mean_abs_dw"],
                "elapsed_s": round(elapsed, 1),
                "error": None,
            }
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            traceback.print_exc()
            r = {
                "label": label,
                "eval_accuracy": -1,
                "passed": False,
                "first_pass_trial": -1,
                "trial_acc_rolling": -1,
                "sample_accuracy": -1,
                "mean_abs_dw": -1,
                "elapsed_s": round(elapsed, 1),
                "error": str(exc)[:200],
            }

        phase1_results.append(r)
        tag = "PASS" if r["passed"] else f"acc={r['eval_accuracy']:.2f}"
        err = f"  ERR: {r['error']}" if r["error"] else ""
        print(f"{tag}  ({r['elapsed_s']}s){err}")

    print_results(phase1_results, "PHASE 1 RESULTS (sorted by accuracy)")

    # ── Phase 2: combine best settings ───────────────────────────────
    # Pick the best setting from each axis (highest accuracy in phase 1)
    # and combine into a final validation run at 800 trials.

    print(f"\n{'#' * 90}")
    print("  PHASE 2: combine best per-axis settings + validate at 800 trials")
    print(f"{'#' * 90}\n")

    # Group by axis prefix and find best in each group
    axes: dict[str, dict[str, Any]] = {}
    for r in phase1_results:
        prefix = r["label"].split("=")[0]
        if prefix not in axes or r["eval_accuracy"] > axes[prefix]["eval_accuracy"]:
            axes[prefix] = r

    print("Best per axis:")
    for prefix, r in sorted(axes.items()):
        print(f"  {prefix}: {r['label']}  acc={r['eval_accuracy']:.2f}")

    # Build combined spec from best values
    best_lr = 0.01
    best_da = 1.0
    best_af_amp = 1.0
    best_af_mode = "always"
    best_eps = 0.20
    best_delay = 3
    best_da_tau = 1.0
    best_homeo_eta = 1e-3
    best_homeo_on = True
    best_gc_amp = 0.30
    best_sim_steps = 15
    best_rds = 2

    for r in phase1_results:
        label = r["label"]
        acc = r["eval_accuracy"]
        if label.startswith("lr=") and acc >= axes.get("lr", {}).get("eval_accuracy", -1):
            best_lr = float(label.split("=")[1])
        elif label.startswith("da_amt=") and acc >= axes.get("da_amt", {}).get("eval_accuracy", -1):
            best_da = float(label.split("=")[1])
        elif label.startswith("af_amp=") and acc >= axes.get("af_amp", {}).get("eval_accuracy", -1):
            best_af_amp = float(label.split("=")[1])
        elif label.startswith("af_mode=") and acc >= axes.get("af_mode", {}).get(
            "eval_accuracy", -1
        ):
            best_af_mode = label.split("=")[1]
        elif label.startswith("eps_start=") and acc >= axes.get("eps_start", {}).get(
            "eval_accuracy", -1
        ):
            best_eps = float(label.split("=")[1])
        elif label.startswith("delay=") and acc >= axes.get("delay", {}).get("eval_accuracy", -1):
            best_delay = int(label.split("=")[1])
        elif label.startswith("da_tau=") and acc >= axes.get("da_tau", {}).get("eval_accuracy", -1):
            best_da_tau = float(label.split("=")[1])
        elif label.startswith("homeo_eta=") and acc >= axes.get("homeo_eta", {}).get(
            "eval_accuracy", -1
        ):
            best_homeo_eta = float(label.split("=")[1])
            best_homeo_on = True
        elif label == "homeo=off" and acc >= axes.get("homeo", {}).get("eval_accuracy", -1):
            best_homeo_on = False
        elif label.startswith("gc_amp=") and acc >= axes.get("gc_amp", {}).get("eval_accuracy", -1):
            best_gc_amp = float(label.split("=")[1])
        elif label.startswith("sim_steps=") and acc >= axes.get("sim_steps", {}).get(
            "eval_accuracy", -1
        ):
            best_sim_steps = int(label.split("=")[1])
        elif label.startswith("reward_steps=") and acc >= axes.get("reward_steps", {}).get(
            "eval_accuracy", -1
        ):
            best_rds = int(label.split("=")[1])

    combined_spec = _deep_merge(
        BASE_SPEC,
        {
            "delay_steps": best_delay,
            "learning": {"lr": best_lr},
            "modulators": {"amount": best_da, "decay_tau": best_da_tau},
            "homeostasis": {"enabled": best_homeo_on, "eta": best_homeo_eta},
            "logic": {
                "exploration": {
                    "enabled": best_eps > 0,
                    "epsilon_start": best_eps,
                    "epsilon_end": 0.01 if best_eps > 0 else 0.0,
                    "epsilon_decay_trials": 500,
                },
                "action_force": {
                    "enabled": True,
                    "mode": best_af_mode,
                    "amplitude": best_af_amp,
                },
                "gate_context": {
                    "enabled": best_gc_amp > 0,
                    "amplitude": best_gc_amp,
                },
            },
        },
    )

    combined_label = (
        f"COMBINED: lr={best_lr} da={best_da} af={best_af_amp}/{best_af_mode} "
        f"eps={best_eps} delay={best_delay} tau={best_da_tau} "
        f"homeo={'on' if best_homeo_on else 'off'}/{best_homeo_eta} "
        f"gc={best_gc_amp} sim={best_sim_steps} rds={best_rds}"
    )
    print(f"\nCombined spec: {combined_label}")

    # Run 3 seeds at 800 trials to check robustness
    PHASE2_STEPS = 800
    phase2_results: list[dict] = []
    for seed in [42, 99, 777]:
        label = f"combined_seed={seed}"
        run_dir = tmp_root / f"phase2_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = LogicGateRunConfig(
            gate=LogicGate.OR,
            seed=seed,
            steps=PHASE2_STEPS,
            sim_steps_per_trial=best_sim_steps,
            device=DEVICE,
            learning_mode="rstdp",
            engine_learning_rule="three_factor_elig_stdp",
            debug=False,
            out_dir=run_dir,
            artifacts_root=run_dir,
            reward_delivery_steps=best_rds,
        )
        print(f"\n  Phase 2: {label} ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            result = run_logic_gate_engine(cfg, combined_spec)
            elapsed = time.perf_counter() - t0
            r = {
                "label": label,
                "eval_accuracy": result["eval_accuracy"],
                "sample_accuracy": result["sample_accuracy"],
                "passed": result["passed"],
                "first_pass_trial": result["first_pass_trial"],
                "trial_acc_rolling": result["trial_acc_rolling"],
                "mean_abs_dw": result["mean_abs_dw"],
                "elapsed_s": round(elapsed, 1),
                "error": None,
            }
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            traceback.print_exc()
            r = {
                "label": label,
                "eval_accuracy": -1,
                "passed": False,
                "first_pass_trial": -1,
                "trial_acc_rolling": -1,
                "sample_accuracy": -1,
                "mean_abs_dw": -1,
                "elapsed_s": round(elapsed, 1),
                "error": str(exc)[:200],
            }
        phase2_results.append(r)
        tag = "PASS" if r["passed"] else f"acc={r['eval_accuracy']:.2f}"
        print(f"{tag}  ({r['elapsed_s']}s)")

    print_results(phase2_results, "PHASE 2 RESULTS")

    # ── Final summary ────────────────────────────────────────────────
    passed_p2 = [r for r in phase2_results if r["passed"]]
    print(f"\nPhase 2: {len(passed_p2)}/{len(phase2_results)} seeds passed.")
    if passed_p2:
        best = min(passed_p2, key=lambda x: x["first_pass_trial"] or 99999)
        print(f"Fastest convergence: {best['label']} at trial {best['first_pass_trial']}")
    print(f"\nBest parameters:\n{combined_label}")
    print("\nFull combined spec:")
    import json

    print(json.dumps(combined_spec, indent=2, default=str))


if __name__ == "__main__":
    main()
