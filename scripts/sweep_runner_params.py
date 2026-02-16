"""Sweep: Runner-equivalent params on engine path.

Key findings from runner path (which passes acceptance tests):
  a_minus=0.0  (no LTD!)
  w_min=-1.0   (negative weights)
  tau_e=0.01   (fast eligibility decay)
  lr=0.08      (relative to w_scale 0.03)

We translate those to the engine path's weight scale.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_cfg(
    *,
    seed: int = 42,
    gws: float = 5e-7,
    af_mode: str = "always",
    trials: int = 800,
    sim_steps: int = 30,
    reward_steps: int = 10,
    lr_mult: float = 0.08,
    tau_e: float = 0.010,
    da_decay: float = 0.05,
    excit_target: str = "soma",
    synaptic_scaling: bool = False,
    weight_decay: float = 0.0,
    a_minus: float = 0.0,
    w_min_neg: bool = True,
    baseline: float = 0.0,
    label: str = "",
) -> dict[str, Any]:
    w_max = gws * 3.0
    lr = w_max * lr_mult
    w_min = -w_max if w_min_neg else 0.0

    spec: dict[str, Any] = {
        "dtype": "float32",
        "delay_steps": 3,
        "synapse": {
            "backend": "spmm_fused",
            "fused_layout": "auto",
            "ring_strategy": "dense",
            "ring_dtype": "none",
            "receptor_mode": "exc_only",
            "global_weight_scale": gws,
            "skip_fan_in": 2,
            "in_to_hidden_fan_in": 2,
            "excit_target_compartment": excit_target,
        },
        "learning": {
            "enabled": True,
            "rule": "rstdp_eligibility",
            "lr": lr,
            "w_min": w_min,
            "w_max": w_max,
            "tau_e": tau_e,
            "tau_pre": 0.020,
            "tau_post": 0.020,
            "a_plus": 1.0,
            "a_minus": a_minus,
            "dopamine_scale": 1.0,
            "baseline": baseline,
            "weight_decay": weight_decay,
            "synaptic_scaling": synaptic_scaling,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": da_decay,
        },
        "wrapper": {
            "enabled": True,
            "spike_window": sim_steps,
            "decision_mode": "spike_count",
        },
        "homeostasis": {
            "enabled": True,
            "target_rate": 0.05,
            "tau": 1.0,
            "gain": 0.001,
        },
        "logic": {
            "exploration": {
                "enabled": True,
                "epsilon_start": 0.3,
                "epsilon_end": 0.01,
                "epsilon_decay_trials": 200,
                "tie_break": "random_among_max",
            },
            "action_force": {
                "enabled": True,
                "mode": af_mode,
                "amplitude": 1.0,
                "window": "reward_window",
                "steps": reward_steps,
                "compartment": "soma",
                "suppression_factor": -3.0,
            },
        },
    }

    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=seed,
        steps=trials,
        dt=1e-3,
        sim_steps_per_trial=sim_steps,
        device="cuda",
        learning_mode="rstdp",
        engine_learning_rule="rstdp_elig",
        inter_trial_reset=True,
        drive_scale=1e-9,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=reward_steps,
        reward_delivery_clamp_input=True,
        debug=True,
    )

    r = run_logic_gate_engine(cfg, spec)
    ev = r["eval_accuracy"]
    preds = r["preds"]
    tag = "OK" if ev >= 1.0 else "  "
    print(f"  {tag} {label:55s}  eval={ev:.2f}  preds={preds}")
    sys.stdout.flush()
    return r


def main() -> None:
    # ── Phase 1: Runner-equivalent params ──────────────────────────────
    print("=" * 80)
    print("Phase 1: Runner-equivalent params (a_minus=0, w_min=-w_max, tau_e=0.01)")
    print("  seed=42  trials=800  gws=5e-7  excit_target=soma  af_mode=always")
    print("=" * 80)

    results_p1: list[tuple[str, float]] = []
    for lr_m in [0.01, 0.05, 0.08, 0.15]:
        label = f"lr_mult={lr_m:<6}"
        r = run_cfg(lr_mult=lr_m, a_minus=0.0, w_min_neg=True, tau_e=0.01, trials=800, label=label)
        results_p1.append((label, r["eval_accuracy"]))

    # Also test with a_minus=0.0 but w_min=0 (positive only)
    print("\n  -- Positive-only weights (w_min=0), a_minus=0.0 --")
    for lr_m in [0.01, 0.05, 0.08]:
        label = f"lr_mult={lr_m:<6} w_min=0"
        r = run_cfg(lr_mult=lr_m, a_minus=0.0, w_min_neg=False, tau_e=0.01, trials=800, label=label)
        results_p1.append((label, r["eval_accuracy"]))

    # Test with a_minus=0.6 (original) but negative weights
    print("\n  -- Negative weights + a_minus=0.6 (old LTD) --")
    for lr_m in [0.01, 0.05, 0.08]:
        label = f"lr_mult={lr_m:<6} a_minus=0.6"
        r = run_cfg(lr_mult=lr_m, a_minus=0.6, w_min_neg=True, tau_e=0.01, trials=800, label=label)
        results_p1.append((label, r["eval_accuracy"]))

    print("\n--- Phase 1 summary ---")
    best = max(results_p1, key=lambda x: x[1])
    for label, ev in results_p1:
        star = " ***" if ev >= 1.0 else ""
        print(f"  {label}  eval={ev:.2f}{star}")
    print(f"  Best: {best[0]}  eval={best[1]:.2f}")

    # ── Phase 2: Seed sweep with best config ──────────────────────────
    print("\n" + "=" * 80)
    print("Phase 2: Seed sweep with best Phase 1 config")
    print("=" * 80)

    # Parse best config
    best_label = best[0]
    best_lr_m = float(best_label.split("lr_mult=")[1].split()[0])
    best_w_neg = "w_min=0" not in best_label
    best_a_minus = 0.6 if "a_minus=0.6" in best_label else 0.0

    seed_results: list[tuple[int, float]] = []
    for seed in [42, 7, 23, 123, 314, 500, 999]:
        label = f"seed={seed:<6} lr_mult={best_lr_m} a_minus={best_a_minus}"
        r = run_cfg(
            seed=seed,
            lr_mult=best_lr_m,
            a_minus=best_a_minus,
            w_min_neg=best_w_neg,
            tau_e=0.01,
            trials=800,
            label=label,
        )
        seed_results.append((seed, r["eval_accuracy"]))

    print("\n--- Phase 2 summary ---")
    pass_count = sum(1 for _, ev in seed_results if ev >= 1.0)
    for seed, ev in seed_results:
        print(f"  seed={seed}  eval={ev:.2f}")
    print(f"  Passed: {pass_count}/{len(seed_results)}")

    # ── Phase 3: If not all seeds pass, try recovery ──────────────────
    fail_seeds = [s for s, ev in seed_results if ev < 1.0]
    if fail_seeds and pass_count >= 3:
        print("\n" + "=" * 80)
        print("Phase 3: Recovery for failed seeds with tau_e/lr adjustments")
        print("=" * 80)
        for seed in fail_seeds[:3]:
            for lr_m in [0.02, 0.05, 0.10]:
                label = f"seed={seed} lr_mult={lr_m} tau_e=0.005"
                run_cfg(
                    seed=seed,
                    lr_mult=lr_m,
                    a_minus=0.0,
                    w_min_neg=True,
                    tau_e=0.005,
                    trials=800,
                    label=label,
                )

    print("\n\nDone.")


if __name__ == "__main__":
    main()
