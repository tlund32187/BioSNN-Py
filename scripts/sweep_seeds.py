"""Sweep: Seed search + baseline comparison for engine-path OR learning.

The skip topology is seed-dependent. Seed=42's topology is adversarial for OR.
Let's find seeds where the topology enables OR learning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run(
    *,
    seed: int = 42,
    gws: float = 5e-7,
    trials: int = 800,
    sim_steps: int = 10,
    reward_steps: int = 5,
    lr_mult: float = 0.08,
    tau_e: float = 0.010,
    da_decay: float = 0.05,
    a_minus: float = 0.0,
    w_min_neg: bool = True,
    excit_target: str = "soma",
    learning_enabled: bool = True,
    label: str = "",
) -> float:
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
            "enabled": learning_enabled,
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
            "baseline": 0.0,
            "weight_decay": 0.0,
            "synaptic_scaling": False,
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
                "mode": "always",
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
    p = r["preds"]
    tag = "OK" if ev >= 1.0 else "  "
    print(f"  {tag} {label:55s} eval={ev:.2f} p={p}")
    sys.stdout.flush()
    return ev


def main() -> None:
    # ── Phase 1: Baseline (no learning) across seeds ──────────────
    print("=" * 80)
    print("Phase 1: BASELINE (learning OFF) across seeds  (ss=10, rw=5)")
    print("=" * 80)
    baselines: dict[int, float] = {}
    for seed in range(1, 51):
        label = f"seed={seed:3d} BASELINE"
        ev = run(seed=seed, trials=16, learning_enabled=False, label=label)
        baselines[seed] = ev

    print("\n--- Phase 1 summary: seeds with baseline > 0.75 ---")
    good_seeds = [(s, e) for s, e in baselines.items() if e > 0.75]
    for s, e in sorted(good_seeds, key=lambda x: -x[1]):
        print(f"  seed={s}  baseline={e:.2f}")
    if not good_seeds:
        print("  No seeds with baseline > 0.75. Using seeds with baseline=0.75.")
        good_seeds = [(s, e) for s, e in baselines.items() if e >= 0.75]

    # ── Phase 2: Learning on diverse seeds ────────────────────────
    print("\n" + "=" * 80)
    print("Phase 2: Learning ON for diverse seeds (800 trials)")
    print("=" * 80)
    test_seeds = [s for s, _ in sorted(good_seeds, key=lambda x: -x[1])[:15]]
    learn_results: list[tuple[int, float]] = []
    for seed in test_seeds:
        label = f"seed={seed:3d} lr=0.08 a_minus=0 w_neg"
        ev = run(
            seed=seed,
            trials=800,
            lr_mult=0.08,
            a_minus=0.0,
            w_min_neg=True,
            tau_e=0.01,
            label=label,
        )
        learn_results.append((seed, ev))

    print("\n--- Phase 2 summary ---")
    pass_count = sum(1 for _, ev in learn_results if ev >= 1.0)
    for seed, ev in learn_results:
        star = " ***" if ev >= 1.0 else ""
        print(f"  seed={seed}  eval={ev:.2f}{star}")
    print(f"  Passed: {pass_count}/{len(learn_results)}")

    # ── Phase 3: If found good seeds, tune; else try wider params ─
    ok_seeds = [s for s, ev in learn_results if ev >= 1.0]
    if ok_seeds:
        print("\n" + "=" * 80)
        print(f"Phase 3: Robustness check on seed={ok_seeds[0]}")
        print("=" * 80)
        for lr_m in [0.01, 0.03, 0.05, 0.08, 0.12, 0.20]:
            label = f"seed={ok_seeds[0]} lr={lr_m}"
            run(
                seed=ok_seeds[0],
                trials=800,
                lr_mult=lr_m,
                a_minus=0.0,
                w_min_neg=True,
                tau_e=0.01,
                label=label,
            )
    else:
        print("\n" + "=" * 80)
        print("Phase 3: Wide lr sweep on diverse seeds")
        print("=" * 80)
        for seed in test_seeds[:5]:
            for lr_m in [0.001, 0.005, 0.01, 0.05]:
                label = f"seed={seed:3d} lr={lr_m}"
                ev = run(
                    seed=seed,
                    trials=800,
                    lr_mult=lr_m,
                    a_minus=0.0,
                    w_min_neg=True,
                    tau_e=0.01,
                    label=label,
                )
                if ev >= 1.0:
                    print(f"    >>> FOUND: seed={seed} lr={lr_m}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
