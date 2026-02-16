"""Sweep: synaptic_scaling + tau_e + lr_mult calibration for OR learning."""

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
    trials: int = 400,
    sim_steps: int = 30,
    reward_steps: int = 10,
    lr_mult: float = 0.01,
    tau_e: float = 0.050,
    da_decay: float = 0.05,
    excit_target: str = "soma",
    synaptic_scaling: bool = True,
    weight_decay: float = 0.0,
    label: str = "",
) -> dict[str, Any]:
    w_max = gws * 3.0
    lr = w_max * lr_mult

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
            "w_min": 0.0,
            "w_max": w_max,
            "tau_e": tau_e,
            "tau_pre": 0.020,
            "tau_post": 0.020,
            "a_plus": 1.0,
            "a_minus": 0.6,
            "dopamine_scale": 1.0,
            "baseline": 0.0,
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
                "epsilon_end": 0.05,
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
    print(f"  {tag} {label:50s}  eval={ev:.2f}  preds={preds}")
    sys.stdout.flush()
    return r


def main() -> None:
    print("=" * 78)
    print("Phase 1: synaptic_scaling + tau_e + lr_mult grid  (seed=42, 400 trials)")
    print("=" * 78)
    results_p1: list[tuple[str, float, float, float]] = []
    for tau_e in [0.005, 0.010, 0.020, 0.050]:
        for lr_m in [0.0001, 0.001, 0.01]:
            label = f"tau_e={tau_e:<6} lr_mult={lr_m}"
            r = run_cfg(
                lr_mult=lr_m,
                tau_e=tau_e,
                trials=400,
                synaptic_scaling=True,
                label=label,
            )
            results_p1.append((label, r["eval_accuracy"], tau_e, lr_m))

    # Summary
    print("\n--- Phase 1 summary ---")
    best = max(results_p1, key=lambda x: x[1])
    for label, ev, te, lrm in results_p1:
        star = " ***" if ev >= 1.0 else ""
        print(f"  {label}  eval={ev:.2f}{star}")
    print(f"  Best: {best[0]}  eval={best[1]:.2f}")
    best_tau_e = best[2]
    best_lr_m = best[3]

    print("\n" + "=" * 78)
    print(f"Phase 2: Seed sweep with best config  (tau_e={best_tau_e}, lr_mult={best_lr_m})")
    print("=" * 78)
    seed_results = []
    for seed in [42, 7, 123, 314, 500, 999]:
        label = f"seed={seed:<6} tau_e={best_tau_e} lr_mult={best_lr_m}"
        r = run_cfg(
            seed=seed,
            lr_mult=best_lr_m,
            tau_e=best_tau_e,
            trials=400,
            synaptic_scaling=True,
            label=label,
        )
        seed_results.append((seed, r["eval_accuracy"]))

    print("\n--- Phase 2 summary ---")
    pass_count = sum(1 for _, ev in seed_results if ev >= 1.0)
    for seed, ev in seed_results:
        print(f"  seed={seed}  eval={ev:.2f}")
    print(f"  Passed: {pass_count}/{len(seed_results)}")

    # Phase 3: If Phase1 had no perfect result, also try synaptic_scaling=False
    # as a comparison, and try with weight_decay
    if best[1] < 1.0:
        print("\n" + "=" * 78)
        print("Phase 3: Alternatives (weight_decay, scaling off)")
        print("=" * 78)
        for wd in [0.01, 0.1, 1.0]:
            for lr_m in [0.001, 0.01]:
                label = f"wd={wd:<5} lr_mult={lr_m} scaling=off"
                run_cfg(
                    lr_mult=lr_m,
                    weight_decay=wd,
                    tau_e=0.010,
                    trials=400,
                    synaptic_scaling=False,
                    label=label,
                )
        # Also try with negative baseline (bias toward weight decrease)
        for bl in [-0.3, -0.5]:
            label = f"baseline={bl} lr_mult=0.001 scaling=on"
            # Need to set baseline in spec, but run_cfg doesn't expose it.
            # Skip for now — would require modifying run_cfg.
            pass

    print("\n\nDone.")


if __name__ == "__main__":
    main()
