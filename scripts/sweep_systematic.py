"""Sweep: Systematic search for working engine-path OR learning.

Key hypothesis: sim_steps=30 creates overwhelming stimulus-phase eligibility
that masks the reward-phase differential. Try shorter sim_steps and denser
topology (higher fan_in) to match the runner path's success factors.
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
    skip_fan_in: int = 2,
    hidden_fan_in: int = 2,
    excit_target: str = "soma",
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
            "skip_fan_in": skip_fan_in,
            "in_to_hidden_fan_in": hidden_fan_in,
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
    print(f"  {tag} {label:60s} eval={ev:.2f} p={p}")
    sys.stdout.flush()
    return ev


def main() -> None:
    best_ev = 0.0
    best_cfg = ""

    # ── Phase 1: sim_steps impact ──────────────────────────────────
    print("=" * 80)
    print("Phase 1: sim_steps × lr_mult  (a_minus=0, w_min=-wm, tau_e=0.01)")
    print("=" * 80)
    for ss in [5, 10, 15]:
        for lr_m in [0.05, 0.08, 0.15]:
            label = f"ss={ss:2d} rw=5 lr={lr_m}"
            ev = run(
                sim_steps=ss,
                reward_steps=5,
                lr_mult=lr_m,
                trials=400,
                a_minus=0.0,
                w_min_neg=True,
                label=label,
            )
            if ev > best_ev:
                best_ev = ev
                best_cfg = label

    # ── Phase 2: Topology density ──────────────────────────────────
    print("\n" + "=" * 80)
    print("Phase 2: fan_in sweep  (ss=10, lr=0.08, a_minus=0, w_min=-wm)")
    print("=" * 80)
    for h_fi in [2, 3, 5]:
        for s_fi in [2, 3, 5]:
            label = f"h_fi={h_fi} s_fi={s_fi}"
            ev = run(
                sim_steps=10,
                reward_steps=5,
                lr_mult=0.08,
                hidden_fan_in=h_fi,
                skip_fan_in=s_fi,
                trials=400,
                a_minus=0.0,
                w_min_neg=True,
                label=label,
            )
            if ev > best_ev:
                best_ev = ev
                best_cfg = label

    # ── Phase 3: excit_target (dendrite vs soma) ──────────────────
    print("\n" + "=" * 80)
    print("Phase 3: dendrite vs soma targeting (ss=10, lr=0.08)")
    print("=" * 80)
    for tgt in ["soma", "dendrite"]:
        for lr_m in [0.05, 0.08]:
            label = f"target={tgt:8s} lr={lr_m}"
            ev = run(
                sim_steps=10,
                reward_steps=5,
                lr_mult=lr_m,
                excit_target=tgt,
                trials=400,
                a_minus=0.0,
                w_min_neg=True,
                label=label,
            )
            if ev > best_ev:
                best_ev = ev
                best_cfg = label

    # ── Phase 4: DA parameters ────────────────────────────────────
    print("\n" + "=" * 80)
    print("Phase 4: DA decay + amount sweep (ss=10, lr=0.08)")
    print("=" * 80)
    for da_d in [0.01, 0.05, 0.20, 1.0]:
        label = f"da_decay={da_d}"
        ev = run(
            sim_steps=10,
            reward_steps=5,
            lr_mult=0.08,
            da_decay=da_d,
            trials=400,
            a_minus=0.0,
            w_min_neg=True,
            label=label,
        )
        if ev > best_ev:
            best_ev = ev
            best_cfg = label

    # ── Phase 5: gws sweep ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Phase 5: Global weight scale  (ss=10, lr_mult=0.08)")
    print("=" * 80)
    for gws in [1e-7, 5e-7, 1e-6, 5e-6]:
        label = f"gws={gws:.0e}"
        ev = run(
            sim_steps=10,
            reward_steps=5,
            lr_mult=0.08,
            gws=gws,
            trials=400,
            a_minus=0.0,
            w_min_neg=True,
            label=label,
        )
        if ev > best_ev:
            best_ev = ev
            best_cfg = label

    print(f"\n\nBest overall: {best_cfg}  eval={best_ev:.2f}")

    # ── Phase 6: Seed sweep if found something ────────────────────
    if best_ev >= 1.0:
        print("\n" + "=" * 80)
        print("Phase 6: Seed sweep with best config")
        print("=" * 80)
        for seed in [42, 7, 23, 123, 314, 500]:
            label = f"seed={seed} BEST={best_cfg}"
            # Parse params from best_cfg and run — simplified for now
            run(
                seed=seed,
                sim_steps=10,
                reward_steps=5,
                lr_mult=0.08,
                trials=800,
                a_minus=0.0,
                w_min_neg=True,
                label=label,
            )

    print("\n\nDone.")


if __name__ == "__main__":
    main()
