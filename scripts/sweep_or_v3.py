"""Sweep to push from 0.75 → 1.0 accuracy.
We know a_minus=0 gives 0.75 (always predicts 1 — gets 3/4 right).
We need a_minus > 0 so that suppressed outputs (post=0) get negative elig ->
when combined with negative DA (wrong trial), creates positive dW for correct output.
But a_plus must dominate: a_plus > 2*a_minus for net positive elig when post fires.
Also try larger tau_e to let elig accumulate more before reward.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def make_spec(
    *,
    lr: float,
    a_plus: float,
    a_minus: float,
    tau_e: float,
    rds: int,
    sf: float,
    homeo: bool,
    mod_thresh: float,
    explore_start: float = 0.20,
) -> dict[str, Any]:
    return {
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
            "lr": lr,
            "w_min": 0.0,
            "w_max": 1.0,
            "a_plus": a_plus,
            "a_minus": a_minus,
            "tau_e": tau_e,
            "modulator_threshold": mod_thresh,
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
            "enabled": homeo,
            "alpha": 0.01,
            "eta": 1e-3,
            "r_target": 0.05,
            "clamp_min": 0.0,
            "clamp_max": 0.05,
        },
        "logic": {
            "exploration": {
                "enabled": True,
                "epsilon_start": explore_start,
                "epsilon_end": 0.01,
                "epsilon_decay_trials": 300,
            },
            "action_force": {
                "enabled": True,
                "mode": "always",
                "amplitude": 1.0,
                "window": "reward_window",
                "steps": rds,
                "compartment": "soma",
                "suppression_factor": sf,
            },
            "gate_context": {
                "enabled": True,
                "amplitude": 0.30,
                "compartment": "dendrite",
                "targets": ["hidden"],
            },
        },
    }


def main() -> None:
    device = "cuda"
    rds = 10
    n_trials = 800

    configs = [
        # (label, a_plus, a_minus, lr, tau_e, sf, homeo, mod_thresh, explore)
        # Baseline: 0.75 confirmed
        ("baseline_am0", 1.0, 0.0, 0.01, 0.001, -3.0, False, 0.5, 0.20),
        # Small a_minus with high ratio
        ("ap3_am05", 3.0, 0.5, 0.01, 0.001, -3.0, False, 0.5, 0.20),
        ("ap5_am1", 5.0, 1.0, 0.01, 0.001, -3.0, False, 0.5, 0.20),
        ("ap3_am03", 3.0, 0.3, 0.01, 0.001, -3.0, False, 0.5, 0.20),
        # Small a_minus, smaller lr
        ("ap3_am05_lr005", 3.0, 0.5, 0.005, 0.001, -3.0, False, 0.5, 0.20),
        # Slightly longer tau_e (2ms — 1 refractory period)
        ("ap3_am05_te2", 3.0, 0.5, 0.01, 0.002, -3.0, False, 0.5, 0.20),
        ("ap3_am1_te2", 3.0, 1.0, 0.01, 0.002, -3.0, False, 0.5, 0.20),
        # Original tau_e=0.001 but very small a_minus
        ("ap1_am005", 1.0, 0.05, 0.01, 0.001, -3.0, False, 0.5, 0.20),
        ("ap1_am01", 1.0, 0.1, 0.01, 0.001, -3.0, False, 0.5, 0.20),
        # Same with higher exploration to help discover (0,0) case
        ("ap1_am01_exp50", 1.0, 0.1, 0.01, 0.001, -3.0, False, 0.5, 0.50),
        # With homeostasis
        ("ap3_am05_hom", 3.0, 0.5, 0.01, 0.001, -3.0, True, 0.5, 0.20),
        # Weaker suppression (maybe out0 needs some action during wrong trials)
        ("ap1_am01_sf05", 1.0, 0.1, 0.01, 0.001, -0.5, False, 0.5, 0.20),
    ]

    print(f"Running {len(configs)} experiments, {n_trials} trials each")
    print("=" * 80)

    results: list[tuple[str, float, float, float]] = []
    for label, a_plus, a_minus, lr, tau_e, sf, homeo, mt, exp in configs:
        spec = make_spec(
            lr=lr,
            a_plus=a_plus,
            a_minus=a_minus,
            tau_e=tau_e,
            rds=rds,
            sf=sf,
            homeo=homeo,
            mod_thresh=mt,
            explore_start=exp,
        )
        out = Path(tempfile.mkdtemp(prefix=f"biosnn_v3_{label}_"))
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
            reward_delivery_steps=rds,
            reward_delivery_clamp_input=True,
        )
        t0 = time.perf_counter()
        result = run_logic_gate_engine(config=cfg, run_spec=spec)
        elapsed = time.perf_counter() - t0
        acc = float(result.get("eval_accuracy", 0.0))
        w0 = float(result.get("w_out0_mean", 0.0))
        w1 = float(result.get("w_out1_mean", 0.0))
        results.append((label, acc, w0, w1))
        tag = "***" if acc > 0.75 else "== " if acc == 0.75 else "   "
        print(f"{tag} {label:22s}  acc={acc:.4f}  w0={w0:.4f}  w1={w1:.4f}  ({elapsed:.1f}s)")

    print("\n" + "=" * 80)
    print("SORTED RESULTS")
    for label, acc, w0, w1 in sorted(results, key=lambda x: -x[1]):
        tag = "***" if acc > 0.75 else "== " if acc == 0.75 else "   "
        print(f"{tag} {label:22s}  acc={acc:.4f}  w0={w0:.4f}  w1={w1:.4f}")


if __name__ == "__main__":
    main()
