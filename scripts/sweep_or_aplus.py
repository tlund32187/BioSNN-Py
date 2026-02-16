"""Sweep a_plus / a_minus ratio to compensate for refractory-period elig inversion.

Root cause: AdEx3c refrac_period=0.002, dt=0.001 → neurons fire 1/3 of steps.
With a_plus=a_minus=1.0, average eligibility increment is -1/9 (negative!).
Need a_plus > 2*a_minus for net positive elig on coincident firing.
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
    suppression_factor: float,
    homeostasis_enabled: bool,
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
            "enabled": homeostasis_enabled,
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
                "steps": rds,
                "compartment": "soma",
                "suppression_factor": suppression_factor,
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
    n_trials = 800
    device = "cuda"
    rds = 10

    experiments: list[tuple[str, dict[str, Any]]] = []

    # Key parameter combinations to test
    configs = [
        # (label, a_plus, a_minus, lr, tau_e, suppression_factor, homeostasis)
        ("ap3_am1_lr01", 3.0, 1.0, 0.01, 0.001, -3.0, True),
        ("ap3_am1_lr05", 3.0, 1.0, 0.05, 0.001, -3.0, True),
        ("ap5_am1_lr01", 5.0, 1.0, 0.01, 0.001, -3.0, True),
        ("ap5_am1_lr05", 5.0, 1.0, 0.05, 0.001, -3.0, True),
        ("ap1_am0_lr01", 1.0, 0.0, 0.01, 0.001, -3.0, True),  # pure Hebbian
        ("ap1_am0_lr05", 1.0, 0.0, 0.05, 0.001, -3.0, True),
        ("ap3_am1_lr01_noh", 3.0, 1.0, 0.01, 0.001, -3.0, False),  # no homeostasis
        ("ap3_am1_lr05_noh", 3.0, 1.0, 0.05, 0.001, -3.0, False),
        ("ap3_am1_lr10", 3.0, 1.0, 0.10, 0.001, -3.0, True),
        ("ap5_am1_lr10", 5.0, 1.0, 0.10, 0.001, -3.0, True),
        ("ap10_am1_lr01", 10.0, 1.0, 0.01, 0.001, -3.0, True),  # strong compensation
        ("ap3_am1_lr01_te3", 3.0, 1.0, 0.01, 0.003, -3.0, True),  # tau_e = 1 refrac cycle
    ]

    for label, a_plus, a_minus, lr, tau_e, sf, homeo in configs:
        spec = make_spec(
            lr=lr,
            a_plus=a_plus,
            a_minus=a_minus,
            tau_e=tau_e,
            rds=rds,
            suppression_factor=sf,
            homeostasis_enabled=homeo,
        )
        experiments.append((label, spec))

    print(f"Running {len(experiments)} experiments, {n_trials} trials each")
    print("=" * 80)

    results: list[tuple[str, float, float, float, float]] = []

    for label, spec in experiments:
        out = Path(tempfile.mkdtemp(prefix=f"biosnn_ap_{label}_"))
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
        dw = float(result.get("dw", 0.0))
        w0 = float(result.get("w_out0_mean", 0.0))
        w1 = float(result.get("w_out1_mean", 0.0))
        results.append((label, acc, dw, w0, w1))
        tag = "***" if acc > 0.70 else "   "
        print(
            f"{tag} {label:25s}  acc={acc:.4f}  dw={dw:.6f}  w0={w0:.4f}  w1={w1:.4f}  ({elapsed:.1f}s)"
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for label, acc, dw, w0, w1 in sorted(results, key=lambda x: -x[1]):
        tag = "***" if acc > 0.70 else "   "
        print(f"{tag} {label:25s}  acc={acc:.4f}  w0={w0:.4f}  w1={w1:.4f}")


if __name__ == "__main__":
    main()
