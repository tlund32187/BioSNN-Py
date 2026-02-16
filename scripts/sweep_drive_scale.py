"""Sweep drive_scale to fix external drives from raw Amperes to biophysical nA range.

AdEx3c params: C_s=200pF, g_L=10nS, E_L=-70mV, V_T=-50mV → rheobase=200pA.
Without drive_scale, input high=1.0A → 5GV/step (5 billion × rheobase).
drive_scale=1e-9 → input=1nA (5× rheobase), gate_context=0.3nA, action_force~1nA.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def run_one(*, drive_scale: float, gws: float, label: str) -> dict[str, Any]:
    w_max = gws * 3.0
    lr = w_max * 0.01

    spec: dict[str, Any] = {
        "dtype": "float32",
        "delay_steps": 3,
        "synapse": {
            "backend": "spmm_fused",
            "fused_layout": "auto",
            "ring_strategy": "dense",
            "ring_dtype": "none",
            "store_sparse_by_delay": None,
            "receptor_mode": "exc_only",
            "global_weight_scale": gws,
        },
        "learning": {
            "enabled": True,
            "rule": "three_factor_elig_stdp",
            "lr": lr,
            "w_min": 0.0,
            "w_max": w_max,
            "a_plus": 1.0,
            "a_minus": 0.6,
            "tau_e": 0.001,
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
        "homeostasis": {"enabled": False},
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
                "steps": 10,
                "compartment": "soma",
                "suppression_factor": -3.0,
            },
        },
    }

    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=42,
        steps=800,
        dt=1e-3,
        sim_steps_per_trial=15,
        device="cuda",
        learning_mode="rstdp",
        engine_learning_rule="three_factor_elig_stdp",
        inter_trial_reset=True,
        drive_scale=drive_scale,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=10,
        reward_delivery_clamp_input=True,
        debug=False,
        dump_last_trials_csv=False,
    )

    result = run_logic_gate_engine(cfg, spec)
    eval_acc = result.get("eval_accuracy", 0.0)
    sample_acc = result.get("sample_accuracy", 0.0)
    spk = result.get("eval_spike_counts", [0, 0])
    tie_pct = result.get("tie_count", 0) / max(1, result.get("eval_trials", 1)) * 100.0
    print(
        f"  {label}: eval={eval_acc:.4f}  sample={sample_acc:.4f}  "
        f"spk={[round(s, 2) for s in spk]}  ties={tie_pct:.0f}%"
    )
    return result


def main() -> None:
    # Sweep drive_scale and gws combinations
    drive_scales = [1e-9, 5e-10, 2e-10, 1e-10]
    # With drive_scale, weights need to match drive magnitude.
    # Input drive (effective) = 1.0 × ds.  Weight ≈ 0.03 × gws.
    # For balance: 0.03 × gws ≈ ds → gws ≈ ds / 0.03 ≈ 33 × ds.
    # But we want weights slightly below drives for selective firing.
    gws_options = [1e-7, 1e-8, 1e-9]

    for ds in drive_scales:
        print(f"\n=== drive_scale = {ds:.0e} ===")
        print(
            f"  (effective input drive = {1.0 * ds:.1e} A, "
            f"rheobase = 200 pA = 2e-10 A, "
            f"ratio = {1.0 * ds / 2e-10:.1f}x)"
        )
        for gws in gws_options:
            w_max = gws * 3.0
            lr = w_max * 0.01
            label = f"gws={gws:.0e} lr={lr:.0e} w_max={w_max:.0e}"
            try:
                run_one(drive_scale=ds, gws=gws, label=label)
            except Exception as e:
                print(f"  {label}: ERROR {e}")


if __name__ == "__main__":
    main()
