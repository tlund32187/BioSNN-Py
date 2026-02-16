"""Diagnostic: track DA+eligibility co-occurrence and actual weight changes.

Key question: When DA > 0 and elig > 0, what is dw?
Also tracks per-projection info to identify which projections are learning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _tmax(t: Any) -> float:
    if t is None:
        return 0.0
    if hasattr(t, "is_sparse") and t.is_sparse:
        v = t.values()
        return float(v.abs().max().item()) if v.numel() > 0 else 0.0
    return float(t.abs().max().item()) if hasattr(t, "numel") and t.numel() > 0 else 0.0


def _tsum(t: Any) -> float:
    if t is None:
        return 0.0
    if hasattr(t, "is_sparse") and t.is_sparse:
        v = t.values()
        return float(v.sum().item()) if v.numel() > 0 else 0.0
    return float(t.sum().item()) if hasattr(t, "numel") and t.numel() > 0 else 0.0


def _tnnz(t: Any) -> int:
    if t is None:
        return 0
    if hasattr(t, "is_sparse") and t.is_sparse:
        v = t.values()
        return int((v != 0).sum().item()) if v.numel() > 0 else 0
    return int((t != 0).sum().item()) if hasattr(t, "numel") and t.numel() > 0 else 0


# Monkey-patch RSTDP step
import biosnn.learning.rules.rstdp_eligibility as rstdp_mod

_orig_step = rstdp_mod.RStdpEligibilityRule.step

_call_count = 0
_cooccurrence = []  # (call#, DA, elig, dw, pre_nnz, post_nnz, w_mean, w_range)
_weight_snapshots = {}  # proj_id -> [initial_mean, final_mean]


def _patched_step(self, state, batch, *, dt, t=None, ctx=None):
    global _call_count
    _call_count += 1

    from biosnn.contracts.modulators import ModulatorKind

    da_val = 0.0
    if batch.modulators:
        da_tensor = batch.modulators.get(ModulatorKind.DOPAMINE)
        if da_tensor is not None:
            da_val = _tmax(da_tensor)

    # Capture eligibility BEFORE step (to see what it was when DA arrived)
    elig_before = _tmax(state.eligibility)
    w_mean = _tmax(batch.weights) if batch.weights is not None else 0.0

    new_state, result = _orig_step(self, state, batch, dt=dt, t=t, ctx=ctx)

    elig_after = _tmax(new_state.eligibility)
    dwmax = _tmax(result.d_weights)
    pre_nnz = _tnnz(batch.pre_spikes)
    post_nnz = _tnnz(batch.post_spikes)

    # Track weight statistics using projection identity (edge count)
    e = batch.weights.numel() if batch.weights is not None else 0
    pid = e  # Use edge count as proxy for projection ID
    if pid not in _weight_snapshots:
        w_mean_val = (
            float(batch.weights.mean().item())
            if batch.weights is not None and batch.weights.numel() > 0
            else 0.0
        )
        _weight_snapshots[pid] = {"initial_mean": w_mean_val, "latest_mean": w_mean_val, "calls": 0}
    _weight_snapshots[pid]["calls"] += 1
    if batch.weights is not None and batch.weights.numel() > 0:
        _weight_snapshots[pid]["latest_mean"] = float(batch.weights.mean().item())

    # Log ALL co-occurrences where effective_DA ≠ 0 (baseline or real DA) and elig ≠ 0
    effective_da = da_val * float(self.params.dopamine_scale) + float(self.params.baseline)
    if elig_before > 1e-10 and abs(effective_da) > 1e-10:
        _cooccurrence.append(
            {
                "call": _call_count,
                "DA": da_val,
                "eff_DA": effective_da,
                "elig_before": elig_before,
                "elig_after": elig_after,
                "dw": dwmax,
                "pre_nnz": pre_nnz,
                "post_nnz": post_nnz,
                "pid": pid,
            }
        )

    if _call_count <= 3:
        print(
            f"  RSTDP #{_call_count}: edges={e} DA={da_val:.4f} eff_DA={effective_da:.4f} elig={elig_before:.6f}->{elig_after:.6f} dw={dwmax:.2e} pre={pre_nnz} post={post_nnz}",
            flush=True,
        )

    return new_state, result


rstdp_mod.RStdpEligibilityRule.step = _patched_step

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def main() -> None:
    gws = 5e-6
    w_max = gws * 3.0
    lr = w_max * 1.0  # 100x boost

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
            "skip_fan_in": 5,
            "in_to_hidden_fan_in": 2,
        },
        "learning": {
            "enabled": True,
            "rule": "rstdp_elig",
            "lr": lr,
            "w_min": 0.0,
            "w_max": w_max,
            "a_plus": 1.0,
            "a_minus": 0.6,
            "tau_e": 0.020,
            "tau_pre": 0.020,
            "tau_post": 0.020,
            "weight_decay": 0.0,
            "dopamine_scale": 1.0,
            "baseline": 0.5,
            "synaptic_scaling": False,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "amount": 1.0,
            "field_type": "global_scalar",
            "decay_tau": 0.01,
        },
        "wrapper": {
            "enabled": True,
            "spike_window": 30,
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
                "epsilon_start": 0.20,
                "epsilon_end": 0.01,
                "epsilon_decay_trials": 400,
            },
            "action_force": {
                "enabled": True,
                "mode": "supervised",
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
        steps=20,
        dt=1e-3,
        sim_steps_per_trial=30,
        device="cuda",
        learning_mode="rstdp",
        engine_learning_rule="rstdp_elig",
        inter_trial_reset=True,
        drive_scale=1e-9,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=10,
        reward_delivery_clamp_input=True,
        debug=False,
        dump_last_trials_csv=False,
    )

    r = run_logic_gate_engine(cfg, spec)
    print("\n=== RESULTS ===")
    print(f"eval={r['eval_accuracy']:.4f} preds={r['preds']}")
    print(f"Total RSTDP calls: {_call_count}")
    print(f"\n=== DA+ELIG CO-OCCURRENCES ({len(_cooccurrence)} events) ===")
    for i, evt in enumerate(_cooccurrence[:20]):
        print(
            f"  #{evt['call']:5d}: DA={evt['DA']:.4f} eff={evt['eff_DA']:.4f} "
            f"elig={evt['elig_before']:.6f} dw={evt['dw']:.2e} pre={evt['pre_nnz']} post={evt['post_nnz']} pid={evt['pid']}"
        )
    if len(_cooccurrence) > 20:
        print(f"  ... ({len(_cooccurrence) - 20} more events)")

    # Show dw distribution
    dw_values = [evt["dw"] for evt in _cooccurrence]
    nonzero_dw = [d for d in dw_values if d > 0]
    print("\n=== DW STATS ===")
    print(f"Events with eff_DA > 0 & elig > 0: {len(_cooccurrence)}")
    print(f"Events with dw > 0: {len(nonzero_dw)}")
    if nonzero_dw:
        print(f"dw range: {min(nonzero_dw):.2e} to {max(nonzero_dw):.2e}")
        print(f"dw mean: {sum(nonzero_dw) / len(nonzero_dw):.2e}")

    print("\n=== WEIGHT CHANGES BY PROJECTION ===")
    for pid, info in sorted(_weight_snapshots.items()):
        delta = info["latest_mean"] - info["initial_mean"]
        print(
            f"  edges={pid}: initial={info['initial_mean']:.6e} final={info['latest_mean']:.6e} delta={delta:.2e} calls={info['calls']}"
        )


if __name__ == "__main__":
    main()
