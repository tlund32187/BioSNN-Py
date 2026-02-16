"""Diagnostic: Correct weight tracking (unwrapping ModulatedRuleWrapperState)
+ test with proper learning rates (lr_mult ≈ 0.001–0.01, not 1–100)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.simulation.engine.torch_network_engine import TorchNetworkEngine
from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def _get_inner_state(ls: Any) -> Any:
    """Unwrap ModulatedRuleWrapperState → RStdpEligibilityState."""
    if hasattr(ls, "inner_state"):
        return ls.inner_state
    return ls


def run_diag(
    *,
    seed: int = 42,
    gws: float = 5e-7,
    af_mode: str = "always",
    trials: int = 100,
    sim_steps: int = 30,
    reward_steps: int = 10,
    lr_mult: float = 0.01,
    tau_e: float = 0.050,
    da_decay: float = 0.05,
    excit_target: str = "soma",
    weight_decay: float = 0.0,
    label: str = "",
) -> dict[str, Any]:
    w_max = gws * 3.0
    lr = w_max * lr_mult

    print(f"\n{'=' * 70}")
    print(f"Config: {label}")
    print(f"  seed={seed} gws={gws:.0e} mode={af_mode} tgt={excit_target}")
    print(f"  lr={lr:.2e} (lr_mult={lr_mult}) w_max={w_max:.2e} tau_e={tau_e} da_decay={da_decay}")

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

    weight_history: list[dict[str, Any]] = []
    trial_counter = [0]
    orig_reset = TorchNetworkEngine.reset_inter_trial_state

    def tracked_reset(self_eng: Any) -> None:
        orig_reset(self_eng)
        trial_counter[0] += 1
        snap: dict[str, Any] = {"trial": trial_counter[0]}
        for runtime in getattr(self_eng, "_proj_runtime_list", []):
            if runtime.learning is not None:
                pname = runtime.name
                w = getattr(runtime.state.state, "weights", None)
                if w is None:
                    continue
                snap[pname] = {
                    "mean": float(w.mean().cpu()),
                    "std": float(w.std().cpu()),
                    "min": float(w.min().cpu()),
                    "max": float(w.max().cpu()),
                    "n": int(w.numel()),
                    "vals": w.cpu().tolist() if w.numel() <= 20 else None,
                }
                ls = runtime.state.learning_state
                inner = _get_inner_state(ls) if ls is not None else None
                if inner is not None:
                    if hasattr(inner, "eligibility"):
                        snap[pname + "_elig"] = float(inner.eligibility.abs().mean().cpu())
                    if hasattr(inner, "last_mean_abs_dw"):
                        snap[pname + "_dw"] = float(inner.last_mean_abs_dw.cpu())
                    if hasattr(inner, "pre_trace"):
                        snap[pname + "_pre_tr"] = (
                            float(inner.pre_trace.abs().mean().cpu())
                            if inner.pre_trace is not None
                            else 0.0
                        )
                    if hasattr(inner, "post_trace"):
                        snap[pname + "_post_tr"] = (
                            float(inner.post_trace.abs().mean().cpu())
                            if inner.post_trace is not None
                            else 0.0
                        )
        weight_history.append(snap)

    TorchNetworkEngine.reset_inter_trial_state = tracked_reset  # type: ignore[assignment]

    try:
        r = run_logic_gate_engine(cfg, spec)
    finally:
        TorchNetworkEngine.reset_inter_trial_state = orig_reset  # type: ignore[assignment]

    print(f"\n  RESULT: eval={r['eval_accuracy']:.2f}  preds={r['preds']}")

    if weight_history:
        first = weight_history[0]
        proj_names = [k for k in first if isinstance(first[k], dict) and "mean" in first[k]]

        for pname in proj_names:
            print(f"\n  [{pname}] (n={first[pname]['n']})")
            for snap in weight_history:
                if pname not in snap:
                    continue
                ws = snap[pname]
                t = snap["trial"]
                elig = snap.get(pname + "_elig", -1)
                dw = snap.get(pname + "_dw", -1)
                pre_tr = snap.get(pname + "_pre_tr", -1)
                post_tr = snap.get(pname + "_post_tr", -1)
                show = t <= 3 or t % 20 == 0 or t == len(weight_history)
                if not show:
                    continue
                line = f"    t={t:4d}  mean={ws['mean']:+.3e} [{ws['min']:+.2e},{ws['max']:+.2e}]"
                if elig >= 0:
                    line += f"  elig={elig:.2e}"
                if dw >= 0:
                    line += f"  dw={dw:.2e}"
                if pre_tr >= 0:
                    line += f"  pre={pre_tr:.2e}"
                if post_tr >= 0:
                    line += f"  post={post_tr:.2e}"
                if ws.get("vals") is not None and ws["n"] <= 8:
                    line += f"  w={[f'{v:.2e}' for v in ws['vals']]}"
                print(line)

            f_w = weight_history[0].get(pname, {})
            l_w = weight_history[-1].get(pname, {})
            if f_w and l_w:
                dm = l_w["mean"] - f_w["mean"]
                print(f"    DELTA mean={dm:+.3e}")

    sys.stdout.flush()
    return r


def main() -> None:
    # Phase 1: Calibrate lr_mult — find the range where weights change gradually
    print("Phase 1: lr_mult calibration (20 trials, always mode, soma)")
    for lr_m in [0.001, 0.005, 0.01, 0.05, 0.1]:
        run_diag(lr_mult=lr_m, trials=20, label=f"lr_mult={lr_m}")

    # Phase 2: Best lr_mult with longer training
    print("\n\nPhase 2: Extended training with calibrated lr")
    for lr_m in [0.005, 0.01]:
        run_diag(lr_mult=lr_m, trials=400, label=f"400 trials, lr_mult={lr_m}")

    # Phase 3: Different seeds
    print("\n\nPhase 3: Seed sweep with best lr")
    for seed in [42, 7, 314, 500]:
        run_diag(seed=seed, lr_mult=0.01, trials=400, label=f"seed={seed}")

    # Phase 4: Larger DA with small lr
    print("\n\nPhase 4: Larger DA decay (longer signal)")
    for da_d in [0.01, 0.05, 0.20, 0.50]:
        run_diag(lr_mult=0.01, da_decay=da_d, trials=400, label=f"da_decay={da_d}")


if __name__ == "__main__":
    main()
