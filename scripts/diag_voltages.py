"""Minimal diagnostic: check membrane voltages and spikes at each timestep."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine


def main():
    """Run 2 trials, print per-step state of hidden & output neurons."""
    import torch
    gws = 5e-7
    w_max = gws * 3.0
    spec = {
        "dtype": "float32", "delay_steps": 3,
        "synapse": {
            "backend": "spmm_fused", "fused_layout": "auto",
            "ring_strategy": "dense", "ring_dtype": "none",
            "receptor_mode": "exc_only", "global_weight_scale": gws,
            "skip_fan_in": 2, "in_to_hidden_fan_in": 2,
            "excit_target_compartment": "soma",
        },
        "learning": {"enabled": False},
        "modulators": {"enabled": False},
        "wrapper": {"enabled": False},
        "homeostasis": {"enabled": False},
        "logic": {
            "exploration": {"enabled": False, "tie_break": "random_among_max"},
            "action_force": {"enabled": False},
        },
    }
    cfg = LogicGateRunConfig(
        gate=LogicGate.OR, seed=42, steps=2, dt=1e-3,
        sim_steps_per_trial=5, device="cuda",
        learning_mode="rstdp", engine_learning_rule="rstdp_elig",
        inter_trial_reset=True, drive_scale=1.0,
        curriculum_gate_context={"enabled": False},
        reward_delivery_steps=0, reward_delivery_clamp_input=False,
        debug=True,
    )

    # Monkey-patch _run_trial_steps to print per-step voltages
    from biosnn.tasks.logic_gates import engine_runner as er

    _orig_run_trial_steps = er._run_trial_steps

    def _patched_run_trial_steps(**kwargs):
        engine_build = kwargs["engine_build"]
        engine = engine_build.engine
        sim_steps = kwargs["sim_steps_per_trial"]
        input_drive = kwargs["input_drive"]

        # Set input
        engine_build.current_input_drive["tensor"] = input_drive

        print(f"\n  === Trial step block ({sim_steps} steps) ===")
        print(f"  Input drive: {input_drive.cpu().tolist()}")

        # Check all population states before stepping
        for pop_name in engine._pop_states:
            ps = engine._pop_states[pop_name]
            v = getattr(ps, 'v_soma', None)
            if v is None:
                v = getattr(ps, 'v', None)
            spk = ps.spikes
            if v is not None:
                print(f"  PRE  {pop_name}: v_soma={v.cpu().tolist()[:4]}...  spikes={spk.cpu().tolist()[:4]}...")

        # Now call the original
        result = _orig_run_trial_steps(**kwargs)

        # Check after
        for pop_name in engine._pop_states:
            ps = engine._pop_states[pop_name]
            v = getattr(ps, 'v_soma', None)
            if v is None:
                v = getattr(ps, 'v', None)
            spk = ps.spikes
            if v is not None:
                print(f"  POST {pop_name}: v_soma={v.cpu().tolist()[:4]}...  spikes={spk.cpu().tolist()[:4]}...")

        out_counts, hidden_counts, step = result
        print(f"  out_counts: {out_counts.cpu().tolist()}, hidden_counts_sum: {hidden_counts.sum().item()}")
        return result

    er._run_trial_steps = _patched_run_trial_steps

    r = run_logic_gate_engine(cfg, spec)
    print(f"\nFinal: eval={r['eval_accuracy']:.2f}  preds={r['preds']}")

    # Also check the raw population state type
    engine = r.get("_engine_build")
    if engine is None:
        print("\nChecking population state attributes via engine_runner module...")
        # Just print what happened


if __name__ == "__main__":
    main()
