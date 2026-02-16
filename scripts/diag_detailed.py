"""Detailed per-step diagnostic: match exact learning config."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig, run_logic_gate_engine
from biosnn.tasks.logic_gates import engine_runner as er

gws = 5e-7
w_max = gws * 3.0
sim_steps = 10

cfg = LogicGateRunConfig(
    gate=LogicGate.OR,
    seed=42,
    steps=1,
    dt=1e-3,
    sim_steps_per_trial=sim_steps,
    device="cuda",
    learning_mode="rstdp",
    engine_learning_rule="rstdp_elig",
    inter_trial_reset=True,
    drive_scale=1.0,
    curriculum_gate_context={"enabled": True, "amplitude": 0.30},
    reward_delivery_steps=5,
    reward_delivery_clamp_input=True,
    debug=True,
)
spec = {
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
        "excit_target_compartment": "soma",
    },
    "learning": {
        "enabled": True,
        "rule": "rstdp_eligibility",
        "lr": w_max * 0.08,
        "w_min": -w_max,
        "w_max": w_max,
        "tau_e": 0.01,
        "tau_pre": 0.020,
        "tau_post": 0.020,
        "a_plus": 1.0,
        "a_minus": 0.0,
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
        "decay_tau": 0.05,
    },
    "wrapper": {"enabled": True, "spike_window": sim_steps, "decision_mode": "spike_count"},
    "homeostasis": {"enabled": True, "alpha": 0.01, "eta": 1e-3, "r_target": 0.05},
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
            "steps": 5,
            "compartment": "soma",
            "suppression_factor": -0.25,
        },
    },
}

orig = er._run_trial_steps
call_count = [0]


def patched(**kw):
    call_count[0] += 1
    eb = kw["engine_build"]
    eng = eb.engine
    eb.current_input_drive["tensor"] = kw["input_drive"]
    is_main = call_count[0] == 1
    label = "MAIN" if is_main else "REWARD"
    n_steps = kw["sim_steps_per_trial"]
    print(f"\n  === {label} ({n_steps} steps) ===")
    print(f"  Input: {kw['input_drive'].cpu().tolist()}")
    af = kw.get("action_force_steps", 0)
    print(f"  Action force steps: {af}")
    # Check action_drive_by_population
    ad = eb.action_drive_by_population
    if ad:
        for pn, drives in ad.items():
            for comp, val in drives.items():
                print(f"  Action drive {pn}/{comp}: {val.cpu().tolist()}")
    else:
        print("  No action drive")

    out_cnt = torch.zeros(2, device="cuda", dtype=torch.float32)
    hid_cnt = torch.zeros(16, device="cuda", dtype=torch.float32)
    mod_state = eb.modulator_state
    if mod_state:
        input_active = bool(kw["input_drive"].any())
        mod_state["input_active"] = input_active
        mod_state["ach_pulse_emitted"] = False
        mod_state["trial_steps"] = max(1, n_steps)

    for sn in range(n_steps):
        if mod_state:
            mod_state["trial_step_idx"] = sn
        if af > 0 and sn >= af:
            er._clear_action_force_drive(engine_build=eb)
        eng.step()
        for pn, ps in eng._pop_states.items():
            spk = ps.spikes
            v = getattr(ps, "v_soma", None)
            if v is None:
                state = getattr(ps, "state", None)
                if state:
                    v = getattr(state, "v_soma", None)
            n_spk = int(spk.sum().item())
            if n_spk > 0 or pn in ("Hidden", "Out"):
                v_str = ""
                if v is not None:
                    v_str = f"  v={[round(x, 4) for x in v.cpu().tolist()[:4]]}"
                print(f"    step {sn}: {pn}: spikes={n_spk}{v_str}")
        out_cnt.add_(eng._pop_states[eb.output_population].spikes.float())
        for hp in eb.hidden_populations:
            hid_cnt.add_(eng._pop_states[hp].spikes.float())

    if af > 0:
        er._clear_action_force_drive(engine_build=eb)
    if mod_state:
        mod_state["input_active"] = False
        mod_state["trial_step_idx"] = -1
    print(f"  out_cnt: {out_cnt.cpu().tolist()}, hid_cnt_sum: {hid_cnt.sum().item()}")
    return out_cnt, hid_cnt, kw.get("step_offset", 0) + n_steps


er._run_trial_steps = patched
r = run_logic_gate_engine(cfg, spec)
print(f"\nFinal: eval={r['eval_accuracy']:.2f}  preds={r['preds']}")
