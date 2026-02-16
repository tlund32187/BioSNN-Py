"""Minimal per-step diagnostic for drive_scale verification."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from biosnn.contracts.neurons import Compartment
from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig
from biosnn.tasks.logic_gates.encoding import encode_inputs
from biosnn.tasks.logic_gates.engine_runner import (
    _apply_gate_context_drive,
    _build_engine,
    _resolve_device,
)


def main() -> None:
    drive_scale = 1e-9

    gws_val = 1e-5  # Calibrated for ~0.75 mV/step per synapse
    spec: dict[str, Any] = {
        "dtype": "float32",
        "delay_steps": 3,
        "synapse": {
            "backend": "spmm_fused",
            "fused_layout": "auto",
            "ring_strategy": "dense",
            "ring_dtype": "none",
            "receptor_mode": "exc_only",
            "global_weight_scale": gws_val,
        },
        "learning": {"enabled": False},
        "modulators": {"enabled": False},
        "wrapper": {"enabled": False},
        "homeostasis": {"enabled": False},
        "logic": {},
    }

    cfg = LogicGateRunConfig(
        gate=LogicGate.OR,
        seed=42,
        steps=10,
        dt=1e-3,
        sim_steps_per_trial=15,
        device="cuda",
        learning_mode="rstdp",
        inter_trial_reset=True,
        drive_scale=drive_scale,
        curriculum_gate_context={"enabled": True, "amplitude": 0.30},
        reward_delivery_steps=0,
        debug=False,
    )

    device = _resolve_device(torch, str(cfg.device))
    engine_build = _build_engine(
        config=cfg,
        gate=LogicGate.OR,
        run_spec=spec,
        device=device,
    )
    engine = engine_build.engine

    # Make input drive for case [1,1] (OR → target=1)
    inp = torch.tensor([1.0, 1.0], device=device)
    drive = encode_inputs(
        inp, mode="rate", dt=cfg.dt, high=1.0, low=0.0, compartment=Compartment.SOMA
    )
    input_drive = drive[Compartment.SOMA]
    _apply_gate_context_drive(input_drive, gate="or")

    print(f"drive_scale = {drive_scale:.0e}")
    print(f"Input drive tensor: {input_drive.cpu().tolist()}")
    print(f"Expected effective drive: {[v * drive_scale for v in input_drive.cpu().tolist()]}")
    print()

    # Set the input drive
    engine_build.current_input_drive["tensor"] = input_drive

    # Skip gate context setup for simplicity - focus on input→hidden→output chain

    print("=== Per-step neuron state ===")
    print(
        f"{'step':>4}  {'in_v1':>10}  {'in_spk':>6}  "
        f"{'hid_vs0':>10}  {'hid_vd0':>10}  {'hid_spk':>7}  "
        f"{'out_vs0':>10}  {'out_vd0':>10}  {'out_spk':>7}"
    )

    in_pop = engine_build.input_population
    out_pop = engine_build.output_population
    hid_pops = engine_build.hidden_populations

    for step in range(30):
        engine.step()

        # Get neuron states
        in_state = engine._pop_states[in_pop].state
        out_state = engine._pop_states[out_pop].state
        in_spikes = engine._pop_states[in_pop].spikes

        in_vs = in_state.v_soma.cpu().tolist()
        in_spk_count = int(in_spikes.sum().item())

        out_vs = out_state.v_soma.cpu().tolist()
        out_vd = out_state.v_dend.cpu().tolist()
        out_spikes = engine._pop_states[out_pop].spikes
        out_spk_count = int(out_spikes.sum().item())

        hid_spk_total = 0
        hid_vs0 = 0.0
        hid_vd0 = 0.0
        for hp in hid_pops:
            hs = engine._pop_states[hp]
            hid_spk_total += int(hs.spikes.sum().item())
            hid_vs0 = hs.state.v_soma[0].item()
            hid_vd0 = hs.state.v_dend[0].item()

        # Check drive buffer AND synapse state for In→Hidden
        hid_drive = ""
        for hp in hid_pops:
            drv = engine._drive_buffers.get(hp, {})
            for comp, buf in drv.items():
                vals = buf.cpu().tolist()
                nz = sum(1 for v in vals if abs(v) > 1e-20)
                mx = max(abs(v) for v in vals) if vals else 0
                hid_drive += f" {comp.name}:nz={nz},mx={mx:.2e}"
            break

        # Check In→Hidden synapse state
        syn_info = ""
        for rt in engine._proj_runtime_list:
            if rt.plan.name == "In->Hidden":
                ss = rt.state.state
                # Check receptor_g
                if ss.receptor_g is not None:
                    for comp, g in ss.receptor_g.items():
                        g_sum = g.abs().sum().item()
                        syn_info += f" g_{comp.name}={g_sum:.2e}"
                # Check ring buffer
                if ss.post_ring is not None:
                    for comp, ring in ss.post_ring.items():
                        r_sum = ring.abs().sum().item()
                        syn_info += f" ring_{comp.name}={r_sum:.2e}"
                # Check pre_activity_buf
                if ss.pre_activity_buf is not None:
                    pa = ss.pre_activity_buf
                    syn_info += f" pre_act={pa.abs().sum().item():.2e}"
                break

        print(
            f"{step:4d}  {in_vs[1]:10.6f}  {in_spk_count:6d}  "
            f"{hid_vs0:10.6f}  {hid_vd0:10.6f}  {hid_spk_total:7d}  "
            f"{out_vs[0]:10.6f}  {out_vd[0]:10.6f}  {out_spk_count:7d}"
            f"  {hid_drive}{syn_info}"
        )

    # Print projection topology details
    print("\n=== Projection details ===")
    for rt in engine._proj_runtime_list:
        plan = rt.plan
        w = rt.state.state.weights
        w_vals = w.to_dense() if w.is_sparse else w
        print(
            f"  {plan.name}: pre={plan.pre_name} post={plan.post_name} "
            f"target={plan.topology.target_compartment} "
            f"w_shape={tuple(w_vals.shape)} "
            f"w_mean={w_vals.mean().item():.4e} w_max={w_vals.max().item():.4e}"
        )

    print("\n=== Engine drive buffer keys ===")
    for k, v in engine._drive_buffers.items():
        print(f"  '{k}': comps={[c.name for c in v.keys()]}")
    print("\n=== Engine build hidden_populations ===")
    print(f"  {engine_build.hidden_populations}")
    print(f"  input_population: {engine_build.input_population}")
    print(f"  output_population: {engine_build.output_population}")
    print("\n=== Pop states ===")
    for k in engine._pop_states:
        ps = engine._pop_states[k]
        print(f"  '{k}': spikes_shape={ps.spikes.shape} v_soma_shape={ps.state.v_soma.shape}")

    # Also check: what is the drive_scale value captured in the engine?
    print(f"\ndrive_scale on config: {getattr(cfg, 'drive_scale', 'NOT SET')}")


if __name__ == "__main__":
    main()
