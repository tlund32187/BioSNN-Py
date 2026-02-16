"""Trace receptor profile signal path inside engine step."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

# Monkey-patch to trace the receptor profile path
import biosnn.synapses.dynamics.delayed_sparse_matmul as dsm
from biosnn.contracts.neurons import Compartment
from biosnn.tasks.logic_gates import LogicGate, LogicGateRunConfig
from biosnn.tasks.logic_gates.encoding import encode_inputs
from biosnn.tasks.logic_gates.engine_runner import _build_engine, _resolve_device

_orig_step = dsm._step_sparse_matmul_into


def _traced_step(
    model, state, topology, pre_spikes, device, weights, out_drive, dt, *, inputs_meta
):
    """Wrapper that prints intermediate values."""
    import os

    step_idx = int(os.environ.get("_DIAG_STEP", "-1"))

    # Call original
    _orig_step(
        model, state, topology, pre_spikes, device, weights, out_drive, dt, inputs_meta=inputs_meta
    )

    # Only print at specific steps
    if step_idx in (9, 10, 11, 12, 13, 14):
        proj_name = getattr(topology, "name", "?")
        if proj_name == "?":
            meta = getattr(topology, "meta", {}) or {}
            proj_name = meta.get("name", "?")

        # Check post_out (proj_drive when receptor enabled)
        if state.post_out:
            for comp, out in state.post_out.items():
                v = out.abs().sum().item()
                print(f"  [TRACE step={step_idx}] post_out[{comp.name}] sum={v:.4e}")

        # Check receptor_g
        if state.receptor_g:
            for comp, g in state.receptor_g.items():
                v = g.abs().sum().item()
                print(f"  [TRACE step={step_idx}] receptor_g[{comp.name}] sum={v:.4e}")

        # Check ring
        if state.post_ring:
            for comp, ring in state.post_ring.items():
                v = ring.abs().sum().item()
                print(f"  [TRACE step={step_idx}] ring[{comp.name}] sum={v:.4e}")

        # Check out_drive
        for comp, drv in out_drive.items():
            v = drv.abs().sum().item()
            print(f"  [TRACE step={step_idx}] out_drive[{comp.name}] sum={v:.4e}")


dsm._step_sparse_matmul_into = _traced_step

# Also trace the receptor profile itself
_orig_receptor = dsm._apply_receptor_profile_if_enabled


def _traced_receptor(
    *, model, state, topology, drive_by_comp, dt, n_post, device, weights_dtype, inputs_meta
):
    import os

    step_idx = int(os.environ.get("_DIAG_STEP", "-1"))

    if step_idx in (9, 10, 11, 12, 13, 14):
        for comp, out in drive_by_comp.items():
            v = out.abs().sum().item()
            print(f"    [RECEPTOR-IN step={step_idx}] drive_by_comp[{comp.name}] sum={v:.4e}")

    _orig_receptor(
        model=model,
        state=state,
        topology=topology,
        drive_by_comp=drive_by_comp,
        dt=dt,
        n_post=n_post,
        device=device,
        weights_dtype=weights_dtype,
        inputs_meta=inputs_meta,
    )

    if step_idx in (9, 10, 11, 12, 13, 14):
        for comp, out in drive_by_comp.items():
            v = out.abs().sum().item()
            print(f"    [RECEPTOR-OUT step={step_idx}] drive_by_comp[{comp.name}] sum={v:.4e}")
        if state.receptor_g:
            for comp, g in state.receptor_g.items():
                for i in range(g.shape[0]):
                    v = g[i].abs().sum().item()
                    print(f"    [RECEPTOR step={step_idx}] g[{comp.name}][{i}] sum={v:.4e}")


dsm._apply_receptor_profile_if_enabled = _traced_receptor


def main() -> None:
    import os

    drive_scale = 1e-9
    gws_val = 1e-6

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
    engine_build = _build_engine(config=cfg, gate=LogicGate.OR, run_spec=spec, device=device)
    engine = engine_build.engine

    inp = torch.tensor([1.0, 1.0], device=device)
    drive = encode_inputs(
        inp, mode="rate", dt=cfg.dt, high=1.0, low=0.0, compartment=Compartment.SOMA
    )
    input_drive = drive[Compartment.SOMA]
    from biosnn.tasks.logic_gates.engine_runner import _apply_gate_context_drive

    _apply_gate_context_drive(input_drive, gate="or")
    engine_build.current_input_drive["tensor"] = input_drive

    print("=== Running steps and tracing receptor profile ===")
    for step in range(20):
        os.environ["_DIAG_STEP"] = str(step)
        engine.step()

        # Read key state
        in_spk = int(engine._pop_states["In"].spikes.sum().item())
        hid_spk = sum(
            int(engine._pop_states[hp].spikes.sum().item())
            for hp in engine_build.hidden_populations
        )

        # Drive buffer for Hidden
        hid_drv = engine._drive_buffers.get(
            engine_build.hidden_populations[0] if engine_build.hidden_populations else "Hidden", {}
        )
        drv_info = ""
        for comp, buf in hid_drv.items():
            nz = int((buf.abs() > 1e-20).sum().item())
            drv_info += f" {comp.name}:nz={nz}"

        # Receptor g for In->Hidden
        g_info = ""
        for rt in engine._proj_runtime_list:
            if rt.plan.name == "In->Hidden":
                ss = rt.state.state
                if ss.receptor_g:
                    for comp, g in ss.receptor_g.items():
                        g_info += f" g_{comp.name}={g.abs().sum().item():.4e}"
                break

        if step in range(8, 16) or in_spk > 0 or hid_spk > 0:
            print(f"step={step:2d} in_spk={in_spk} hid_spk={hid_spk}{drv_info}{g_info}")

    del os.environ["_DIAG_STEP"]


if __name__ == "__main__":
    main()
