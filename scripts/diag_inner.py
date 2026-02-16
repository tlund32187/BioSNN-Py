"""Diagnostic: trace receptor profile internals using source-level debug."""

from __future__ import annotations

import os
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
    _apply_gate_context_drive(input_drive, gate="or")
    engine_build.current_input_drive["tensor"] = input_drive

    for step in range(20):
        os.environ["_DIAG_STEP"] = str(step)
        engine.step()
        # Brief summary
        in_spk = int(engine._pop_states["In"].spikes.sum().item())
        if in_spk > 0:
            print(f"step={step}: in_spk={in_spk}")

    del os.environ["_DIAG_STEP"]


if __name__ == "__main__":
    main()
