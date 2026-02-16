"""Minimal debug: verify supervised mode reaches engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosnn.tasks.logic_gates.engine_runner import _resolve_action_force_cfg

spec: dict[str, Any] = {
    "logic": {
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

cfg = _resolve_action_force_cfg(
    spec,
    default_enabled=True,
    default_window="reward_window",
    default_steps=1,
    default_amplitude=0.75,
    default_compartment="soma",
)

print("Resolved action_force_cfg:", cfg)
print("Mode:", repr(cfg["mode"]))
assert cfg["mode"] == "supervised", f"Mode is {cfg['mode']!r}, expected 'supervised'"
print("OK: mode is 'supervised'")
