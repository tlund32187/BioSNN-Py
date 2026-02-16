#!/usr/bin/env python
"""Test engine build with wrapper logging."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

from biosnn.tasks.logic_gates.configs import LogicGateRunConfig
from biosnn.tasks.logic_gates.engine_runner import run_logic_gate_curriculum_engine

# Create a minimal config
config = LogicGateRunConfig(
    steps=10,  # Very short run
    seed=123,
    device="cpu",
    gate="or",
)

run_spec = {
    "learning": {"enabled": True, "rule": "rstdp", "lr": 0.001},
    "modulators": {"enabled": True, "kinds": ["dopamine"], "amount": 0.3},
    # Don't specify wrapper - let it auto-enable
}

# Run and capture output
with tempfile.TemporaryDirectory() as tmpdir:
    config.out_dir = Path(tmpdir)
    config.artifacts_root = Path(tmpdir)
    print("Starting engine with debug logging...", file=sys.stderr, flush=True)
    result = run_logic_gate_curriculum_engine(config, run_spec)
    print("Engine completed.", file=sys.stderr, flush=True)
