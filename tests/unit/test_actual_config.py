#!/usr/bin/env python
"""Test wrapper resolution with actual problem run config."""

import json
import sys

sys.path.insert(0, "src")

from biosnn.tasks.logic_gates.engine_runner import _resolve_wrapper_cfg

# Load the actual run config
with open("artifacts/run_20260212_121859_012400/run_config.json") as f:
    full_config = json.load(f)

# Extract the relevant parts
run_spec = {
    "learning": full_config.get("learning"),
    "modulators": full_config.get("modulators"),
    "wrapper": full_config.get("wrapper"),
}

print("Input run_spec:")
print(f"  learning.enabled: {run_spec['learning'].get('enabled')}")
print(f"  modulators.enabled: {run_spec['modulators'].get('enabled')}")
print(f"  wrapper.enabled: {run_spec['wrapper'].get('enabled')}")
print()

# Resolve
result = _resolve_wrapper_cfg(run_spec)

print("Output from _resolve_wrapper_cfg:")
print(f"  wrapper.enabled: {result.get('enabled')}")
