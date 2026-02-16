#!/usr/bin/env python
"""Debug wrapper configuration."""

import sys

sys.path.insert(0, "src")

from biosnn.tasks.logic_gates.engine_runner import _resolve_wrapper_cfg

# Test case 1: learning enabled, modulators enabled
print("=" * 60)
print("TEST 1: learning=True, modulators=True")
print("=" * 60)
run_spec = {"learning": {"enabled": True}, "modulators": {"enabled": True}, "wrapper": {}}
result = _resolve_wrapper_cfg(run_spec)
print(f"Result: wrapper.enabled={result.get('enabled')}")
print()

# Test case 2: learning disabled, modulators enabled
print("=" * 60)
print("TEST 2: learning=False, modulators=True")
print("=" * 60)
run_spec = {"learning": {"enabled": False}, "modulators": {"enabled": True}, "wrapper": {}}
result = _resolve_wrapper_cfg(run_spec)
print(f"Result: wrapper.enabled={result.get('enabled')}")
print()

# Test case 3: learning enabled, modulators disabled
print("=" * 60)
print("TEST 3: learning=True, modulators=False")
print("=" * 60)
run_spec = {"learning": {"enabled": True}, "modulators": {"enabled": False}, "wrapper": {}}
result = _resolve_wrapper_cfg(run_spec)
print(f"Result: wrapper.enabled={result.get('enabled')}")
print()

# Test case 4: both disabled
print("=" * 60)
print("TEST 4: learning=False, modulators=False")
print("=" * 60)
run_spec = {"learning": {"enabled": False}, "modulators": {"enabled": False}, "wrapper": {}}
result = _resolve_wrapper_cfg(run_spec)
print(f"Result: wrapper.enabled={result.get('enabled')}")
