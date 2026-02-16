#!/usr/bin/env python3
"""Check learning rates for curriculum vs single OR."""

from biosnn.experiments.demo_registry import default_run_spec, resolve_run_spec

# Check OR curriculum learning rate
curriculum_spec = default_run_spec(demo_id="logic_curriculum")
resolved = resolve_run_spec(curriculum_spec)
print("Logic Curriculum Learning Rule:")
print(f"  enabled: {resolved['learning']['enabled']}")
print(f"  rule: {resolved['learning']['rule']}")
print(f"  lr: {resolved['learning']['lr']}")

# Compare to single OR
or_spec = default_run_spec(demo_id="logic_or")
resolved_or = resolve_run_spec(or_spec)
print("\nLogic OR Single Gate Learning Rule:")
print(f"  enabled: {resolved_or['learning']['enabled']}")
print(f"  rule: {resolved_or['learning']['rule']}")
print(f"  lr: {resolved_or['learning']['lr']}")

if resolved["learning"]["lr"] > 0:
    ratio = resolved_or["learning"]["lr"] / resolved["learning"]["lr"]
    print(f"\nRatio: OR_lr / Curriculum_lr = {ratio:.0f}x")
