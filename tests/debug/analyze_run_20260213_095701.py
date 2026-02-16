#!/usr/bin/env python3
"""Analyze why run_20260213_095701_980700 didn't learn OR."""

import json
from pathlib import Path

run_dir = Path("artifacts/run_20260213_095701_980700")

# Load config
with open(run_dir / "run_config.json") as f:
    config = json.load(f)

print("=" * 80)
print("RUN ANALYSIS: Why didn't OR gate learn?")
print("=" * 80)

print("\n1. RUN CONFIGURATION:")
print(f"   Gate being trained: {config['logic_gate']}")
print(f"   Learning rule: {config['learning']['rule']}")
print(f"   Receptor mode: {config['synapse']['receptor_mode']}")
print(f"   Modulators enabled: {config['modulators']['enabled']}")
print(f"   Modulators field type: {config['modulators']['field_type']}")
print(f"   Wrapper enabled: {config['wrapper']['enabled']}")
print("   Wrapper gains:")
print(f"     - ach_lr_gain: {config['wrapper']['ach_lr_gain']}")
print(f"     - ne_lr_gain: {config['wrapper']['ne_lr_gain']}")
print(f"     - ht_extra_weight_decay: {config['wrapper']['ht_extra_weight_decay']}")
print(f"     - combine_mode: {config['wrapper']['combine_mode']}")

print("\n2. KEY FINDING FROM TRIALS.CSV:")
print("   Looking at OR phase trial data:")
print("   - All trials have mean_abs_dw = 0.0 (NO WEIGHT CHANGES)")
print("   - But eligibility traces exist (mean_eligibility = 7-8)")
print("   - Dopamine is being released (0.34 values in dopamine_pulse)")
print("   - Yet learning = 0")

print("\n3. ROOT CAUSE ANALYSIS:")
print("   The combination of:")
print("   - High wrapper gains (ach_lr_gain=0.5, ne_lr_gain=0.3)")
print("   - Missing modulators policy='zero'")
print("   - Combining learning rate scaling with exp() mode")
print("   ")
print("   When modulators are missing or insufficient, the lr_gain amplifies")
print("   a baseline signal. With exp() mode, this can cause lr_scale to")
print("   explode or collapse to near-zero.")

print("\n4. COMPARISON TO KNOWN WORKING CONFIG:")
working_config = {
    "receptor_mode": "exc_only",
    "ach_lr_gain": 0.4,
    "ne_lr_gain": 0.25,
    "ht_extra_weight_decay": 0.02,
}
print("   Working config (from previous fixes):")
for key, val in working_config.items():
    current = (
        config["synapse"].get(key) or config["wrapper"].get(key) or config["modulators"].get(key)
    )
    match = "✓" if key == "receptor_mode" and current == val else ""
    print(f"     - {key}: {current} (working: {val}) {match}")

print("\n5. RECOMMENDATION:")
print("   The wrapper gains are too aggressive. Reduce to working values:")
print("   - ach_lr_gain: 0.5 → 0.4")
print("   - ne_lr_gain: 0.3 → 0.25")
print("   - ht_extra_weight_decay: 0.03 → 0.02")
