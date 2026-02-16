#!/usr/bin/env python3
"""Compare learning between working and broken configurations."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def compare_configurations():
    """Compare two run configs side-by-side."""
    print("=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)

    # Working config
    working_config_path = (
        project_root / "artifacts/run_20260212_132912_619800/run_config.json"
    )
    # Broken config
    broken_config_path = (
        project_root / "artifacts/run_20260212_140714_134900/run_config.json"
    )

    with open(working_config_path) as f:
        working = json.load(f)
    with open(broken_config_path) as f:
        broken = json.load(f)

    # Find differences
    print("\nDifferences between working and broken configs:")
    print("\nWORKING configuration:")
    print(f"  receptor_mode: {working['synapse']['receptor_mode']}")
    print(f"  wrapper.ach_lr_gain: {working['wrapper']['ach_lr_gain']}")
    print(f"  wrapper.ne_lr_gain: {working['wrapper']['ne_lr_gain']}")
    print(f"  wrapper.ht_extra_weight_decay: {working['wrapper']['ht_extra_weight_decay']}")
    print(f"  reward_delivery_steps: {working['logic']['reward_delivery_steps']}")

    print("\nBROKEN configuration:")
    print(f"  receptor_mode: {broken['synapse']['receptor_mode']}")
    print(f"  wrapper.ach_lr_gain: {broken['wrapper']['ach_lr_gain']}")
    print(f"  wrapper.ne_lr_gain: {broken['wrapper']['ne_lr_gain']}")
    print(f"  wrapper.ht_extra_weight_decay: {broken['wrapper']['ht_extra_weight_decay']}")
    print(f"  reward_delivery_steps: {broken['logic']['reward_delivery_steps']}")

    print("\n" + "-" * 80)
    print("KEY DIFFERENCES:")
    print("-" * 80)

    diffs = {
        "receptor_mode": (
            working["synapse"]["receptor_mode"],
            broken["synapse"]["receptor_mode"],
        ),
        "ach_lr_gain": (
            working["wrapper"]["ach_lr_gain"],
            broken["wrapper"]["ach_lr_gain"],
        ),
        "ne_lr_gain": (
            working["wrapper"]["ne_lr_gain"],
            broken["wrapper"]["ne_lr_gain"],
        ),
        "ht_decay": (
            working["wrapper"]["ht_extra_weight_decay"],
            broken["wrapper"]["ht_extra_weight_decay"],
        ),
        "reward_steps": (
            working["logic"]["reward_delivery_steps"],
            broken["logic"]["reward_delivery_steps"],
        ),
    }

    for name, (w_val, b_val) in diffs.items():
        if w_val != b_val:
            print(f"  {name}: {w_val} → {b_val}")
        else:
            print(f"  {name}: {w_val} (same)")

    # Analysis
    print("\n" + "-" * 80)
    print("HYPOTHESIS:")
    print("-" * 80)
    print("""
The broken run changed multiple parameters simultaneously:
1. receptor_mode "exc_only" → "ei_ampa_nmda_gabaa_gabab" (adds GABA inhibition)
2. ach_lr_gain 0.4 → 0.5 (LR modulation gain +25%)
3. ne_lr_gain 0.25 → 0.3 (LR modulation gain +20%)
4. ht_decay 0.02 → 0.03 (homeostatic decay +50%)
5. reward_steps 3 → 5 (reward window +67%)

Each of these alone might be fine, but combined they might cause:
- Instability in learning dynamics
- Excessive LR modulation reducing effective learning
- GABA inhibition suppressing activity leading to zero gradients
- Or an actual bug in modulation with complex receptors
""")

if __name__ == "__main__":
    compare_configurations()
