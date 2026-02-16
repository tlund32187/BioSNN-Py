#!/usr/bin/env python3
"""Debug script to instrument curriculum to trace dopamine releases."""

import sys
from pathlib import Path
from typing import Any

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_curriculum_with_debug() -> None:
    """Run a minimal curriculum with debug output for dopamine."""
    print("=" * 80)
    print("TEST: Curriculum with dopamine tracing")
    print("=" * 80)

    # Use working parameters
    run_spec: dict[str, Any] = {
        "logic": {"reward_delivery_steps": 5},
        "steps": 10,  # Just 10 trials for debugging
        "synapse": {"receptor_mode": "exc_only"},
        "wrapper": {
            "ach_lr_gain": 0.4,
            "ne_lr_gain": 0.25,
            "ht_extra_weight_decay": 0.02,
        },
    }

    print("\nRun spec:")
    print(f"  steps: {run_spec['steps']}")
    print(f"  receptor_mode: {run_spec['synapse']['receptor_mode']}")
    print(f"  reward_delivery_steps: {run_spec['logic']['reward_delivery_steps']}")
    print(
        f"  wrapper gains: ach={run_spec['wrapper']['ach_lr_gain']}, ne={run_spec['wrapper']['ne_lr_gain']}"
    )

    print("\n⏳ This would require importing and running the full curriculum...")
    print("The curriculum infrastructure is large; debugging via instrumentation")
    print("would require modifying engine_runner.py to add debug print statements.")


if __name__ == "__main__":
    test_curriculum_with_debug()
