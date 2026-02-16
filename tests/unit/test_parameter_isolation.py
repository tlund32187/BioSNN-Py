#!/usr/bin/env python3
"""Systematically test which parameter change breaks learning."""

import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def run_short_curriculum_trial(config_dict):
    """Run 10 curriculum trials and return mean_abs_dw."""
    from biosnn.tasks.logic_gates.engine_runner import run_logic_gate_engine
    from biosnn.tasks.logic_gates.runner import LogicGateRunConfig

    # Create a minimal config for a short test run
    cfg = LogicGateRunConfig(
        gate="xnor",
        device="cpu",
        steps=10,  # Just 10 trials
        seed=123,
    )

    try:
        result = run_logic_gate_engine(cfg, config_dict)
        mean_abs_dw = result.get("mean_abs_dw", 0.0)
        return mean_abs_dw
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def test_parameter_isolation():
    """Test each parameter change independently."""
    print("=" * 80)
    print("PARAMETER ISOLATION TEST")
    print("=" * 80)

    # Baseline (working) configuration
    baseline: dict[str, Any] = {
        "synapse": {"receptor_mode": "exc_only"},
        "wrapper": {
            "ach_lr_gain": 0.4,
            "ne_lr_gain": 0.25,
            "ht_extra_weight_decay": 0.02,
        },
        "logic": {"reward_delivery_steps": 3},
    }

    configs_to_test: list[tuple[str, dict[str, Any]]] = [
        ("BASELINE (working)", baseline),
        (
            "Change receptor_mode only",
            {
                **baseline,
                "synapse": {"receptor_mode": "ei_ampa_nmda_gabaa_gabab"},
            },
        ),
        (
            "Change ach_lr_gain only",
            {
                **baseline,
                "wrapper": {**baseline["wrapper"], "ach_lr_gain": 0.5},
            },
        ),
        (
            "Change ne_lr_gain only",
            {
                **baseline,
                "wrapper": {**baseline["wrapper"], "ne_lr_gain": 0.3},
            },
        ),
        (
            "Change ht_decay only",
            {
                **baseline,
                "wrapper": {**baseline["wrapper"], "ht_extra_weight_decay": 0.03},
            },
        ),
        (
            "Change reward_steps only",
            {
                **baseline,
                "logic": {"reward_delivery_steps": 5},
            },
        ),
        (
            "All changes together (BROKEN)",
            {
                "synapse": {"receptor_mode": "ei_ampa_nmda_gabaa_gabab"},
                "wrapper": {
                    "ach_lr_gain": 0.5,
                    "ne_lr_gain": 0.3,
                    "ht_extra_weight_decay": 0.03,
                },
                "logic": {"reward_delivery_steps": 5},
            },
        ),
    ]

    print("\nRunning 10-trial tests for each configuration:")
    print("(This will take a minute per test)\n")

    results = []
    for test_name, config_dict in configs_to_test:
        print(f"Testing: {test_name}...", end=" ", flush=True)
        mean_abs_dw = run_short_curriculum_trial(config_dict)
        results.append((test_name, mean_abs_dw))
        if mean_abs_dw is not None:
            print(f"mean_abs_dw = {mean_abs_dw:.6f}")
        else:
            print("FAILED")

    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    baseline_dw = results[0][1]
    print(f"\nBaseline mean_abs_dw: {baseline_dw:.6f}")

    print("\nParameter changes:")
    for test_name, dw in results[1:]:
        if dw is not None:
            pct_change = ((dw - baseline_dw) / baseline_dw * 100) if baseline_dw != 0 else 0
            status = "✅ OK" if dw > 0.0001 else "❌ BROKEN"
            print(f"  {test_name}: {dw:.6f} ({pct_change:+.1f}%) {status}")
        else:
            print(f"  {test_name}: FAILED TO RUN")

    # Identify culprit
    print("\n" + "-" * 80)
    print("ANALYSIS:")
    print("-" * 80)

    broken_tests = [
        (name, dw)
        for name, dw in results[1:-1]  # Exclude baseline and all-changes
        if dw is not None and dw <= 0.0001
    ]

    if broken_tests:
        print("\n⚠️  FOUND BREAKING PARAMETER(S):")
        for name, _dw in broken_tests:
            print(f"  - {name}")
    else:
        print("\n⚠️  No single parameter breaks learning!")
        print("   It might be a combination of changes.")
        print(f"   All changes together: {results[-1][1]:.6f}")


if __name__ == "__main__":
    test_parameter_isolation()
