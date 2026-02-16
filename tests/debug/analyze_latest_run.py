#!/usr/bin/env python3
"""Analyze the latest curriculum run to diagnose OR gate learning issues."""

import csv
from pathlib import Path
from statistics import mean, stdev


def load_csv(filepath):
    """Load CSV file and return list of dicts."""
    with open(filepath) as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_run(run_dir="artifacts/tmp_curriculum_probe_ctx80"):
    """Analyze the run directory."""
    run_path = Path(run_dir)

    # Load data
    phases = load_csv(run_path / "phase_summary.csv")
    trials = load_csv(run_path / "trials.csv")

    print("=" * 80)
    print("ANALYSIS: Latest Curriculum Run")
    print("=" * 80)

    # 1. Phase Summary
    print("\n1. PHASE SUMMARY")
    print("-" * 80)
    for phase in phases:
        gate = phase["gate"]
        accuracy = float(phase["eval_accuracy"])
        phase_trials = int(phase["phase_trials"])
        passed = phase["passed"]
        print(
            f"  Phase {phase['phase']}: {gate:6s} | Acc: {accuracy:.1%} | Trials: {phase_trials} | Passed: {passed if passed else 'NO'}"
        )

    # 2. Detailed OR Analysis
    print("\n2. DETAILED OR GATE ANALYSIS")
    print("-" * 80)

    or_trials = [t for t in trials if t["gate"] == "or"]
    print(f"Total OR trials: {len(or_trials)}")

    if or_trials:
        # Extract numeric data
        or_data = []
        for t in or_trials:
            try:
                or_data.append(
                    {
                        "trial": int(t["trial"]),
                        "dopamine": float(t["dopamine_pulse"]),
                        "eligibility": float(t["mean_eligibility_abs"]),
                        "dw": float(t["mean_abs_dw"]),
                        "correct": int(t["correct"]),
                        "weights_mean": float(t["weights_mean"]),
                    }
                )
            except (ValueError, KeyError) as e:
                print(f"  Error parsing trial: {e}")
                continue

        if or_data:
            # Statistics
            dws = [d["dw"] for d in or_data]
            dopamines = [d["dopamine"] for d in or_data if d["dopamine"] != 0]
            eligibilities = [d["eligibility"] for d in or_data if d["eligibility"] > 0]
            correct_count = sum(1 for d in or_data if d["correct"])

            print(
                f"\n  Accuracy: {correct_count}/{len(or_data)} = {correct_count / len(or_data):.1%}"
            )
            print("\n  Weight Changes (mean_abs_dw):")
            print(f"    Mean:   {mean(dws):.6f}")
            if len(dws) > 1:
                print(f"    StDev:  {stdev(dws):.6f}")
            print(f"    Min:    {min(dws):.6f}")
            print(f"    Max:    {max(dws):.6f}")

            print("\n  Dopamine Signals:")
            print(f"    Non-zero dopamine events: {len(dopamines)}")
            print(f"    Mean dopamine (non-zero): {mean(dopamines) if dopamines else 'N/A'}")

            print("\n  Eligibility Traces:")
            print(f"    Trials with eligibility > 0: {len(eligibilities)}")
            print(
                f"    Mean eligibility (when > 0): {mean(eligibilities) if eligibilities else 'N/A':.6f}"
            )

            print("\n  First 5 OR trials:")
            print("    Trial | Dopamine | Eligibility | dW       | Correct")
            for d in or_data[:5]:
                print(
                    f"    {d['trial']:3d}   | {d['dopamine']:+7.2f}   | {d['eligibility']:11.4f} | {d['dw']:.6f} | {d['correct']}"
                )

            print("\n  Last 5 OR trials:")
            for d in or_data[-5:]:
                print(
                    f"    {d['trial']:3d}   | {d['dopamine']:+7.2f}   | {d['eligibility']:11.4f} | {d['dw']:.6f} | {d['correct']}"
                )

    # 3. Compare to AND
    print("\n3. COMPARISON: OR vs AND")
    print("-" * 80)

    and_trials = [t for t in trials if t["gate"] == "and"]
    if and_trials:
        and_data = []
        for t in and_trials:
            try:
                and_data.append(
                    {
                        "dw": float(t["mean_abs_dw"]),
                        "eligibility": float(t["mean_eligibility_abs"]),
                        "correct": int(t["correct"]),
                    }
                )
            except (ValueError, KeyError):
                continue

        if and_data:
            and_accuracy = sum(1 for d in and_data if d["correct"]) / len(and_data)
            and_dw = mean([d["dw"] for d in and_data])

            print("  OR gate:")
            print(f"    Accuracy:    {correct_count / len(or_data):.1%}")
            print(f"    Mean dW:     {mean(dws):.6f}")

            print("\n  AND gate:")
            print(f"    Accuracy:    {and_accuracy:.1%}")
            print(f"    Mean dW:     {and_dw:.6f}")

            print("\n  Difference:")
            print(f"    Accuracy gap: {and_accuracy - correct_count / len(or_data):.1%}")
            print(f"    dW ratio:     {and_dw / mean(dws) if mean(dws) > 0 else 'inf':.2f}x")

    # 4. Diagnosis
    print("\n4. DIAGNOSIS")
    print("-" * 80)

    if or_data and mean(dws) < 0.001:
        print("  ❌ PROBLEM DETECTED: Near-zero weight changes despite training")
        print(f"     mean_abs_dw = {mean(dws):.6f} (should be > 0.001)")

        if len(dopamines) > 0:
            print("  ✓ Dopamine signals ARE being released")
        else:
            print("  ✗ Dopamine signals NOT released (or always 0)")

        if len(eligibilities) > 0:
            print(f"  ✓ Eligibility traces ARE present (mean: {mean(eligibilities):.4f})")
        else:
            print("  ✗ Eligibility traces NOT present")

        print("\n  Likely causes:")
        print("    1. Learning rate (lr) is too small")
        print("    2. Wrapper gains are too aggressive, collapsing lr_scale")
        print("    3. Weight clipping is too tight")
        print("    4. Synapse model configuration issue")
    else:
        print(f"  ✓ Weight changes appear normal (mean_abs_dw = {mean(dws):.6f})")


if __name__ == "__main__":
    analyze_run()
