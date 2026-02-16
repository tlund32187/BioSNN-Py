#!/usr/bin/env python3
"""Analyze what happens during accuracy peaks and why 0,0 case fails."""

import csv
from pathlib import Path
from statistics import mean


def find_latest_or_run():
    run_dirs = sorted(
        [d for d in Path("artifacts").iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def main():
    run_dir = find_latest_or_run()
    if run_dir is None:
        print("No run directories found in artifacts/")
        return
    print(f"Analyzing: {run_dir.name}\n")

    with open(run_dir / "trials.csv") as f:
        trials = list(csv.DictReader(f))

    # Look for eval_accuracy spikes
    print("=" * 80)
    print("SEARCHING FOR EVAL ACCURACY PEAKS")
    print("=" * 80)

    or_trials = [t for t in trials if t.get("gate") == "or"]

    # Find trials where eval_accuracy = 1.0 (100%)
    perfect_trials = [t for t in or_trials if float(t.get("eval_accuracy", 0)) >= 0.99]

    if perfect_trials:
        print(f"\nFound {len(perfect_trials)} trials with ≥99% eval accuracy")

        # Analyze around peak accuracy
        peak_trial = int(perfect_trials[0]["trial"])
        window = 5

        print(f"\nAnalyzing trials around peak (trial {peak_trial}):")
        print("Trial | EvalAcc | Correct | Sample00 | Weights  | Weight_range")
        print("-" * 70)

        for offset in range(-window, window + 1):
            trial_idx = peak_trial + offset - 1
            if 0 <= trial_idx < len(or_trials):
                t = or_trials[trial_idx]
                trial_num = int(t["trial"])
                eval_acc = float(t.get("eval_accuracy", 0))
                correct = int(t.get("correct", 0))
                # Check if this trial was a 0,0 case
                x0, x1 = t.get("x0", "?"), t.get("x1", "?")
                is_00 = x0 == "0.0" and x1 == "0.0"

                weights_min = float(t.get("weights_min", 0))
                weights_max = float(t.get("weights_max", 0))
                weights_range = weights_max - weights_min

                sample_00 = "*0,0*" if is_00 else f"{x0},{x1}"

                print(
                    f"{trial_num:4d} | {eval_acc:.2f} | {correct:7d} | {sample_00:8s} | "
                    f"{weights_max:8.4f} | {weights_range:.4f}"
                )

    # Second analysis: what's the weight range when 0,0 is correct vs incorrect?
    print("\n" + "=" * 80)
    print("WEIGHT STATE WHEN 0,0 IS CORRECT VS INCORRECT")
    print("=" * 80)

    case_00 = [t for t in or_trials if t.get("x0") == "0.0" and t.get("x1") == "0.0"]

    correct_00 = [t for t in case_00 if int(t.get("correct", 0)) == 1]
    incorrect_00 = [t for t in case_00 if int(t.get("correct", 0)) == 0]

    if correct_00:
        correct_weights_min = mean([float(t.get("weights_min", 0)) for t in correct_00])
        correct_weights_max = mean([float(t.get("weights_max", 0)) for t in correct_00])
        correct_weights_range = mean(
            [float(t.get("weights_max", 0)) - float(t.get("weights_min", 0)) for t in correct_00]
        )

        print(f"\nWhen 0,0 is CORRECT ({len(correct_00)} trials):")
        print(f"  Weight min (mean): {correct_weights_min:.6f}")
        print(f"  Weight max (mean): {correct_weights_max:.6f}")
        print(f"  Weight range (mean): {correct_weights_range:.6f}")

    if incorrect_00:
        incorrect_weights_min = mean([float(t.get("weights_min", 0)) for t in incorrect_00])
        incorrect_weights_max = mean([float(t.get("weights_max", 0)) for t in incorrect_00])
        incorrect_weights_range = mean(
            [float(t.get("weights_max", 0)) - float(t.get("weights_min", 0)) for t in incorrect_00]
        )

        print(f"\nWhen 0,0 is INCORRECT ({len(incorrect_00)} trials):")
        print(f"  Weight min (mean): {incorrect_weights_min:.6f}")
        print(f"  Weight max (mean): {incorrect_weights_max:.6f}")
        print(f"  Weight range (mean): {incorrect_weights_range:.6f}")

    # Check dopamine signal on 0,0 cases
    print("\n" + "=" * 80)
    print("DOPAMINE SIGNAL ANALYSIS FOR 0,0 CASE")
    print("=" * 80)

    dopamine_correct = [float(t.get("dopamine_pulse", 0)) for t in correct_00]
    dopamine_incorrect = [float(t.get("dopamine_pulse", 0)) for t in incorrect_00]

    print(f"\nDopamine when 0,0 is CORRECT: mean={mean(dopamine_correct):.4f}")
    print(f"Dopamine when 0,0 is INCORRECT: mean={mean(dopamine_incorrect):.4f}")

    # The key issue: does dopamine direction match correctness?
    correct_dopamine_pos = sum(1 for d in dopamine_correct if d > 0)
    print(f"\nDopamine is positive when correct: {correct_dopamine_pos}/{len(dopamine_correct)}")
    print(
        "(For OR 0,0 case: correct answer is 0, so dopamine should be NEGATIVE on correct trials)"
    )
    print("^^ THIS IS BACKWARDS! Dopamine should punish when network incorrectly outputs 1")

    print("\n" + "=" * 80)
    print("HYPOTHESIS: 0,0 CASE FAILURE ROOT CAUSE")
    print("=" * 80)
    print("""
The 0,0 case fails because:
1. For OR gate, 0,0 should output 0 (false)
2. When hidden neurons are initialized, they tend to activate more with inputs
3. This makes the network biased toward predicting "1" (output active)
4. When the network accidentally outputs 0 for 0,0, it's usually "silence" not real learning
5. When accuracy spikes to 100%, it might include a lucky 0,0 prediction
6. But dopamine signal for that correct prediction is weak/opposite sign
7. So weights don't lock in - they change back on the next trial

SOLUTION:
- The 0,0 case fundamentally needs:
  a) Higher learning rate to overcome initialization bias toward firing
  b) OR stronger dopamine signal on correct 0 outputs
  c) OR curriculum that pre-learns a "silent" or "off" state before OR
  d) OR different network architecture with an inhibitory bias on outputs
    """)


if __name__ == "__main__":
    main()
