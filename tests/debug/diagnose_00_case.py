#!/usr/bin/env python3
"""
Deep diagnostic: Why OR gate 0,0 case fails and weights don't stick.
"""

import csv
from pathlib import Path
from statistics import mean, stdev


def load_csv(filepath):
    with open(filepath) as f:
        return list(csv.DictReader(f))


def find_latest_or_run():
    """Find most recently modified run directory."""
    artifacts = Path("artifacts")
    run_dirs = sorted(
        [d for d in artifacts.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0]
    return None


def analyze_by_case(trials, gate="or"):
    """Analyze performance by input case for OR gate."""
    # OR truth table: 0,0->0  0,1->1  1,0->1  1,1->1
    case_names = {
        "0,0": ("0.0", "0.0"),
        "0,1": ("0.0", "1.0"),
        "1,0": ("1.0", "0.0"),
        "1,1": ("1.0", "1.0"),
    }

    or_trials = [t for t in trials if t.get("gate") == gate]

    print("\n" + "=" * 80)
    print("ANALYSIS BY INPUT CASE")
    print("=" * 80)

    for case_label, (x0_str, x1_str) in case_names.items():
        case_trials = [t for t in or_trials if t.get("x0") == x0_str and t.get("x1") == x1_str]

        if not case_trials:
            continue

        correct = sum(1 for t in case_trials if int(t["correct"]))
        accuracy = correct / len(case_trials)

        # Get dW stats
        dws = [float(t.get("mean_abs_dw", 0)) for t in case_trials]
        # Get prediction confidence
        pred_vals = [float(t.get("pred_00", 0)) for t in case_trials]

        # For OR gate: 0,0 should predict 0 (pred_00 should be high)
        # Other cases should predict 1 (pred_01, pred_10, pred_11 should be high)

        print(f"\nCase {case_label}:")
        print(f"  Trials: {len(case_trials)}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Mean dW: {mean(dws):.6f}" + (f" (σ={stdev(dws):.6f})" if len(dws) > 1 else ""))
        print(f"  Mean pred: {mean(pred_vals):.4f}")

        # Show trajectory of first, middle, last trials
        if len(case_trials) > 2:
            first = case_trials[0]
            mid = case_trials[len(case_trials) // 2]
            last = case_trials[-1]

            print(
                f"  First trial: dW={first.get('mean_abs_dw', 'N/A')}, pred={first.get('pred_00', 'N/A')}, correct={first.get('correct', 'N/A')}"
            )
            print(
                f"  Mid trial:   dW={mid.get('mean_abs_dw', 'N/A')}, pred={mid.get('pred_00', 'N/A')}, correct={mid.get('correct', 'N/A')}"
            )
            print(
                f"  Last trial:  dW={last.get('mean_abs_dw', 'N/A')}, pred={last.get('pred_00', 'N/A')}, correct={last.get('correct', 'N/A')}"
            )


def analyze_weight_stability(trials, gate="or"):
    """Check if weights are changing after learning."""
    print("\n" + "=" * 80)
    print("WEIGHT STABILITY ANALYSIS")
    print("=" * 80)

    or_trials = [t for t in trials if t.get("gate") == gate]

    if not or_trials:
        print("No OR trials found")
        return

    dws = [float(t.get("mean_abs_dw", 0)) for t in or_trials]

    print("\nWeight change over time:")
    print(f"  Trial 1-5:   mean dW = {mean(dws[0:5]):.6f}")
    print(f"  Trial 10-15: mean dW = {mean(dws[9:15]) if len(dws) > 15 else 'N/A':.6f}")
    print(f"  Trial 20-25: mean dW = {mean(dws[19:25]) if len(dws) > 25 else 'N/A':.6f}")
    print(f"  Trial ...-N: mean dW = {mean(dws[-5:]):.6f}")

    # Check if accuracy spikes but dW stays low
    print("\nSearching for 'stuck' accuracy spikes (high acc, low dW)...")
    high_acc_trials = [t for t in or_trials if int(t.get("correct", 0)) == 1]

    if high_acc_trials:
        dws_correct = [float(t.get("mean_abs_dw", 0)) for t in high_acc_trials]
        print(f"  When correct: mean dW = {mean(dws_correct):.6f}")
        print(f"  Correct trials: {len(high_acc_trials)}/{len(or_trials)}")

        # Find trials with many consecutive correct
        max_streak = 0
        current_streak = 0
        streak_dw = []

        for t in or_trials:
            if int(t.get("correct", 0)) == 1:
                current_streak += 1
                streak_dw.append(float(t.get("mean_abs_dw", 0)))
            else:
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_streak_dw = mean(streak_dw)
                current_streak = 0
                streak_dw = []

        print(f"  Max correct streak: {max_streak} trials")
        if max_streak > 0:
            print(f"  During max streak: mean dW = {max_streak_dw:.6f}")


def analyze_dopamine_eligibility(trials, gate="or"):
    """Check dopamine and eligibility trace on 0,0 case."""
    print("\n" + "=" * 80)
    print("DOPAMINE & ELIGIBILITY ON 0,0 CASE")
    print("=" * 80)

    or_trials = [t for t in trials if t.get("gate") == gate]
    case_00 = [t for t in or_trials if t.get("x0") == "0.0" and t.get("x1") == "0.0"]

    if not case_00:
        print("No 0,0 trials found")
        return

    dopamines = [float(t.get("dopamine_pulse", 0)) for t in case_00]
    eligibilities = [float(t.get("mean_eligibility_abs", 0)) for t in case_00]
    dws = [float(t.get("mean_abs_dw", 0)) for t in case_00]

    print(f"\n0,0 case statistics (n={len(case_00)}):")
    print(
        f"  Dopamine: mean={mean(dopamines):.4f}, min={min(dopamines):.4f}, max={max(dopamines):.4f}"
    )
    print(
        f"  Eligibility: mean={mean(eligibilities):.4f}, min={min(eligibilities):.4f}, max={max(eligibilities):.4f}"
    )
    print(f"  dW: mean={mean(dws):.6f}, min={min(dws):.6f}, max={max(dws):.6f}")

    # Check correlation: when dopamine is high, is dW also high on 0,0?
    high_dopamine = [t for t in case_00 if float(t.get("dopamine_pulse", 0)) > 0.5]
    if high_dopamine:
        dws_high_dop = [float(t.get("mean_abs_dw", 0)) for t in high_dopamine]
        print(
            f"\n  When dopamine > 0.5: mean dW = {mean(dws_high_dop):.6f} ({len(high_dopamine)} trials)"
        )

    low_dopamine = [t for t in case_00 if float(t.get("dopamine_pulse", 0)) < -0.5]
    if low_dopamine:
        dws_low_dop = [float(t.get("mean_abs_dw", 0)) for t in low_dopamine]
        print(
            f"  When dopamine < -0.5: mean dW = {mean(dws_low_dop):.6f} ({len(low_dopamine)} trials)"
        )


def main():
    run_dir = find_latest_or_run()
    if not run_dir:
        print("No run directories found")
        return

    print(f"Analyzing: {run_dir.name}")

    trials = load_csv(run_dir / "trials.csv")
    print(f"Total trials: {len(trials)}")

    analyze_by_case(trials)
    analyze_weight_stability(trials)
    analyze_dopamine_eligibility(trials)

    print("\n" + "=" * 80)
    print("HYPOTHESIS:")
    print("=" * 80)
    print("""
1. The 0,0 case is fundamentally hard for the network architecture
   - Likely needs stronger synaptic weights or better hidden representations

2. Accuracy spikes but dW is low means:
   - When network gets the case right, minimal weight changes occur
   - This suggests weights are already saturated or at boundaries
   - Random correct predictions happen without learning

3. Stuck weights mean:
   - Learning rule isn't strong enough (lr too low)
   - OR eligibility traces aren't accumulating properly on 0,0
   - OR dopamine signal is too weak on this case

REAL FIX NEEDED:
- Increase learning rate for OR curriculum phase
- OR check if 0,0 is getting correct dopamine signal
- OR verify hidden layer can distinguish 0,0 from other cases
    """)


if __name__ == "__main__":
    main()
