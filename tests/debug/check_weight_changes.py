#!/usr/bin/env python3
"""Check if actual weights are changing in the successful run."""

import contextlib
import csv

# Check successful run weights
print("Checking successful run weights (run_20260213_101744_206300):")
with open("artifacts/run_20260213_101744_206300/weights.csv") as f:
    reader = csv.DictReader(f)
    weights = list(reader)

if weights:
    print(f"Total weight samples: {len(weights)}")
    print(f"Columns: {list(weights[0].keys())[:5]}...")

    # Get first and last weight samples
    first = weights[0]
    last = weights[-1]

    # Pick a weight to track
    weight_keys = list(first.keys())
    if weight_keys:
        key = weight_keys[0]
        try:
            w_first = float(first[key])
            w_last = float(last[key])
            print(f"\nSample weight changes (first weight key: {key}):")
            print(f"  Trial 1: {w_first:.6f}")
            print(f"  Trial {len(weights)}: {w_last:.6f}")
            print(f"  Changed: {w_first != w_last}")
        except ValueError:
            pass

    # Check if all weights are identical across trials
    all_same = True
    for key in weight_keys[:5]:  # Check first 5 weights
        vals = []
        for w in weights:
            try:
                vals.append(float(w[key]))
            except ValueError:
                break
        if len(set(vals)) > 1:
            all_same = False
            print(f"\nWeight {key} CHANGED over trials (unique values: {len(set(vals))})")
            break

    if all_same:
        print(
            "\nAll sampled weights appear IDENTICAL across trials - weights might not be tracked per-trial"
        )

print("\n" + "=" * 80)
print("CHECKING LATEST RUN WEIGHTS:")
with open("artifacts/run_20260213_104400_682000/trials.csv") as f:
    reader = csv.DictReader(f)
    trials = list(reader)

# Get first few trials with different weights_mean values
trial_weights = []
for t in trials:
    with contextlib.suppress(ValueError, KeyError):
        trial_weights.append((int(t["trial"]), float(t["weights_mean"])))
    trial_weights.sort()
    print("\nLatest run weight_mean over trials:")
    print(f"  Trial {trial_weights[0][0]}: {trial_weights[0][1]:.6f}")
    print(
        f"  Trial {trial_weights[len(trial_weights) // 2][0]}: {trial_weights[len(trial_weights) // 2][1]:.6f}"
    )
    print(f"  Trial {trial_weights[-1][0]}: {trial_weights[-1][1]:.6f}")
    print(f"  Range: {trial_weights[-1][1] - trial_weights[0][1]:.6f}")
