"""
Analyze firing patterns in both runs to understand which neurons are suppressed.
"""

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

# Load eval.csv from both runs
working_run_dir = Path("artifacts/run_20260212_132912_619800")
broken_run_dir = Path("artifacts/run_20260212_140714_134900")

working_eval = pd.read_csv(working_run_dir / "eval.csv")
broken_eval = pd.read_csv(broken_run_dir / "eval.csv")

print("=" * 70)
print("FIRING PATTERN ANALYSIS")
print("=" * 70)

# Check what columns are available
print("\nAvailable columns in eval.csv:")
for col in working_eval.columns[:20]:  # First 20 columns
    print(f"  {col}")

print("\nLooking for spike-related columns...")
spike_cols = [
    col
    for col in working_eval.columns
    if "spike" in col.lower() or "fire" in col.lower() or "spk" in col.lower()
]
print(f"Spike columns found: {spike_cols}")

# Look for population-level statistics
pop_cols = [
    col for col in working_eval.columns if "pop_" in col.lower() or "population" in col.lower()
]
print(f"Population columns found (first 10): {pop_cols[:10]}")

# Show all columns
print(f"\nAll columns ({len(working_eval.columns)}):")
for i, col in enumerate(working_eval.columns):
    print(f"  {i:3d}: {col}")
    if i > 50:
        print(f"  ... ({len(working_eval.columns) - 50} more)")
        break

# Focus on neurons' activity
print("\n" + "=" * 70)
print("NEURON STATISTICS COMPARISON")
print("=" * 70)

for col in working_eval.columns:
    if "mean_" in col or "max_" in col or "spike" in col:
        working_vals = working_eval[col].values
        broken_vals = broken_eval[col].values

        # Compare first 10 trials
        working_mean = working_vals[:10].mean()
        broken_mean = broken_vals[:10].mean()

        if working_mean > 0 or broken_mean > 0:
            ratio = broken_mean / working_mean if working_mean > 0 else 0
            if ratio < 0.95 or ratio > 1.05:  # Show differences
                print(
                    f"{col:40s}: Working={working_mean:10.6f}, Broken={broken_mean:10.6f}, Ratio={ratio:.2f}"
                )

# Focus on layer-specific activity if available
print("\n" + "=" * 70)
print("SEARCHING FOR LAYER-SPECIFIC METRICS")
print("=" * 70)

for col in working_eval.columns:
    if any(x in col.lower() for x in ["input", "hidden", "out", "layer", "pop"]):
        print(f"  {col}")
