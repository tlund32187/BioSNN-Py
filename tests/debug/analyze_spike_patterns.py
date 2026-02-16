"""
Analyze spike patterns by population (input, hidden, output) in both runs.
"""

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

working_run_dir = Path("artifacts/run_20260212_132912_619800")
broken_run_dir = Path("artifacts/run_20260212_140714_134900")

working_spikes = pd.read_csv(working_run_dir / "spikes.csv")
broken_spikes = pd.read_csv(broken_run_dir / "spikes.csv")

print("=" * 70)
print("SPIKE DATA ANALYSIS - POPULATION COMPARISON")
print("=" * 70)

print("\nWORKING RUN spikes.csv shape:", working_spikes.shape)
print("BROKEN RUN spikes.csv shape:", broken_spikes.shape)

print("\nWORKING RUN - First few rows:")
print(working_spikes.head())

print("\nWORKING RUN - Column info:")
print(working_spikes.dtypes)

print("\nWORKING RUN - Summary statistics:")
print(working_spikes.describe())

print("\n" + "=" * 70)
print("BROKEN RUN - First few rows:")
print(broken_spikes.head())

print("\nBROKEN RUN - Summary statistics:")
print(broken_spikes.describe())

# Compare population spike rates
print("\n" + "=" * 70)
print("SPIKE COUNT BY POPULATION")
print("=" * 70)

for run_name, spikes_df in [("WORKING", working_spikes), ("BROKEN", broken_spikes)]:
    print(f"\n{run_name} RUN:")

    # Group by neuron_id if available
    if "neuron_id" in spikes_df.columns:
        print(f"  Total spikes: {len(spikes_df)}")
        print(f"  Unique neurons: {spikes_df['neuron_id'].nunique()}")
        print(
            f"  Spikes per neuron (mean): {len(spikes_df) / spikes_df['neuron_id'].nunique():.2f}"
        )

        # Look for population designation
        if "population" in spikes_df.columns:
            print("\n  Spikes by population:")
            for pop in spikes_df["population"].unique():
                pop_spikes = len(spikes_df[spikes_df["population"] == pop])
                print(f"    {pop}: {pop_spikes:6d} spikes")

        # Show neuron IDs to infer populations
        print(
            f"\n  Neuron ID range: {spikes_df['neuron_id'].min()} - {spikes_df['neuron_id'].max()}"
        )

    # Show all columns
    print(f"\n  Columns: {list(spikes_df.columns)}")

# Deep comparison
print("\n" + "=" * 70)
print("POPULATION-LEVEL ANALYSIS")
print("=" * 70)

working_total = len(working_spikes)
broken_total = len(broken_spikes)
ratio = broken_total / working_total if working_total > 0 else 0

print("\nTotal spikes:")
print(f"  Working: {working_total:8d}")
print(f"  Broken:  {broken_total:8d}")
print(f"  Ratio (Broken/Working): {ratio:.3f}")

if "neuron_id" in working_spikes.columns and "neuron_id" in broken_spikes.columns:
    unique_working = working_spikes["neuron_id"].nunique()
    unique_broken = broken_spikes["neuron_id"].nunique()

    print("\nUnique active neurons:")
    print(f"  Working: {unique_working}")
    print(f"  Broken:  {unique_broken}")

    # Which neurons were active in working but not broken?
    neurons_working = set(working_spikes["neuron_id"])
    neurons_broken = set(broken_spikes["neuron_id"])
    only_working = neurons_working - neurons_broken
    only_broken = neurons_broken - neurons_working

    print(f"\nActive only in Working: {len(only_working)} neurons")
    print(f"Active only in Broken: {len(only_broken)} neurons")
    print(f"Active in both: {len(neurons_working & neurons_broken)} neurons")
