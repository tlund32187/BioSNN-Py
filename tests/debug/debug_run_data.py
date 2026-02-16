#!/usr/bin/env python3
"""Check trial data from broken run."""

import csv
import json
from pathlib import Path

project_root = Path(__file__).parent

# Check one of the trial files
run_dir = project_root / "artifacts/run_20260212_140714_134900"
trials_csv = run_dir / "trials.csv"

print(f"Checking trials CSV: {trials_csv}")
if trials_csv.exists():
    with open(trials_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"Total trials: {len(rows)}")
        if rows:
            first_row = rows[0]
            print(f"\nFirst trial keys: {list(first_row.keys())[:20]}")  # First 20 keys
            print(f"All keys: {list(first_row.keys())}")  # All keys
            
            # Look for dopamine or modulator columns
            modulator_cols = [k for k in first_row if "dopamine" in k.lower() or "modulator" in k.lower()]
            if modulator_cols:
                print("\nModulator columns found:")
                for col in modulator_cols:
                    try:
                        values = [float(row[col]) for row in rows if row[col]]
                        print(f"  {col}: min={min(values):.4f}, max={max(values):.4f}, mean={sum(values)/len(values):.4f}")
                    except Exception as e:
                        print(f"  {col}: Error - {e}")
            else:
                print("\nNo dopamine/modulator columns in trials CSV")
else:
    print(f"File not found: {trials_csv}")

# Check run_features
features_file = run_dir / "run_features.json"
print(f"\nChecking run_features: {features_file}")
if features_file.exists():
    with open(features_file) as f:
        features = json.load(f)
        if "mean_abs_dw" in features:
            print(f"mean_abs_dw: {features['mean_abs_dw']}")
        if "modulators" in features:
            print(f"modulators: {features['modulators']}")
        else:
            print("No 'modulators' key in features")
else:
    print(f"File not found: {features_file}")
