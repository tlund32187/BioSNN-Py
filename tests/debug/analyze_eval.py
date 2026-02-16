#!/usr/bin/env python3
"""Analyze eval.csv to see learning progression in both runs."""

import csv
from pathlib import Path

project_root = Path(__file__).parent

runs = {
    "working": project_root / "artifacts/run_20260212_132912_619800",
    "broken": project_root / "artifacts/run_20260212_140714_134900",
}

print("=" * 80)
print("LEARNING PROGRESSION ANALYSIS")
print("=" * 80)

for run_name, run_dir in runs.items():
    eval_csv = run_dir / "eval.csv"
    
    if not eval_csv.exists():
        print(f"\n{run_name.upper()}: File not found")
        continue
    
    print(f"\n{run_name.upper()} RUN:")
    print("-" * 80)
    
    with open(eval_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total eval points: {len(rows)}")
    
    # Get key columns
    if rows:
        print(f"Columns: {list(rows[0].keys())}\n")
        
        # Sample some rows at different points
        samples = [
            rows[0],
            rows[len(rows) // 4],
            rows[len(rows) // 2],
            rows[-1],
        ]
        
        print("Sample learning progression:")
        for _, row in enumerate(samples):
            trial = row.get("trial", "?")
            mean_dw = float(row.get("mean_abs_dw", 0))
            elig = float(row.get("mean_eligibility_abs", 0))
            acc = float(row.get("eval_accuracy", 0))
            print(f"  Trial {trial:>4}: mean_abs_dw={mean_dw:.6f}, elig={elig:.6f}, acc={acc:.4f}")
        
        # Check if learning happens
        mean_dws = [float(row.get("mean_abs_dw", 0)) for row in rows]
        max_dw = max(mean_dws) if mean_dws else 0
        min_dw = min(mean_dws) if mean_dws else 0
        avg_dw = sum(mean_dws) / len(mean_dws) if mean_dws else 0
        
        print("\nmean_abs_dw statistics:")
        print(f"  Min: {min_dw:.6f}")
        print(f"  Max: {max_dw:.6f}")
        print(f"  Avg: {avg_dw:.6f}")
        
        if max_dw < 0.0001:
            print("  ❌ ZERO LEARNING DETECTED!")
        else:
            print("  ✅ Learning is happening")

        # Check eligibility
        eligs = [float(row.get("mean_eligibility_abs", 0)) for row in rows]
        max_elig = max(eligs) if eligs else 0
        print("\nEligibility trace statistics:")
        print(f"  Max: {max_elig:.6f}")
        if max_elig < 0.0001:
            print("  ⚠️  ZERO ELIGIBILITY - no weight changes possible!")
