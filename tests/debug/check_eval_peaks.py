#!/usr/bin/env python3
"""Check eval CSV for accuracy peaks and what happened to 0,0."""

import csv
from pathlib import Path

run_dir = sorted(
    [d for d in Path("artifacts").iterdir() if d.is_dir() and d.name.startswith("run_")],
    key=lambda x: x.stat().st_mtime,
    reverse=True,
)[0]

with open(run_dir / "eval.csv") as f:
    evals = list(csv.DictReader(f))

or_evals = [e for e in evals if e.get("gate") == "or"]
print(f"OR gate eval samples: {len(or_evals)}")

peaks = [e for e in or_evals if float(e.get("eval_accuracy", 0)) >= 0.99]
print(f"Eval accuracy peaks (>=99%): {len(peaks)}")

if peaks:
    print("\nFirst peak:")
    p = peaks[0]
    print(f"  Trial: {p.get('trial')}")
    print(f"  Accuracy: {p.get('eval_accuracy')}")
    print(f"  eval_00: {p.get('eval_00', 'N/A')}")
    print(f"  eval_01: {p.get('eval_01', 'N/A')}")
    print(f"  eval_10: {p.get('eval_10', 'N/A')}")
    print(f"  eval_11: {p.get('eval_11', 'N/A')}")

    # Show next few to see if peaks are sustained
    print("\nPeak accuracy timeline:")
    for _i, p in enumerate(peaks[:10]):
        acc = float(p.get("eval_accuracy", 0))
        trial = int(p.get("trial", 0))
        print(f"  Trial {trial:3d}: {acc:.1%}")

# Check if 0,0 specifically ever reaches high accuracy
print("\nDuring peaks, what's 0,0 accuracy?")
if peaks:
    for _i, p in enumerate(peaks[:3]):
        trial = int(p.get("trial", 0))
        eval_00 = p.get("eval_00", "N/A")
        print(f"  Trial {trial}: eval_00 = {eval_00}")
