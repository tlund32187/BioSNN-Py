"""Read and display trial CSV data."""

import csv
import sys

csv_path = sys.argv[1]
rows = list(csv.DictReader(open(csv_path)))

print(f"Total trials: {len(rows)}")
print("\nFIRST 15 trials:")
for r in rows[:15]:
    t = int(r["trial"])
    corr = r["correct"]
    x0 = r["x0"]
    x1 = r["x1"]
    pred = r["pred"]
    target = r["target"]
    w0 = float(r["w_out0_mean"])
    w1 = float(r["w_out1_mean"])
    e0 = float(r["elig_out0_mean"])
    e1 = float(r["elig_out1_mean"])
    exp = r.get("explored", "?")
    print(
        f"  t={t:4d} x=({x0},{x1}) tgt={target} pred={pred} corr={corr} exp={exp} "
        f"w0={w0:.4f} w1={w1:.4f} e0={e0:.5f} e1={e1:.5f}"
    )

print("\nTRIALS 100-110:")
for r in rows[99:110]:
    t = int(r["trial"])
    corr = r["correct"]
    x0 = r["x0"]
    x1 = r["x1"]
    pred = r["pred"]
    target = r["target"]
    w0 = float(r["w_out0_mean"])
    w1 = float(r["w_out1_mean"])
    print(f"  t={t:4d} x=({x0},{x1}) tgt={target} pred={pred} corr={corr} w0={w0:.4f} w1={w1:.4f}")

print("\nTRIALS 400-410:")
for r in rows[399:410]:
    t = int(r["trial"])
    corr = r["correct"]
    x0 = r["x0"]
    x1 = r["x1"]
    pred = r["pred"]
    target = r["target"]
    w0 = float(r["w_out0_mean"])
    w1 = float(r["w_out1_mean"])
    print(f"  t={t:4d} x=({x0},{x1}) tgt={target} pred={pred} corr={corr} w0={w0:.4f} w1={w1:.4f}")

print("\nLAST 5 trials:")
for r in rows[-5:]:
    t = int(r["trial"])
    corr = r["correct"]
    x0 = r["x0"]
    x1 = r["x1"]
    pred = r["pred"]
    target = r["target"]
    w0 = float(r["w_out0_mean"])
    w1 = float(r["w_out1_mean"])
    print(f"  t={t:4d} x=({x0},{x1}) tgt={target} pred={pred} corr={corr} w0={w0:.4f} w1={w1:.4f}")
