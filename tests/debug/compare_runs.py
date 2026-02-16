#!/usr/bin/env python3
"""Compare successful and failed curriculum runs."""

import csv
from pathlib import Path


def load_csv(filepath):
    with open(filepath) as f:
        return list(csv.DictReader(f))


def analyze_run(run_dir, name):
    path = Path(run_dir)

    if not (path / "phase_summary.csv").exists():
        print(f"\n{name}: NO PHASE SUMMARY FOUND")
        return

    phases = load_csv(path / "phase_summary.csv")
    trials = load_csv(path / "trials.csv") if (path / "trials.csv").exists() else []

    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")

    # Phase summary
    print(f"\nPhase Summary ({len(phases)} phases):")
    for phase in phases[:3]:  # Show first 3
        gate = phase["gate"]
        acc = float(phase["eval_accuracy"])
        phase_trials = int(phase["phase_trials"])
        phase_gate_trials = int(phase["phase_gate_trials"])
        print(
            f"  Phase {phase['phase']}: {gate:6s} | Trials: {phase_trials:4d} | Gate-only: {phase_gate_trials:4d} | Accuracy: {acc:.1%}"
        )

    if len(phases) > 3:
        print(f"  ... {len(phases) - 3} more phases")

    # Statistics
    if trials:
        total_trials = len(trials)
        print(f"\nTrials data: {total_trials} total trials")

        # Count trials per gate
        gates: dict[str, int] = {}
        for t in trials:
            gate = t.get("gate", "unknown")
            gates[gate] = gates.get(gate, 0) + 1

        print("Trials per gate:")
        for gate in sorted(gates.keys()):
            print(f"  {gate:6s}: {gates[gate]:4d} trials")

        # Check for OR phase
        or_trials = [t for t in trials if t.get("gate") == "or"]
        if or_trials:
            # Get a sample trial to check dW range
            sample_or = or_trials[min(5, len(or_trials) - 1)]
            print("OR gate sample trial:")
            print(
                f"  Trial {sample_or['trial']}: dW={sample_or.get('mean_abs_dw', 'N/A')}, "
                + f"dopamine={sample_or.get('dopamine_pulse', 'N/A')}, "
                + f"correct={sample_or.get('correct', 'N/A')}"
            )


# Compare runs
print("COMPARING CURRICULUM RUNS")
print("=" * 80)

analyze_run("artifacts/tmp_curriculum_probe_ctx80", "PROBE RUN (recent, FAILED to learn OR)")
analyze_run("artifacts/run_20260213_101744_206300", "SUCCESSFUL RUN (learned OR to 75%)")

print("\n" + "=" * 80)
print("KEY DIFFERENCES TO INVESTIGATE:")
print("=" * 80)
print("""
1. PHASE CONFIGURATION:
   - Probe: 20 trials per phase with curriculum
   - Successful: 2500 trials for OR alone

2. HYPOTHESIS:
   - 20 trials is TOO SHORT for the network to learn complex gates
   - OR gate requires more exploration and weight updates
   - Curriculum approach may be TOO FAST between gates

3. POTENTIAL FIXES:
   a) Increase phase_trials in curriculum (e.g., 500 per gate)
   b) Add early stopping threshold check before curriculum advance
   c) Tune learning rate specifically for curriculum mode
   d) Check epsilon-greedy exploration rate during early phases
   e) Verify dopamine scaling in curriculum vs single-gate runs
""")
