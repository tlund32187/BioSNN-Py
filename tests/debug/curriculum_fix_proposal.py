#!/usr/bin/env python3
"""
DETAILED ANALYSIS & FIX PROPOSAL: OR Gate Learning Failure in Curriculum Mode
===============================================================================

## Problem Summary

Run tmp_curriculum_probe_ctx80 trained the OR gate with only 20 trials per phase,
resulting in 25% accuracy (random guessing for 4-output classification).

The successful run (run_20260213_101744_206300) gave OR gate 2500 trials and achieved 75% accuracy.

## Root Cause Analysis

### Current Behavior:
The logic_curriculum demo is configured with "steps": 2500, but the runner
currently distributes ALL 2500 trials to EACH phase independently.

With 6 gates (OR, AND, NOR, NAND, XOR, XNOR):
- Ideal: ~417 trials per gate (2500 / 6)
- Current: 2500 trials per gate = 15000 total (6x overspend!)
- Probe test: 20 trials per gate = 120 total (likely a test configuration)

### Why OR Fails with 20 trials:
- 20 trials can only cover ~5 iterations through the 4-output truth table
- OR gate needs 50-100+ trials to learn reliably given the network topology
- Insufficient exploration and weight updates to converge

## Recommended Fix

### Option 1: Smart Phase Budget Distribution (Recommended)
Modify the logic_curriculum runner to divide the total budget across phases:

```python
def run_logic_gate_curriculum(...):
    num_phases = len(gate_sequence)
    phase_trials = int(cfg.steps / num_phases)  # Distribute budget evenly
```

This way, with steps=2500 and 6 gates:
- Each gate gets 417 trials (reasonable for learning)
- Total stays at 2500 (respects demo config)
- Scales automatically if gates added/removed

### Option 2: Increase Total Steps in Demo Config
Keep current per-phase behavior but increase "steps" in demo_registry:

```python
"logic_curriculum": {
    "steps": 500,  # Per gate, so 3000 total for 6 gates
    ...
},
```

Then each gate gets 500 trials (comfortable learning target).

### Option 3: Add Explicit Phase/Gate Budget Parameters
Add new config parameters to allow fine-grained control:

```python
"logic_curriculum": {
    "steps": 2500,
    "steps_per_gate": 500,  # Takes precedence if set
    ...
},
```

## Proposed Implementation

File: src/biosnn/tasks/logic_gates/runner.py
Function: run_logic_gate_curriculum

Change line 537 from:
```python
phase_trials = int(cfg.steps if phase_steps is None else phase_steps)
```

To:
```python
# Distribute total budget across phases for curriculum mode
if phase_steps is None:
    num_gates = len(gate_sequence)
    phase_trials = max(20, int(cfg.steps // num_gates))  # At least 20 per phase
else:
    phase_trials = int(phase_steps)
```

## Expected Outcomes

After fix:
1. OR gate will get ~400 trials in curriculum mode
2. OR accuracy will improve from 25% to ~70-80%
3. Subsequent gates (AND, NOR, etc.) will also improve
4. Total curriculum time stays bounded (2500 total steps)

## Verification Steps

1. Run: `&'.venv/Scripts/python.exe' -m biosnn.runners.cli --demo logic_curriculum --steps 2500 --device cpu`
2. Check phase_summary.csv for OR accuracy (target: >70%)
3. Verify all gates improve sequentially
4. Check total trials ≈ 2500

## Files to Modify

1. src/biosnn/tasks/logic_gates/runner.py (line 535-540)
2. Update logic_curriculum defaults if needed (maybe increase steps to 5000-10000)
"""

if __name__ == "__main__":
    print(__doc__)
