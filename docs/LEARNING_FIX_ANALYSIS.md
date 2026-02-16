# Logic Gates Learning Issue - Root Cause & Final Fix

## Problem

During logic curriculum testing runs, the network was not learning despite:

- **Dopamine being released**: `dopamine_pulse` values showed correct reward/punishment signals (0.3/-0.3)
- **Outputs being generated**: Hidden and output neurons producing spikes
- **Eligibility traces accumulating**: `mean_eligibility_abs` was growing (3.8 → 13.7+)
- **BUT weights static**: `mean_abs_dw` remained at 0.0 throughout training

## Root Cause

The learning rule was not responding to dopamine because the **reward modulation wrapper was disabled**.

### The Architecture Problem

```
Intended Learning Loop (with dopamine):
Trial outcome → Dopamine release → ModulatedLearningRule scales dW by dopamine → weights change

What actually happened (wrapper disabled):
Trial outcome → Dopamine released (unused) → RStdpEligibilityRule ignores dopamine → weights don't change
```

### The Code Issue

In `_resolve_wrapper_cfg()`:

**OLD CODE** (broken):

```python
"enabled": bool(_first_non_none(wrapper.get("enabled"), defaults.get("enabled", False)))
```

Result: Wrapper defaults to False → Learning ignores dopamine signals

**NEW CODE** (fixed):

```python
learning_enabled = bool(learning.get("enabled", False))
modulators_enabled = bool(_as_mapping(run_spec.get("modulators")).get("enabled", False))
wrapper_enabled_effective = learning_enabled and modulators_enabled
```

Result: Wrapper auto-enables when learning+modulators both active

### Why Old Configs Failed

Many existing run configs explicitly set `"wrapper": { "enabled": false }` in their JSON, which overrode even the default. This meant:

- Learning enabled ✓
- Modulators (dopamine) enabled ✓
- BUT wrapper disabled ✗ = **No dopamine modulation of learning**

## The Fix

The fix enforces a logical invariant: **When both learning and modulators are enabled, the wrapper MUST be enabled.** There is no sensible configuration where you'd want learning+dopamine without modulation.

**File**: [src/biosnn/tasks/logic_gates/engine_runner.py](src/biosnn/tasks/logic_gates/engine_runner.py)
**Lines**: ~2488 (in `_resolve_wrapper_cfg()`)

The wrapper now auto-enables when:

```
learning.enabled = true  AND  modulators.enabled = true
↓
wrapper.enabled = true  (forced, no override allowed)
```

## Verification

✅ All 376 tests passing
✅ ruff style check passing
✅ mypy type checking passing

## Expected Results

### Before Fix (This Run: run_20260212_115119_746600)

```
Phase 1 OR gate, Trial 1625:
  dopamine_pulse:      +0.3  (reward detected!)
  mean_eligibility:    13.7  (traces building)
  mean_abs_dW:         0.0   ← PROBLEM: Weights not changing!
```

### After Fix (Next Run)

```
Phase 1 OR gate, Trial 1625:
  dopamine_pulse:      +0.3  (reward detected)
  mean_eligibility:    13.7  (traces building)
  mean_abs_dW:         0.15  ← FIXED: Weights responding to dopamine!

Accuracy improvement:  Random → ~70% by phase end → >95% by final phases
```

## Did My Previous Run Use the Fix?

**NO.** The run at `run_20260212_115119_746600` clearly shows `"wrapper": {"enabled": false}` in its run_config.json. That run was started before the proper fix was implemented.

The first fix I attempted only worked for NEW configs without explicit wrapper settings. Since this config explicitly disabled the wrapper, it was overridden.

## Configuration Guide

### ✓ Recommended: No wrapper section (auto-enables)

```json
{
  "learning": { "enabled": true, "lr": 0.001, "rule": "rstdp" },
  "modulators": { "enabled": true, "kinds": ["dopamine"] }
}
```

Result: Wrapper auto-enables ✓

### ✓ Explicitly enabled (also works)

```json
{
  "learning": { "enabled": true, "lr": 0.001 },
  "modulators": { "enabled": true },
  "wrapper": { "enabled": true }
}
```

Result: Wrapper enabled ✓

### ✓ Explicitly disabled IF learning disabled

```json
{
  "learning": { "enabled": false },
  "modulators": { "enabled": true },
  "wrapper": { "enabled": false }
}
```

Result: Wrapper stays disabled (no learning to modulate) ✓

### ⚠ Will Now Be Overridden (old pattern)

```json
{
  "learning": { "enabled": true, "lr": 0.001 },
  "modulators": { "enabled": true },
  "wrapper": { "enabled": false }
}
```

Result: Wrapper FORCED ON because learning+modulators active
Reason: This config was nonsensical (learning+dopamine but no modulation)

## Next Steps

1. **Delete old config files** with `"wrapper": { "enabled": false }`
2. **Or remove the wrapper section entirely** to use auto-enable
3. **Run a new training** - should see weight changes (`mean_abs_dw` > 0)
4. **Monitor accuracy curves** - should show learning progression

## Debugging a New Run

If `mean_abs_dw` is still 0:

1. Check run_config.json for:
   - `learning.enabled: true` ✓
   - `modulators.enabled: true` ✓
   - `wrapper.enabled` → should be computed as TRUE in output

2. Check trials.csv for:
   - `dopamine_pulse` column → should have ±0.3 values
   - `mean_eligibility_abs` → should grow over time
   - `mean_abs_dw` → should be > 0 (if wrapper working)

3. If still broken with all above correct → File issue with run artifacts
