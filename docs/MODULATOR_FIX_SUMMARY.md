# Dopamine Modulator Fix Summary

## Problem Statement

**Mean_abs_dw = 0.0 despite wrapper being created and dopamine being released**

The curriculum engine (engine_runner.py) was not properly delivering dopamine to the learning system because:

1. If `reward_delivery_steps` was 0 or not configured, dopamine was immediately discarded
2. dopamine was queued in `pending_releases` but never got a chance to be converted to `ModulatorRelease` objects
3. Without reward delivery steps, the `releases_fn` was never called, so the modulator field never processed the dopamine

## Root Cause Analysis

### Flow Without Fix

```
1. Trial ends correctly
2. _queue_trial_feedback_releases() queues dopamine: pending_releases[DOPAMINE] = 1.0
3. Check: if reward_delivery_steps > 0?
   - NO → Call _discard_pending_releases() → dopamine = 0.0 ✗
   - YES → Run reward_delivery_steps with engine.step()
4. During first engine.step():
   - releases_fn() calls _build_modulator_releases_for_step()
   - Converts pending dopamine to ModulatorRelease objects
   - Sets pending[DOPAMINE] = 0.0 (clears for future steps)
5. ModulatorSubsystem processes releases
6. Learning happens with dopamine in batch ✓
```

### Why mean_abs_dw = 0

- If reward_delivery_steps = 0 → dopamine discarded immediately
- If reward_delivery_steps > 0 but learn_every is too high → learning doesn't happen during reward window
- Result: RStdpEligibilityRule never sees dopamine, so `dw = eligibility * dopamine = eligibility * 0 = 0`

## Solution Implemented

### Changes to engine_runner.py (lines ~560-600)

**File**: `src/biosnn/tasks/logic_gates/engine_runner.py`

#### Before

```python
dopamine_pulse = _queue_trial_feedback_releases(...)
if reward_delivery_steps > 0:
    # Run reward delivery steps
else:
    # Discard dopamine
```

#### After

```python
dopamine_pulse = _queue_trial_feedback_releases(...)

# CRITICAL FIX: Ensure at least 1 step happens when dopamine is queued
# so the modulator system has a chance to process it
has_pending_dopamine = bool(float(engine_build.pending_releases.get(ModulatorKind.DOPAMINE, 0.0)) != 0.0)
effective_reward_steps = max(1, int(reward_delivery_steps)) if has_pending_dopamine else int(reward_delivery_steps)

if effective_reward_steps > 0:
    # Run reward delivery steps with guaranteed minimum of 1 if dopamine exists
    ...
    reward_out_counts, _, sim_step = _run_trial_steps(
        engine_build=engine_build,
        input_drive=reward_input,
        sim_steps_per_trial=effective_reward_steps,  # <-- Use effective_reward_steps
        ...
    )
    ...
    force_steps = (
        min(max(0, int(action_steps)), max(0, int(effective_reward_steps)))  # <-- Updated
        if action_forced
        else 0
    )
else:
    # Discard dopamine only if no pending dopamine
```

## How the Fix Works

1. **Detects Pending Dopamine**: Checks if dopamine was queued after trial evaluation
2. **Ensures Delivery Window**: If dopamine is pending, forces at least 1 reward delivery step
3. **Maintains Flexibility**: If reward_delivery_steps > 0 already, uses that value
4. **Consistent Timing**: Uses same `effective_reward_steps` for both main reward loop and action forcing

## Expected Outcome

With this fix:

- **Dopamine is guaranteed to be released** into the modulator field (at least 1 step)
- **ModulatorSubsystem processes dopamine** through the field
- **Learning batch receives dopamine** in `batch.modulators[DOPAMINE]`
- **RStdpEligibilityRule multiplies eligibility by dopamine**: `dw = e_local * effective_dopamine`
- **mean_abs_dw > 0.0** ✓

## Testing

All existing tests pass:

- ✓ test_logic_gate_engine_runner_modulator_releases (3 tests)
- ✓ test_compiled_edge_modulator_sampling (2 tests)
- ✓ test_network_engine_learning (2 tests)
- ✓ Ruff linting (all checks passed)

## Impact

This fix ensures that dopamine-modulated learning works correctly in the curriculum learning scenario where:

- Trials are evaluated
- Feedback dopamine is queued
- Reward delivery window should grant learning opportunities
- R-STDP with eligibility traces can apply dopamine gating

The fix is **minimal, focused, and maintains backward compatibility** with existing configurations that already have reward_delivery_steps > 0.

## Files Modified

- `src/biosnn/tasks/logic_gates/engine_runner.py` (lines ~560-600)
  - Added `has_pending_dopamine` check
  - Added `effective_reward_steps` calculation
  - Updated all reward_delivery_steps usages in reward window to use effective_reward_steps
