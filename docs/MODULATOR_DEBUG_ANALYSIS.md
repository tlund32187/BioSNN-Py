# Dopamine Modulator Flow Analysis

## Problem Statement

Despite wrapper being created and dopamine released, mean_abs_dw still = 0.0

## Flow Trace

### 1. Trial Completion

- `engine_runner.py` line 558: `_queue_trial_feedback_releases()` is called
  - This queues dopamine into `engine_build.pending_releases` dict
  - Example: `pending_releases[ModulatorKind.DOPAMINE] = 1.0` if correct

### 2. Reward Delivery Steps

- `engine_runner.py` line 586: `_run_trial_steps()` called with `reward_delivery_steps`
- Inside loop (line 789): `engine_build.engine.step()` is called

### 3. Engine Step (compiled_mode=True)

- `torch_network_engine.py` line 869: `_step_compiled()` method

#### Order of Operations in _step_compiled

1. **Line 869-880**: Modulator subsystem step

   ```python
   if self._mod_specs:
       mod_by_pop, edge_mods_by_proj = self._modulator_subsystem.step_compiled(
           ...
           releases=self._resolve_releases(),
       )
   ```

2. **Line 1003+**: Learning happens

   ```python
   edge_mods=self._modulator_subsystem.learning_modulators_for_projection(
       edge_mods_by_proj.get(runtime.plan.name)
   )
   ```

### 4. Modulator Release Process

- `torch_network_engine.py` line 1130: `_resolve_releases()` calls releases_fn from engine_runner.py
  - `engine_runner.py` line 1956: `releases_fn()` calls `_build_modulator_releases_for_step()`

- `engine_runner.py` line 1243+: `_build_modulator_releases_for_step()`
  - Gets dopamine amount from pending_releases
  - **KEY: Line 1279** `pending[kind] = 0.0` - **CLEARS THE PENDING DOPAMINE**
  - Creates ModulatorRelease objects and returns them

### 5. Modulator Subsystem Processing

- `subsystems/modulators.py` line 193+: `step_compiled()` processes releases
  - Line 197: `spec.field.step(state, releases=rel, dt=dt, t=t, ctx=ctx)`
    - Field processes the dopamine release internally
  - Line 215+: Samples field at edge positions
    - `sampled = spec.field.sample_at(state, positions=edge_positions, kind=kind, ctx=ctx)`
    - **CRITICAL**: This should produce the dopamine values at each edge position

### 6. Learning Batch Creation

- `torch_network_engine.py` line 1022: `build_learning_batch()` is called
  - Passes `edge_mods` that came from `edge_mods_by_proj`
  - This becomes `batch.modulators` in the learning rule

## Potential Issues

### Issue 1: Timing - When Does Dopamine Get Released?

- Dopamine is released at step N (first step of reward delivery)
- Learning happens at step N
- **ARE THEY HAPPENING IN THE SAME STEP FOR THE SAME EDGES?**
- Need to verify: Is dopamine actually sampled at the correct edge positions?

### Issue 2: Modulator Field State

- Does the modulator field correctly update its internal state with the dopamine release?
- Does `GlobalScalarField.step()` or `GridDiffusion2DField.step()` properly incorporate the release?
- Then does `.sample_at()` return the correct values?

### Issue 3: Edge Position Mismatch

- Are the edge positions same when sampling happens vs when learning batch is built?
- Check: `subsystems/modulators.py` line 216 vs the projection's actual edge positions

### Issue 4: Learning Rule Wrapper

- Is `ModulatedRuleWrapper` actually receiving the modulators in the batch?
- Check: `learning/rules/modulated_wrapper.py` line 87+

### Issue 5: Eligibility Trace

- For R-STDP with eligibility trace, dopamine works through eligibility
- The wrapper needs eligibility trace to be present
- **Is the eligibility trace present during dopamine application?**

## Tests to Check

- `tests/unit/test_logic_gate_engine_runner_modulator_releases.py`
- `tests/unit/test_compiled_edge_modulator_sampling.py`
- `tests/acceptance/test_network_engine_learning.py`

## Key Functions to Verify

1. `ModulatorSubsystem.step_compiled()` - correctly updating edge_mods_by_proj?
2. `ModulatedRuleWrapper.step()` - receiving the dopamine batch?
3. `_release_positions_for_field()` - getting correct positions?
4. Modulator field `.sample_at()` - producing correct values?

## ROOT CAUSE IDENTIFIED

**Problem**: ModulatedRuleWrapper does NOT use dopamine for learning rate scaling!

- The wrapper only uses ACH, NE, and HT for learning rate modulation
- Dopamine is only in extras for monitoring
- The inner RStdpEligibilityRule receives dopamine through batch.modulators but...

**Critical Issue**: If reward_delivery_steps = 0 (or learning doesn't happen during reward window):

- Dopamine never gets released
- Even if released, if learning doesn't happen during reward delivery steps, dopamine has no effect
- Learning only happens every `learn_every` steps, which might not align with reward delivery window

## The Fix

Ensure dopamine delivery happens AFTER trial even if reward_delivery_steps is minimal:

1. **Always run at least ONE reward delivery step when dopamine is queued**
2. **Ensure learning happens during reward delivery** by reducing learn_every or forcing learning
3. **Verify ModulatedRuleWrapper correctly passes batch to inner rule** (already correct)
4. **Ensure ModulatorSubsystem properly samples dopamine at edge positions** (appears correct)

## Recommended Implementation

1. Change _discard_pending_releases logic: If dopamine is queued, force at least 1 reward step
2. During reward delivery window, ensure learning is enabled (learn_every check)
3. Add logging to verify dopamine reaches the learning batch
