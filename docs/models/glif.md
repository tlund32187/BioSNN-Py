# GLIF Model (glif)

This document describes units, sign conventions, inputs, and state for the GLIF
model used in BioSNN.

## Units (SI)
- Volt (V): v_* (v_rest, v_reset, v_thresh0)
- Second (s): tau_m, refrac_period, spike_hold_time, theta_tau
- Ohm (Ohm): r_m (membrane resistance)
- Ampere (A): drive currents

## Sign conventions
- Positive drive depolarizes the membrane.
- Leak term uses v_rest as the equilibrium (dv/dt = (-(v - v_rest) + r_m * drive) / tau_m).
- v_thresh0 is the base threshold; theta adds a nonnegative offset.

## State (all shape [N])
- v_soma: soma membrane potential
- refrac_left: refractory time remaining (seconds)
- spike_hold_left: spike hold time remaining (seconds)
- theta: dynamic threshold offset

## Inputs
- NeuronInputs.drive[Compartment.SOMA]: soma drive current (A)

Shape requirement: drive is a 1D tensor of length N. The model can validate
shapes if ctx.extras["validate_shapes"] is True (default).

## Step behavior notes
- Euler integration is used.
- Spikes are detected when v_soma >= (v_thresh0 + theta) and not refractory.
- After a spike: v_soma resets to v_reset and refrac_left is set.
- spike_hold_left keeps spikes high for spike_hold_time.

## Runtime flags (StepContext.extras)
- "no_grad": bool (default True if not training)
- "validate_shapes": bool (default True)
- "assert_finite": bool (default False)
- "const_cache_max": int (default 32; <=0 disables dt-constant caching)
