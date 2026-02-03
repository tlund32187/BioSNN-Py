# AdEx 2-Compartment Model (adex_2c)

This document describes the parameter units, sign conventions, inputs, and state
for the AdEx 2-compartment neuron model (soma + dendrite) used in BioSNN.

## Units (SI)
- Volt (V): v_* (e_l_*, v_t, v_reset, v_spike)
- Second (s): tau_*, refrac_period, spike_hold_time
- Farad (F): c_* (capacitance)
- Siemens (S): g_* (conductance)
- Ampere (A): drive currents

## Sign conventions
- Positive drive depolarizes the membrane.
- Leak current is: -g_l * (v - e_l).
- Coupling current into soma is: g_c * (v_dend - v_soma).
- Coupling current into dendrite is: g_c * (v_soma - v_dend).
- Adaptation current w subtracts from soma current (hyperpolarizing).
- v_spike is a detection threshold (not a physical peak voltage).

## State (all shape [N])
- v_soma: soma membrane potential
- v_dend: dendrite membrane potential
- w: adaptation current
- refrac_left: refractory time remaining (seconds)
- spike_hold_left: spike hold time remaining (seconds)

## Inputs
- NeuronInputs.drive[Compartment.SOMA]: soma drive current (A)
- NeuronInputs.drive[Compartment.DENDRITE]: dendrite drive current (A)

Shape requirement: each drive is a 1D tensor of length N. The model can validate
shapes if ctx.extras["validate_shapes"] is True (default).

## Step behavior notes
- Euler integration is used.
- Spikes are detected when v_soma >= v_spike (and not refractory).
- After a spike: v_soma resets to v_reset and w is incremented by b.
- refrac_left is set to refrac_period; spike_hold_left is set to spike_hold_time.

## Runtime flags (StepContext.extras)
- "no_grad": bool (default True if not training)
- "validate_shapes": bool (default True)
- "assert_finite": bool (default False)
- "require_inputs_on_device": bool (default False; raise if drive is on a different device)
- "const_cache_max": int (default 32; <=0 disables dt-constant caching)

Note: "const_cache_max" is shared by GLIF and AdEx to cap dt-keyed constant caches.
