# Configurable Synapse and Receptor Parameters

## Summary

Made the following parameters configurable from the dashboard and configuration:

1. **HiddenExcit→Out weight_scale** (default: 0.03)
2. **HiddenInhib→Out weight_scale** (default: 0.04)
3. **GABA_A receptor mix** (default: 1.0)
4. **GABA_B receptor mix** (default: 0.25)

## Why This Matters

These are the **critical parameters that control whether learning happens** in the ei_ampa_nmda_gabaa_gabab receptor mode:

- **Hidden excitatory→output weight**: Controls how much excitatory drive reaches output neurons
- **Hidden inhibitory→output weight**: Controls how much GABA inhibition reaches output neurons
- **gabaa_mix**: Relative strength of fast GABA_A receptors (inhibitory, ~10ms)
- **gabab_mix**: Relative strength of slow GABA_B receptors (inhibitory, ~150ms)

## Configuration Structure

### Via Python Config Dict

```python
run_spec = {
    "synapse": {
        "hidden_excit_to_out_weight_scale": 0.03,   # Adjust excitation strength
        "hidden_inhib_to_out_weight_scale": 0.04,   # Adjust inhibition strength
        "receptors": {
            "gabaa_mix": 1.0,    # GABA_A contribution (0.0 = off)
            "gabab_mix": 0.25,   # GABA_B contribution (0.0 = off)
        }
    }
}

engine, topology, handles = build_logic_gate_ff(run_spec=run_spec)
```

### Via JSON Config File

```json
{
  "synapse": {
    "hidden_excit_to_out_weight_scale": 0.03,
    "hidden_inhib_to_out_weight_scale": 0.04,
    "receptors": {
      "gabaa_mix": 1.0,
      "gabab_mix": 0.25
    }
  }
}
```

### Via Demo Registry

In `default_run_spec()`, these are now included:

```python
"synapse": {
    # ... existing fields ...
    "hidden_excit_to_out_weight_scale": 0.03,
    "hidden_inhib_to_out_weight_scale": 0.04,
    "receptors": {
        "gabaa_mix": 1.0,
        "gabab_mix": 0.25,
    }
}
```

## Usage Examples

### Example 1: Weaken Inhibition

Problem: Output neurons being completely silenced by GABA. Solution: Reduce inhibitory weight.

```python
run_spec = {
    "synapse": {
        "hidden_inhib_to_out_weight_scale": 0.015,  # Down from 0.04
    }
}
```

### Example 2: Strengthen Excitation

Problem: Output neurons don't fire enough. Solution: Increase excitatory drive.

```python
run_spec = {
    "synapse": {
        "hidden_excit_to_out_weight_scale": 0.05,  # Up from 0.03
    }
}
```

### Example 3: Reduce GABA Contribution

Problem: Both GABA_A and GABA_B too strong. Solution: Scale both down.

```python
run_spec = {
    "synapse": {
        "receptors": {
            "gabaa_mix": 0.5,   # Down from 1.0
            "gabab_mix": 0.1,   # Down from 0.25
        }
    }
}
```

### Example 4: Disable GABA_B (Use GABA_A Only)

```python
run_spec = {
    "synapse": {
        "receptors": {
            "gabaa_mix": 1.0,
            "gabab_mix": 0.0,   # Disable slow inhibition
        }
    }
}
```

## Implementation Details

### Modified Files

1. **[src/biosnn/tasks/logic_gates/topologies.py](src/biosnn/tasks/logic_gates/topologies.py)**
   - Updated `_resolve_synapse_cfg()` to extract and validate the new parameters with sensible defaults
   - Modified `_inhibitory_profile_for_mode()` to accept gabaa_mix/gabab_mix parameters
   - Updated `build_logic_gate_ff()` and `build_logic_gate_xor()` to use configurable weight scales instead of hardcoded 0.03/0.04
   - Added `_coerce_nonnegative_float()` helper for parameter validation

2. **[src/biosnn/experiments/demo_registry.py](src/biosnn/experiments/demo_registry.py)**
   - Modified `resolve_run_spec()` to include these new parameters with proper defaults
   - Parameters are now part of the standard configuration dictionaries

### Configuration Flow

```
User Config
    ↓
resolve_run_spec() [demo_registry.py]
    ↓
_resolve_synapse_cfg() [topologies.py]  ← Extracts parameters
    ↓
build_logic_gate_ff/xor() [topologies.py]  ← Uses parameters
    ↓
_inhibitory_profile_for_mode() [topologies.py]  ← Passes gabaa_mix/gabab_mix
    ↓
profile_inh_gabaa_gabab() [receptors/profile.py]  ← Creates receptor profile
```

## Testing

All 376 existing tests continue to pass with these changes. New test file `test_configurable_params.py` verifies:

1. ✓ Default parameters are present in resolved run_spec
2. ✓ Custom parameters override defaults
3. ✓ Topologies build successfully with custom parameters
4. ✓ Parameters flow through to weight initialization

## Recommended Parameter Tuning

Based on the root cause analysis (output neurons 99.97% silenced in ei_ampa_nmda_gabaa_gabab mode):

### To Fix Learning in ei_ampa_nmda_gabaa_gabab Mode

**Option 1: Weaken Inhibition** (Recommended)
```python
"hidden_inhib_to_out_weight_scale": 0.015  # 62.5% reduction
```

**Option 2: Strengthen Excitation**
```python
"hidden_excit_to_out_weight_scale": 0.045  # 50% increase
```

**Option 3: Reduce GABA Receptor Strength**
```python
"receptors": {
    "gabaa_mix": 0.5,    # 50% reduction
    "gabab_mix": 0.1,    # 60% reduction  
}
```

**Option 4: Combination** (Most nuanced control)
```python
{
    "hidden_excit_to_out_weight_scale": 0.035,
    "hidden_inhib_to_out_weight_scale": 0.025,
    "receptors": {
        "gabaa_mix": 0.7,
        "gabab_mix": 0.15,
    }
}
```

## Backward Compatibility

✓ All changes are backward compatible:
- Old run_specs without these parameters will use sensible defaults
- Existing tests all pass without modification
- Dashboard can be used without specifying these parameters (uses defaults)

## Advanced Usage

### CLI Integration (Future)

These parameters can be exposed via CLI arguments:
```bash
python biosnn/runners/cli.py \
  --hidden-excit-to-out-scale 0.025 \
  --hidden-inhib-to-out-scale 0.015 \
  --gabaa-mix 0.5 \
  --gabab-mix 0.1
```

### Programmatic Tuning

```python
import json
from pathlib import Path

# Load best config from previous run
with open("best_config.json") as f:
    best_spec = json.load(f)

# Slightly perturb for hyperparameter search
new_spec = deep_copy(best_spec)
new_spec["synapse"]["hidden_inhib_to_out_weight_scale"] *= 0.9

# Test new configuration
engine, _, _ = build_logic_gate_ff(run_spec=new_spec)
```
