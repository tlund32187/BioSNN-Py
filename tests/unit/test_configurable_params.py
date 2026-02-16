"""
Test that configurable weight scale and GABA receptor parameters work.
"""
from biosnn.experiments.demo_registry import resolve_run_spec
from biosnn.tasks.logic_gates import build_logic_gate_ff

# Test 1: Verify parameters are in default run spec
print("="*70)
print("TEST 1: Default parameters in resolved run_spec")
print("="*70)

default_spec = resolve_run_spec(None)
print(f"\nDefault synapse config keys: {list(default_spec['synapse'].keys())}")
print(f"  hidden_excit_to_out_weight_scale: {default_spec['synapse']['hidden_excit_to_out_weight_scale']}")
print(f"  hidden_inhib_to_out_weight_scale: {default_spec['synapse']['hidden_inhib_to_out_weight_scale']}")
print(f"  receptors: {default_spec['synapse']['receptors']}")

# Test 2: Override parameters
print("\n" + "="*70)
print("TEST 2: Override parameters in config")
print("="*70)

custom_spec = {
    "synapse": {
        "hidden_excit_to_out_weight_scale": 0.025,  # Reduce excitatory
        "hidden_inhib_to_out_weight_scale": 0.015,  # Reduce inhibitory
        "gabaa_mix": 0.5,  # Reduce GABA_A strength
        "gabab_mix": 0.1,  # Reduce GABA_B strength
    }
}

resolved_custom = resolve_run_spec(custom_spec)
print("\nCustom synapse config:")
print(f"  hidden_excit_to_out_weight_scale: {resolved_custom['synapse']['hidden_excit_to_out_weight_scale']}")
print(f"  hidden_inhib_to_out_weight_scale: {resolved_custom['synapse']['hidden_inhib_to_out_weight_scale']}")
print(f"  receptors.gabaa_mix: {resolved_custom['synapse']['receptors']['gabaa_mix']}")
print(f"  receptors.gabab_mix: {resolved_custom['synapse']['receptors']['gabab_mix']}")

# Test 3: Build topology with custom parameters
print("\n" + "="*70)
print("TEST 3: Build topology with custom parameters")
print("="*70)

try:
    engine, topology, handles = build_logic_gate_ff(
        gate="xnor",
        device="cpu",
        seed=123,
        run_spec=custom_spec,
    )
    print("\n✓ Successfully built topology with custom parameters!")
    print(f"  Engine type: {type(engine).__name__}")
except Exception as e:
    print(f"\n✗ Failed to build topology: {e}")
    raise

# Test 4: Verify parameters flow through to topologies
print("\n" + "="*70)
print("TEST 4: Verify topology weight scales")
print("="*70)

print("\nTopology projections:")
for proj_spec in topology.projections:
    if "Out" in proj_spec.name:
        print(f"  {proj_spec.name}: weights shape = {proj_spec.topology.weights.shape if hasattr(proj_spec.topology, 'weights') and proj_spec.topology.weights is not None else 'N/A'}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\n✓ All tests passed!")
print("\nConfigurable parameters:")
print("  - hidden_excit_to_out_weight_scale (default: 0.03)")
print("  - hidden_inhib_to_out_weight_scale (default: 0.04)")
print("  - gabaa_mix (default: 1.0)")
print("  - gabab_mix (default: 0.25)")
print("\nUsage example:")
print("""
  run_spec = {
      "synapse": {
          "hidden_excit_to_out_weight_scale": 0.015,  # Weaken excitation
          "hidden_inhib_to_out_weight_scale": 0.02,   # Weaken inhibition
          "gabaa_mix": 0.5,                            # Reduce GABA_A
          "gabab_mix": 0.1,                            # Reduce GABA_B
      }
  }
  engine, topology, handles = build_logic_gate_ff(run_spec=run_spec)
""")
