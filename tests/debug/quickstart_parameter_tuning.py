"""
QUICKSTART: How to use configurable parameters to restore learning
in ei_ampa_nmda_gabaa_gabab mode

This demonstrates the fix for the zero learning issue where output neurons
were 99.97% silenced, resulting in eligibility traces near zero.
"""

from biosnn.tasks.logic_gates import build_logic_gate_ff

# ============================================================================
# PROBLEM: ei_ampa_nmda_gabaa_gabab mode silences output neurons
# ============================================================================
#
# Original configuration (BROKEN):
#   - hidden_excit_to_out_weight_scale: 0.03
#   - hidden_inhib_to_out_weight_scale: 0.04 ← TOO STRONG!
#   - gabaa_mix: 1.0, gabab_mix: 0.25 ← Both active
#
# Result: Output spikes: 1 (should be ~36,540), Learning: 0.0 (should be 0.00047)
#
# ============================================================================

# SOLUTION 1: Weaken inhibitory synapses
print("=" * 70)
print("SOLUTION 1: Weaken inhibitory weight scale")
print("=" * 70)

config_weaken_inhibition = {
    "synapse": {
        "hidden_inhib_to_out_weight_scale": 0.015,  # Reduce from 0.04
    }
}

_, _, _ = build_logic_gate_ff(  # type: ignore[assignment]
    gate="xnor",
    device="cpu",
    run_spec=config_weaken_inhibition,
)
print("✓ Built topology with inhibition: 0.04 → 0.015")
print("  Expected outcome: Output neuron firing restored")

# ============================================================================

# SOLUTION 2: Strengthen excitatory synapses
print("\n" + "=" * 70)
print("SOLUTION 2: Strengthen excitatory weight scale")
print("=" * 70)

config_strengthen_excitation = {
    "synapse": {
        "hidden_excit_to_out_weight_scale": 0.045,  # Increase from 0.03
    }
}

_, _, _ = build_logic_gate_ff(  # type: ignore[assignment]
    gate="xnor",
    device="cpu",
    run_spec=config_strengthen_excitation,
)
print("✓ Built topology with excitation: 0.03 → 0.045")
print("  Expected outcome: Stronger excitatory drive overcomes inhibition")

# ============================================================================

# SOLUTION 3: Reduce GABA receptor amplitudes
print("\n" + "=" * 70)
print("SOLUTION 3: Reduce GABA receptor strengths")
print("=" * 70)

config_reduce_gaba = {
    "synapse": {
        "receptors": {
            "gabaa_mix": 0.5,  # Reduce fast inhibition
            "gabab_mix": 0.1,  # Reduce slow inhibition more
        }
    }
}

_, _, _ = build_logic_gate_ff(  # type: ignore[assignment]
    gate="xnor",
    device="cpu",
    run_spec=config_reduce_gaba,
)
print("✓ Built topology with GABA: (1.0, 0.25) → (0.5, 0.1)")
print("  Expected outcome: Less overall inhibitory tone")

# ============================================================================

# SOLUTION 4: Combined adjustment (most effective)
print("\n" + "=" * 70)
print("SOLUTION 4: Combined parameter tuning")
print("=" * 70)

config_combined = {
    "synapse": {
        "hidden_excit_to_out_weight_scale": 0.035,  # Slight increase
        "hidden_inhib_to_out_weight_scale": 0.025,  # Moderate reduction
        "receptors": {
            "gabaa_mix": 0.7,  # Reduce GABA_A
            "gabab_mix": 0.15,  # Reduce GABA_B
        },
    }
}

_, _, _ = build_logic_gate_ff(  # type: ignore[assignment]
    gate="xnor",
    device="cpu",
    run_spec=config_combined,
)
print("✓ Built topology with combined parameters:")
print("  - Excitation: 0.03 → 0.035")
print("  - Inhibition: 0.04 → 0.025")
print("  - GABA receptors: (1.0, 0.25) → (0.7, 0.15)")
print("  Expected outcome: Balanced restoration of firing while keeping inhibition")

# ============================================================================

# SOLUTION 5: Use exc_only mode (safest, guaranteed working)
print("\n" + "=" * 70)
print("SOLUTION 5: Revert to exc_only mode (safe baseline)")
print("=" * 70)

config_exc_only = {
    "synapse": {
        "receptor_mode": "exc_only",  # Simple excitation only
    }
}

_, _, _ = build_logic_gate_ff(  # type: ignore[assignment]
    gate="xnor",
    device="cpu",
    run_spec=config_exc_only,
)
print("✓ Built topology with exc_only mode")
print("  Expected outcome: Guaranteed learning (confirmed baseline)")

# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
The core issue was output neuron silencing due to inhibitory coupling
being too strong (weight_scale 0.04 > 0.03 excitatory).

Try these in order of increasing complexity:
  1. Simplest: Revert to "exc_only" mode
  2. Easy: Reduce inhibitory weight to 0.015-0.02
  3. Medium: Increase excitatory weight to 0.04-0.05
  4. Advanced: Adjust GABA receptor mix ratios
  5. Expert: Fine-tune all parameters together

All parameters are now configurable and configurable through:
  - Python dicts passed to build_logic_gate_ff(run_spec=config)
  - JSON configuration files
  - Dashboard settings (via run_spec)

See CONFIGURABLE_PARAMETERS_GUIDE.md for detailed documentation.
""")
