#!/usr/bin/env python
"""Debug script to check if releases_fn is being called and returning data."""
import sys

sys.path.insert(0, 'src')

import torch

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.neuromodulators import GlobalScalarField, GlobalScalarParams


def test_field_sampling():
    """Test if modulator field processes releases correctly."""
    print("=" * 80)
    print("TEST: Global Scalar Field Release and Sampling")
    print("=" * 80)
    
    # Create a global scalar field (simpler than grid)
    field = GlobalScalarField(
        kinds=(ModulatorKind.DOPAMINE,),
        params=GlobalScalarParams(decay_tau=0.05),
    )
    
    # Initialize state
    state = field.init_state(ctx=None)
    print(f"Initial field state: {state}")
    
    # Create releases
    dopamine_tensor = torch.tensor([0.5], dtype=torch.float32)
    releases = [
        ModulatorRelease(
            kind=ModulatorKind.DOPAMINE,
            amount=dopamine_tensor,
            positions=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        )
    ]
    print(f"\nCreated release: dopamine={dopamine_tensor.item():.4f}")
    
    # Step the field with releases
    print("\nCalling field.step() with releases...")
    new_state = field.step(state, releases=releases, dt=0.001, t=0.0, ctx=None)
    print(f"Field state after step: {new_state}")
    
    # Sample at a single position
    test_pos = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    print(f"\nSampling at position {test_pos.tolist()}...")
    sampled = field.sample_at(new_state, positions=test_pos, kind=ModulatorKind.DOPAMINE, ctx=None)
    print(f"Sampled dopamine: {sampled.tolist()}")
    
    if sampled[0].item() < 0.01:
        print("❌ ERROR: Field returned near-zero dopamine after release!")
        return False
    else:
        print("✅ Field is sampling dopamine correctly")
    
    print("\n" + "=" * 80)
    print("TEST: Multiple steps with continuous sampling")
    print("=" * 80)
    
    # Reset
    state = field.init_state(ctx=None)
    
    # Apply dopamine for 3 steps
    for step in range(5):
        if step < 3:
            releases = [
                ModulatorRelease(
                    kind=ModulatorKind.DOPAMINE,
                    amount=torch.tensor([0.3], dtype=torch.float32),
                    positions=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
                )
            ]
        else:
            releases = []
        
        state = field.step(state, releases=releases, dt=0.001, t=float(step) * 0.001, ctx=None)
        sampled = field.sample_at(state, positions=test_pos, kind=ModulatorKind.DOPAMINE, ctx=None)
        print(f"Step {step}: dopamine={sampled[0].item():.6f}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_field_sampling()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
