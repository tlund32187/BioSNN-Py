#!/usr/bin/env python3
"""Minimal test to trace dopamine through modulation and learning."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import torch  # noqa: E402

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease  # noqa: E402
from biosnn.contracts.simulation import SimulationConfig  # noqa: E402
from biosnn.tasks.logic_gates.topologies import build_logic_gate_ff  # noqa: E402


def test_dopamine_through_learning():
    """Build engine and run steps with dopamine to see if it reaches learning."""
    print("=" * 80)
    print("TEST: Dopamine through learning pipeline")
    print("=" * 80)

    # Build with broken receptor_mode
    print("\nBuilding engine with ei_ampa_nmda_gabaa_gabab...")
    engine, topology, handles = build_logic_gate_ff(
        gate="xnor",
        device="cpu",
        seed=123,
        run_spec={"synapse": {"receptor_mode": "ei_ampa_nmda_gabaa_gabab"}},
    )
    
    print("Engine created")
    print(f"Learning projections available: {list(engine._proj_states.keys())}")
    
    # Create a simple releases_fn that always gives dopamine
    dopamine_amount = torch.tensor([0.5])
    dopamine_positions = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)  # At hidden layer
    
    step_count = [0]
    def releases_fn(t, step, ctx):
        step_count[0] += 1
        if step_count[0] <= 3:  # First 3 steps
            return [
                ModulatorRelease(
                    kind=ModulatorKind.DOPAMINE,
                    positions=dopamine_positions,
                    amount=dopamine_amount,
                )
            ]
        return []
    
    # Override the engine's releases_fn
    engine._releases_fn = releases_fn
    
    # Reset engine
    engine.reset(config=SimulationConfig(dt=0.001, seed=123))
    
    # Run some steps
    print("\nRunning 5 engine steps with dopamine in first 3...")
    try:
        for i in range(5):
            engine.step()
            print(f"Step {i}: step_count={step_count[0]}")
    except Exception as e:
        print(f"Error during stepping: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test complete. Check stderr for dopamine debug output.")

if __name__ == "__main__":
    test_dopamine_through_learning()
