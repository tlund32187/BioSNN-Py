"""Direct test of receptor profile with known values."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from biosnn.contracts.neurons import Compartment


def test_receptor_profile_directly():
    """Create a synapse, feed known input, check receptor output."""
    from biosnn.contracts.synapses import ReceptorKind
    from biosnn.synapses.dynamics.delayed_sparse_matmul import (
        DelayedSparseMatmulState,
        DelayedSparseMatmulSynapse,
        _apply_receptor_profile_if_enabled,
        _ensure_receptor_profile_state,
    )
    from biosnn.synapses.receptors.profile import ReceptorProfile

    device = torch.device("cuda")
    n_post = 16
    dt = 0.001

    # Create receptor profile matching exc_only
    profile = ReceptorProfile(
        kinds=(ReceptorKind.AMPA, ReceptorKind.NMDA),
        tau={ReceptorKind.AMPA: 0.005, ReceptorKind.NMDA: 0.100},
        mix={ReceptorKind.AMPA: 1.0, ReceptorKind.NMDA: 0.35},
        sign={ReceptorKind.AMPA: 1.0, ReceptorKind.NMDA: 1.0},
    )

    # Create model params
    from biosnn.synapses.dynamics.delayed_sparse_matmul import DelayedSparseMatmulParams as Params

    params = Params(receptor_profile=profile)
    model = DelayedSparseMatmulSynapse(params)

    # Create minimal state
    state = DelayedSparseMatmulState(
        weights=torch.zeros(1, device=device),
        cursor=0,
        post_ring=None,
        post_event_ring=None,
        post_out=None,
    )

    # Create a minimal topology mock
    class FakeTopo:
        target_compartment = Compartment.DENDRITE
        target_compartments = None
        meta = {}

    topology = FakeTopo()

    # Ensure receptor state is initialized
    _ensure_receptor_profile_state(
        model=model,
        state=state,
        topology=topology,
        dt=dt,
        n_post=n_post,
        device=device,
        weights_dtype=torch.float32,
    )

    print(f"receptor_decay: {state.receptor_decay}")
    print(f"receptor_mix: {state.receptor_mix}")
    print(f"receptor_sign: {state.receptor_sign}")
    print(f"receptor_sign_values: {state.receptor_sign_values}")
    print(
        f"receptor_g shapes: {
            {k.name: v.shape for k, v in state.receptor_g.items()} if state.receptor_g else None
        }"
    )
    print(
        f"receptor_g dtype: {
            {k.name: v.dtype for k, v in state.receptor_g.items()} if state.receptor_g else None
        }"
    )
    print()

    # Create drive_by_comp with known nonzero value
    drive_val = 2.0e-7
    drive_by_comp = {
        Compartment.DENDRITE: torch.zeros(n_post, device=device, dtype=torch.float32),
    }
    # Set some values
    drive_by_comp[Compartment.DENDRITE][3] = drive_val
    drive_by_comp[Compartment.DENDRITE][7] = drive_val

    print(
        f"BEFORE receptor: drive sum={drive_by_comp[Compartment.DENDRITE].abs().sum().item():.4e}"
    )
    print(f"BEFORE receptor: g sum={state.receptor_g[Compartment.DENDRITE].abs().sum().item():.4e}")

    # Apply receptor profile
    _apply_receptor_profile_if_enabled(
        model=model,
        state=state,
        topology=topology,
        drive_by_comp=drive_by_comp,
        dt=dt,
        n_post=n_post,
        device=device,
        weights_dtype=torch.float32,
        inputs_meta=None,
    )

    print(f"AFTER receptor: drive sum={drive_by_comp[Compartment.DENDRITE].abs().sum().item():.4e}")
    print(f"AFTER receptor: g sum={state.receptor_g[Compartment.DENDRITE].abs().sum().item():.4e}")
    print(f"AFTER receptor: g values:\n{state.receptor_g[Compartment.DENDRITE][:, :8]}")
    print(f"AFTER receptor: drive values:\n{drive_by_comp[Compartment.DENDRITE][:8]}")

    # Apply a second time with zero input (should show decay)
    drive_by_comp2 = {
        Compartment.DENDRITE: torch.zeros(n_post, device=device, dtype=torch.float32),
    }
    _apply_receptor_profile_if_enabled(
        model=model,
        state=state,
        topology=topology,
        drive_by_comp=drive_by_comp2,
        dt=dt,
        n_post=n_post,
        device=device,
        weights_dtype=torch.float32,
        inputs_meta=None,
    )
    print(
        f"\nAfter 2nd call (zero input): drive sum={drive_by_comp2[Compartment.DENDRITE].abs().sum().item():.4e}"
    )
    print(f"After 2nd call: g sum={state.receptor_g[Compartment.DENDRITE].abs().sum().item():.4e}")


if __name__ == "__main__":
    test_receptor_profile_directly()
