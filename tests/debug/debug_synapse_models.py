#!/usr/bin/env python3
"""Trace if receptor_mode affects synapse learning properties."""

import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from biosnn.tasks.logic_gates.topologies import build_logic_gate_ff  # noqa: E402


def compare_synapse_models() -> None:
    """Check if receptor_mode changes synapse model type."""
    print("=" * 80)
    print("TEST: Synapse models with different receptor modes")
    print("=" * 80)

    configs: list[dict[str, Any]] = [
        {
            "name": "exc_only (working)",
            "run_spec": {"synapse": {"receptor_mode": "exc_only"}},
        },
        {
            "name": "ei_ampa_nmda_gabaa_gabab (broken)",
            "run_spec": {"synapse": {"receptor_mode": "ei_ampa_nmda_gabaa_gabab"}},
        },
    ]

    for cfg in configs:
        print(f"\nConfiguration: {cfg['name']}")
        engine, topology, handles = build_logic_gate_ff(
            gate="xnor",
            device="cpu",
            seed=123,
            run_spec=cfg["run_spec"],
        )

        # Check all projections
        print("  Projections:")
        for proj in topology.projections:
            n_edges = int(proj.topology.pre_idx.numel())
            print(f"    {proj.name}: {n_edges} edges, {proj.pre} -> {proj.post}")

        print(f"\n  Total projections: {len(topology.projections)}")

        # Calculate total edges
        total_edges = sum(int(p.topology.pre_idx.numel()) for p in topology.projections)
        print(f"  Total edges: {total_edges}")


if __name__ == "__main__":
    compare_synapse_models()
