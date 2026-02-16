#!/usr/bin/env python3
"""Debug script to compare edge positions between two receptor modes."""

import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from biosnn.tasks.logic_gates.topologies import build_logic_gate_ff  # noqa: E402


def compare_edge_positions() -> None:
    """Build topologies with both receptor modes and compare edge positions."""
    print("=" * 80)
    print("TEST: Edge positions with different receptor modes")
    print("=" * 80)

    # Test both receptor modes
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

    topologies = {}
    for cfg in configs:
        print(f"\nBuilding topology: {cfg['name']}")
        engine, topology, handles = build_logic_gate_ff(
            gate="xnor",
            device="cpu",
            seed=123,
            run_spec=cfg["run_spec"],
        )
        topologies[cfg["name"]] = topology

        # Print population info
        print("  Populations:")
        for pop in topology.populations:
            if pop.positions is not None:
                print(
                    f"    {pop.name}: n={pop.n}, positions shape={pop.positions.shape}, "
                    f"min={pop.positions.min().item():.3f}, max={pop.positions.max().item():.3f}"
                )
            else:
                print(f"    {pop.name}: n={pop.n}, NO POSITIONS (None)")

        # Print projection info
        print("  Projections (first 3):")
        for proj in topology.projections[:3]:
            print(f"    {proj.name}: {proj.pre} -> {proj.post}")
            print(f"      topology.post_idx shape: {proj.topology.post_idx.shape}")
            print(f"      topology.post_pos: {proj.topology.post_pos}")
            if proj.topology.post_pos is not None:
                print(
                    f"        shape: {proj.topology.post_pos.shape}, "
                    f"min: {proj.topology.post_pos.min().item():.3f}, "
                    f"max: {proj.topology.post_pos.max().item():.3f}"
                )

    # Compare topologies
    print("\n" + "-" * 80)
    print("COMPARISON")
    print("-" * 80)

    topo1 = topologies[configs[0]["name"]]
    topo2 = topologies[configs[1]["name"]]

    print(f"Number of populations: {len(topo1.populations)} vs {len(topo2.populations)}")
    print(f"Number of projections: {len(topo1.projections)} vs {len(topo2.projections)}")

    # Check if positions are the same
    for p1, p2 in zip(topo1.populations, topo2.populations, strict=False):
        if p1.name != p2.name:
            print(f"❌ Population mismatch: {p1.name} vs {p2.name}")
            continue
        if p1.positions is None and p2.positions is None:
            print(f"✅ Population {p1.name}: both have None positions")
        elif p1.positions is None or p2.positions is None:
            print(f"❌ Population {p1.name}: positions mismatch (one None)")
        else:
            diff = (p1.positions - p2.positions).abs().max().item()
            if diff < 1e-6:
                print(f"✅ Population {p1.name}: positions identical (diff={diff:.2e})")
            else:
                print(f"⚠️ Population {p1.name}: positions different (diff={diff:.6f})")

    # Check if edge positions would be the same
    print("\nEdge topology comparison (sampling projections):")
    for i, (proj1, proj2) in enumerate(zip(topo1.projections, topo2.projections, strict=False)):
        if proj1.name != proj2.name or proj1.post != proj2.post:
            print(f"  Projection mismatch at {i}")
            continue

        # Check if post_pos differs
        post_pos_1 = proj1.topology.post_pos
        post_pos_2 = proj2.topology.post_pos

        if post_pos_1 is None and post_pos_2 is None:
            scenario = "both None"
        elif post_pos_1 is None or post_pos_2 is None:
            scenario = "ONE is None (PROBLEM!)"
        else:
            diff = (post_pos_1 - post_pos_2).abs().max().item()
            scenario = f"diff={diff:.2e}"

        print(f"  {proj1.name}: post_pos {scenario}")


if __name__ == "__main__":
    compare_edge_positions()
