"""Generate a population topology payload for the dashboard."""

from __future__ import annotations

from pathlib import Path

import torch

from biosnn.contracts.synapses import SynapseTopology
from biosnn.io.dashboard_export import export_population_topology_json


def random_topology(n_pre: int, n_post: int, p: float) -> SynapseTopology:
    mask = torch.rand((n_pre, n_post)) < p
    pre_idx, post_idx = mask.nonzero(as_tuple=True)
    weights = torch.randn(pre_idx.numel()) * 0.1
    delay_steps = torch.randint(1, 6, (pre_idx.numel(),), dtype=torch.int32)
    return SynapseTopology(
        pre_idx=pre_idx.to(dtype=torch.long),
        post_idx=post_idx.to(dtype=torch.long),
        weights=weights.to(dtype=torch.float32),
        delay_steps=delay_steps,
    )


def main() -> None:
    populations = [
        {"name": "Input", "n": 16, "model_name": "glif", "layer": 0},
        {"name": "Hidden", "n": 64, "model_name": "adex_2c", "layer": 1},
        {"name": "Output", "n": 10, "model_name": "glif", "layer": 2},
    ]

    proj_input_hidden = {
        "name": "Input->Hidden",
        "pre": "Input",
        "post": "Hidden",
        "topology": random_topology(16, 64, 0.25),
    }
    proj_hidden_output = {
        "name": "Hidden->Output",
        "pre": "Hidden",
        "post": "Output",
        "topology": random_topology(64, 10, 0.2),
    }

    out_path = Path("docs/dashboard/data/topology.json")
    export_population_topology_json(
        populations,
        [proj_input_hidden, proj_hidden_output],
        path=out_path,
    )

    print("Wrote population topology to", out_path)
    print("Run: python -m http.server")
    print("Open: http://localhost:8000/docs/dashboard/")


if __name__ == "__main__":
    main()
