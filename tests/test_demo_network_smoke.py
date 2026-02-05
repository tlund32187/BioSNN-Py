from __future__ import annotations

import pytest

from biosnn.experiments.demo_network import DemoNetworkConfig, run_demo_network


def test_demo_network_smoke(tmp_path):
    pytest.importorskip("torch")
    out_dir = tmp_path / "network_demo"
    cfg = DemoNetworkConfig(
        out_dir=out_dir,
        steps=5,
        dt=1e-3,
        seed=123,
        device="cpu",
        n_in=4,
        n_hidden=8,
        n_out=3,
        relay_to_hidden_p=0.5,
        hidden_to_output_p=0.5,
        input_drive=1.0,
    )

    summary = run_demo_network(cfg)

    assert summary["out_dir"] == out_dir
    assert (out_dir / "topology.json").exists()
    assert (out_dir / "neuron.csv").exists()
    assert (out_dir / "synapse.csv").exists()
    assert (out_dir / "spikes.csv").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "weights.csv").exists()
