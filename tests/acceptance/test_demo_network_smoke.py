from __future__ import annotations

import pytest
pytestmark = pytest.mark.acceptance


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


def test_demo_network_cuda_uses_sparse_backend(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import biosnn.experiments.demo_network as demo_network

    calls = {"sparse": 0, "current": 0}

    class SpySparse(demo_network.DelayedSparseMatmulSynapse):
        def __init__(self, *args, **kwargs):
            calls["sparse"] += 1
            super().__init__(*args, **kwargs)

    class SpyCurrent(demo_network.DelayedCurrentSynapse):
        def __init__(self, *args, **kwargs):
            calls["current"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(demo_network, "DelayedSparseMatmulSynapse", SpySparse)
    monkeypatch.setattr(demo_network, "DelayedCurrentSynapse", SpyCurrent)

    out_dir = tmp_path / "network_demo_cuda"
    cfg = DemoNetworkConfig(
        out_dir=out_dir,
        steps=2,
        dt=1e-3,
        seed=123,
        device="cuda",
        n_in=4,
        n_hidden=8,
        n_out=3,
        relay_to_hidden_p=0.5,
        hidden_to_output_p=0.5,
        input_drive=1.0,
    )

    summary = run_demo_network(cfg)

    assert summary["device"] == "cuda"
    assert calls["sparse"] > 0
    assert calls["current"] == 0
