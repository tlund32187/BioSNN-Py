from __future__ import annotations

import json
from pathlib import Path

import pytest

from biosnn.runners import cli

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize(
    ("demo_name", "extra_args"),
    [
        ("propagation_impulse", []),
        ("delay_impulse", ["--delay_steps", "4"]),
        ("learning_gate", ["--learning_lr", "0.08"]),
        ("dopamine_plasticity", ["--da_amount", "1.2", "--da_step", "8"]),
    ],
)
def test_cli_feature_demos_smoke(
    monkeypatch, tmp_path: Path, demo_name: str, extra_args: list[str]
) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / demo_name
    args = cli._parse_args(
        [
            "--demo",
            demo_name,
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "20",
            "--no-open",
            *extra_args,
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    required = (
        "topology.json",
        "neuron.csv",
        "tap.csv",
        "spikes.csv",
        "synapse.csv",
        "weights.csv",
        "metrics.csv",
    )
    for name in required:
        path = run_dir / name
        assert path.exists(), f"Missing artifact {name} for demo {demo_name}"
        assert path.stat().st_size > 0, f"Empty artifact {name} for demo {demo_name}"


def test_cli_network_demo_csr_layout_cpu(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "network_csr_cpu"
    args = cli._parse_args(
        [
            "--demo",
            "network",
            "--mode",
            "fast",
            "--device",
            "cpu",
            "--steps",
            "10",
            "--fused-layout",
            "csr",
            "--n-in",
            "4",
            "--n-hidden",
            "8",
            "--n-out",
            "2",
            "--input-pops",
            "1",
            "--hidden-layers",
            "1",
            "--hidden-pops-per-layer",
            "1",
            "--output-pops",
            "1",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    topology = run_dir / "topology.json"
    assert topology.exists()
    assert topology.stat().st_size > 0


def test_cli_ring_dtype_cuda_smoke(monkeypatch, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    run_dir = tmp_path / "ring_dtype_cuda"
    args = cli._parse_args(
        [
            "--demo",
            "delay_impulse",
            "--mode",
            "fast",
            "--device",
            "cuda",
            "--steps",
            "10",
            "--ring-dtype",
            "float16",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    topology = run_dir / "topology.json"
    assert topology.exists()
    assert topology.stat().st_size > 0


def test_cli_vision_demo_writes_optional_monitor_artifacts(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "vision_demo"
    args = cli._parse_args(
        [
            "--demo",
            "vision",
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "20",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    required = (
        "topology.json",
        "neuron.csv",
        "metrics.csv",
        "modgrid.json",
        "receptors.csv",
        "vision.json",
    )
    for name in required:
        path = run_dir / name
        assert path.exists(), f"Missing artifact {name} for vision demo"
        assert path.stat().st_size > 0, f"Empty artifact {name} for vision demo"


def test_cli_feature_demo_can_disable_monitors(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "propagation_no_monitors"
    args = cli._parse_args(
        [
            "--demo",
            "propagation_impulse",
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "20",
            "--no-monitors",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    required = (
        "topology.json",
        "run_config.json",
        "run_features.json",
        "run_status.json",
    )
    for name in required:
        path = run_dir / name
        assert path.exists(), f"Missing artifact {name} for propagation_impulse"
        assert path.stat().st_size > 0, f"Empty artifact {name} for propagation_impulse"

    monitor_outputs = (
        "neuron.csv",
        "tap.csv",
        "spikes.csv",
        "synapse.csv",
        "weights.csv",
        "metrics.csv",
    )
    for name in monitor_outputs:
        assert not (run_dir / name).exists(), f"Monitor artifact should be disabled: {name}"


def test_cli_compiled_dashboard_mode(monkeypatch, tmp_path: Path) -> None:
    """--compiled flag forces compiled_mode in dashboard mode and records it in run_features.json."""
    pytest.importorskip("torch")
    run_dir = tmp_path / "compiled_dashboard"
    args = cli._parse_args(
        [
            "--demo",
            "propagation_impulse",
            "--mode",
            "dashboard",
            "--compiled",
            "--device",
            "cpu",
            "--steps",
            "20",
            "--no-open",
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    features_path = run_dir / "run_features.json"
    assert features_path.exists(), "run_features.json should be written"
    features = json.loads(features_path.read_text(encoding="utf-8"))
    assert features.get("compiled_mode") is True, (
        "compiled_mode should be True in run_features.json"
    )

    # Artifacts should still be written (dashboard mode).
    for name in ("topology.json", "neuron.csv", "metrics.csv"):
        path = run_dir / name
        assert path.exists(), f"Missing artifact {name}"
        assert path.stat().st_size > 0, f"Empty artifact {name}"
