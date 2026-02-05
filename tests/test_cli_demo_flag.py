from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

from biosnn.runners import cli


def test_cli_demo_flag_defaults():
    args = cli._parse_args([])
    assert args.demo == cli._default_demo()
    assert args.mode == "dashboard"
    assert args.torch_threads == "auto"
    assert args.torch_interop_threads == "auto"
    assert args.set_omp_env is False
    assert args.monitor_safe_defaults is True
    assert args.monitor_neuron_sample == 512
    assert args.monitor_edge_sample == 20000
    assert args.profile is False
    assert args.profile_steps == 20

    args = cli._parse_args(["--demo", "minimal"])
    assert args.demo == "minimal"

    args = cli._parse_args(["--demo", "network"])
    assert args.demo == "network"

    args = cli._parse_args(["--mode", "fast"])
    assert args.mode == "fast"
    args = cli._parse_args(
        [
            "--torch-threads",
            "4",
            "--torch-interop-threads",
            "2",
            "--set-omp-env",
            "--no-monitor-safe-defaults",
            "--monitor-neuron-sample",
            "256",
            "--monitor-edge-sample",
            "10000",
            "--profile",
            "--profile-steps",
            "12",
        ]
    )
    assert args.torch_threads == "4"
    assert args.torch_interop_threads == "2"
    assert args.set_omp_env is True
    assert args.monitor_safe_defaults is False
    assert args.monitor_neuron_sample == 256
    assert args.monitor_edge_sample == 10000
    assert args.profile is True
    assert args.profile_steps == 12


def test_cli_dashboard_mode_gate():
    assert cli._should_launch_dashboard("dashboard") is True
    assert cli._should_launch_dashboard("fast") is False


def test_cli_threading_flags_no_crash(monkeypatch, tmp_path):
    args = cli._parse_args(
        [
            "--demo",
            "minimal",
            "--mode",
            "fast",
            "--device",
            "cpu",
            "--torch-threads",
            "2",
            "--torch-interop-threads",
            "1",
            "--set-omp-env",
        ]
    )

    class _DummyTorch:
        def __init__(self):
            self.calls = []

        def set_num_threads(self, value):
            self.calls.append(("threads", value))

        def set_num_interop_threads(self, value):
            self.calls.append(("interop", value))

    dummy = _DummyTorch()

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "require_torch", lambda: dummy)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: tmp_path)
    monkeypatch.setattr(cli, "run_demo_minimal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    assert ("threads", 2) in dummy.calls
    assert ("interop", 1) in dummy.calls


def test_cli_dashboard_url_params(tmp_path: Path):
    repo_root = tmp_path
    run_dir = repo_root / "artifacts" / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "topology.json").write_text("{}")
    (run_dir / "neuron.csv").write_text("")
    (run_dir / "synapse.csv").write_text("")
    (run_dir / "spikes.csv").write_text("")
    (run_dir / "metrics.csv").write_text("")

    url = cli._build_dashboard_url(8000, run_dir, repo_root, refresh_ms=1200)
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert params["refresh"] == ["1200"]
    assert "topology" in params
    assert "neuron" in params
    assert "synapse" in params
    assert "spikes" in params
    assert "metrics" in params
    assert "weights" not in params
