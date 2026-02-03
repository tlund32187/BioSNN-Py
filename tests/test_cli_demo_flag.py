from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

from biosnn.runners import cli


def test_cli_demo_flag_defaults():
    args = cli._parse_args([])
    assert args.demo == cli._default_demo()

    args = cli._parse_args(["--demo", "minimal"])
    assert args.demo == "minimal"

    args = cli._parse_args(["--demo", "network"])
    assert args.demo == "network"


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
