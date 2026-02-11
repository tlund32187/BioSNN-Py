from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from biosnn.runners import cli

pytestmark = pytest.mark.unit

def test_cli_demo_flag_defaults():
    args = cli._parse_args([])
    assert args.demo == cli._default_demo()
    assert args.mode == "dashboard"
    assert args.monitors is True
    assert args.torch_threads == "auto"
    assert args.torch_interop_threads == "auto"
    assert args.set_omp_env is False
    assert args.monitor_safe_defaults is True
    assert args.monitor_neuron_sample == 512
    assert args.monitor_edge_sample == 20000
    assert args.allow_cuda_monitor_sync is None
    assert args.parallel_compile == "auto"
    assert args.parallel_compile_workers == "auto"
    assert args.parallel_compile_torch_threads == 1
    assert args.profile is False
    assert args.profile_steps == 20
    assert args.delay_steps == 3
    assert args.learning_lr == 0.1
    assert args.da_amount == 1.0
    assert args.da_step == 10
    assert args.logic_neuron_model == "adex_3c"
    assert args.fused_layout == "auto"
    assert args.ring_dtype is None
    assert args.receptor_state_dtype is None
    assert args.ring_strategy == "dense"
    assert args.store_sparse_by_delay is None
    assert args.large_network_safety is False
    assert args.vision_compile is False
    assert args.vision_max_side == 64
    assert args.vision_max_elements == 16384
    assert args.modgrid_max_side == 64
    assert args.modgrid_max_elements == 16384

    args = cli._parse_args(["--demo", "minimal"])
    assert args.demo == "minimal"

    args = cli._parse_args(["--demo", "network"])
    assert args.demo == "network"
    args = cli._parse_args(["--demo", "vision"])
    assert args.demo == "vision"
    args = cli._parse_args(["--demo", "propagation_impulse"])
    assert args.demo == "propagation_impulse"
    args = cli._parse_args(["--demo", "delay_impulse"])
    assert args.demo == "delay_impulse"
    args = cli._parse_args(["--demo", "learning_gate"])
    assert args.demo == "learning_gate"
    args = cli._parse_args(["--demo", "dopamine_plasticity"])
    assert args.demo == "dopamine_plasticity"
    args = cli._parse_args(["--demo", "logic_curriculum"])
    assert args.demo == "logic_curriculum"
    args = cli._parse_args(["--demo", "logic_and"])
    assert args.demo == "logic_and"
    args = cli._parse_args(["--demo", "logic_or"])
    assert args.demo == "logic_or"
    args = cli._parse_args(["--demo", "logic_xor"])
    assert args.demo == "logic_xor"

    args = cli._parse_args(["--mode", "fast"])
    assert args.mode == "fast"
    args = cli._parse_args(
        [
            "--torch-threads",
            "4",
            "--torch-interop-threads",
            "2",
            "--set-omp-env",
            "--no-monitors",
            "--no-monitor-safe-defaults",
            "--monitor-neuron-sample",
            "256",
            "--monitor-edge-sample",
            "10000",
            "--allow-cuda-monitor-sync",
            "--parallel-compile",
            "on",
            "--parallel-compile-workers",
            "2",
            "--parallel-compile-torch-threads",
            "3",
            "--profile",
            "--profile-steps",
            "12",
            "--delay_steps",
            "7",
            "--learning_lr",
            "0.2",
            "--da_amount",
            "2.5",
            "--da_step",
            "9",
            "--logic-gate",
            "xor",
            "--logic-learning-mode",
            "surrogate",
            "--logic-neuron-model",
            "lif_3c",
            "--logic-sim-steps-per-trial",
            "12",
            "--logic-sampling-method",
            "random_balanced",
            "--logic-curriculum-gates",
            "or,and,xor",
            "--logic-curriculum-replay-ratio",
            "0.6",
            "--logic-debug",
            "--logic-debug-every",
            "5",
            "--fused-layout",
            "csr",
            "--ring-dtype",
            "bfloat16",
            "--receptor-state-dtype",
            "float16",
            "--ring-strategy",
            "event_bucketed",
            "--store-sparse-by-delay",
            "true",
            "--large-network-safety",
            "--vision-compile",
            "--vision-max-side",
            "48",
            "--vision-max-elements",
            "12000",
            "--modgrid-max-side",
            "40",
            "--modgrid-max-elements",
            "8000",
        ]
    )
    assert args.torch_threads == "4"
    assert args.torch_interop_threads == "2"
    assert args.set_omp_env is True
    assert args.monitors is False
    assert args.monitor_safe_defaults is False
    assert args.monitor_neuron_sample == 256
    assert args.monitor_edge_sample == 10000
    assert args.allow_cuda_monitor_sync is True
    assert args.parallel_compile == "on"
    assert args.parallel_compile_workers == "2"
    assert args.parallel_compile_torch_threads == 3
    assert args.profile is True
    assert args.profile_steps == 12
    assert args.delay_steps == 7
    assert args.learning_lr == 0.2
    assert args.da_amount == 2.5
    assert args.da_step == 9
    assert args.logic_gate == "xor"
    assert args.logic_learning_mode == "surrogate"
    assert args.logic_neuron_model == "lif_3c"
    assert args.logic_sim_steps_per_trial == 12
    assert args.logic_sampling_method == "random_balanced"
    assert args.logic_curriculum_gates == "or,and,xor"
    assert args.logic_curriculum_replay_ratio == 0.6
    assert args.logic_debug is True
    assert args.logic_debug_every == 5
    assert args.fused_layout == "csr"
    assert args.ring_dtype == "bfloat16"
    assert args.receptor_state_dtype == "float16"
    assert args.ring_strategy == "event_bucketed"
    assert args.store_sparse_by_delay is True
    assert args.large_network_safety is True
    assert args.vision_compile is True
    assert args.vision_max_side == 48
    assert args.vision_max_elements == 12000
    assert args.modgrid_max_side == 40
    assert args.modgrid_max_elements == 8000

    args = cli._parse_args(["--ring-dtype", "none"])
    assert args.ring_dtype is None
    args = cli._parse_args(["--receptor-state-dtype", "none"])
    assert args.receptor_state_dtype is None

    args = cli._parse_args(["--store-sparse-by-delay", "false"])
    assert args.store_sparse_by_delay is False


def test_large_network_safety_overrides() -> None:
    args = cli._parse_args(
        [
            "--large-network-safety",
            "--monitor-neuron-sample",
            "5000",
            "--monitor-edge-sample",
            "200000",
            "--vision-max-side",
            "300",
            "--vision-max-elements",
            "999999",
            "--modgrid-max-side",
            "400",
            "--modgrid-max-elements",
            "888888",
        ]
    )
    assert args.allow_cuda_monitor_sync is None
    assert args.receptor_state_dtype is None

    cli._apply_large_network_safety_overrides(args=args, device="cuda")

    assert args.monitor_safe_defaults is True
    assert args.allow_cuda_monitor_sync is False
    assert args.receptor_state_dtype == "float16"
    assert args.monitor_neuron_sample == 512
    assert args.monitor_edge_sample == 20000
    assert args.vision_max_side == 64
    assert args.vision_max_elements == 16384
    assert args.modgrid_max_side == 64
    assert args.modgrid_max_elements == 16384


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
    assert "weights" in params
    assert "modgrid" in params
    assert "receptors" in params
    assert "vision" in params
    assert params["weights"] == ["none"]
    assert params["modgrid"] == ["none"]
    assert params["receptors"] == ["none"]
    assert params["vision"] == ["none"]
