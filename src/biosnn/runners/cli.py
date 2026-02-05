"""CLI entrypoint for running demo simulations and opening the dashboard."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from collections.abc import Sequence
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

from biosnn.core.torch_utils import require_torch
from biosnn.experiments.demo_minimal import DemoMinimalConfig, run_demo_minimal
from biosnn.experiments.demo_network import DemoNetworkConfig, run_demo_network


def main() -> None:
    args = _parse_args()
    torch = require_torch()

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = _make_run_dir(repo_root / "artifacts")
    device = _resolve_device(torch, args.device)
    steps = args.steps
    dt = args.dt

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Demo: {args.demo}")

    mode = args.mode

    if args.demo == "network":
        input_to_relay_p = args.input_to_relay_p
        relay_to_hidden_p = args.relay_to_hidden_p
        hidden_to_output_p = args.hidden_to_output_p
        if args.p_in_hidden is not None:
            input_to_relay_p = args.p_in_hidden
            relay_to_hidden_p = args.p_in_hidden
        if args.p_hidden_out is not None:
            hidden_to_output_p = args.p_hidden_out
        run_demo_network(
            DemoNetworkConfig(
                out_dir=run_dir,
                mode=mode,
                steps=steps,
                dt=dt,
                seed=args.seed,
                device=device,
                n_in=args.n_in,
                n_hidden=args.n_hidden,
                n_out=args.n_out,
                input_pops=args.input_pops,
                input_depth=args.input_depth,
                hidden_layers=args.hidden_layers,
                hidden_pops_per_layer=args.hidden_pops_per_layer,
                output_pops=args.output_pops,
                input_cross=args.input_cross,
                input_to_relay_p=input_to_relay_p,
                input_to_relay_weight_scale=args.input_to_relay_weight_scale,
                relay_to_hidden_p=relay_to_hidden_p,
                relay_to_hidden_weight_scale=args.relay_to_hidden_weight_scale,
                hidden_to_output_p=hidden_to_output_p,
                hidden_to_output_weight_scale=args.hidden_to_output_weight_scale,
                input_skip_to_hidden=args.input_skip_to_hidden,
                input_skip_p=args.input_skip_p,
                input_skip_weight_scale=args.input_skip_weight_scale,
                relay_cross=args.relay_cross,
                relay_cross_p=args.relay_cross_p,
                relay_cross_weight_scale=args.relay_cross_weight_scale,
                relay_lateral=args.relay_lateral,
                hidden_lateral=args.hidden_lateral,
                weight_init=args.weight_init,
                input_drive=args.input_drive,
                drive_monitor=args.drive_monitor,
            )
        )
    else:
        run_demo_minimal(
            DemoMinimalConfig(
                out_dir=run_dir,
                mode=mode,
                n_neurons=args.n,
                p_connect=args.p,
                steps=steps,
                dt=dt,
                seed=args.seed,
                device=device,
            )
        )

    if not _should_launch_dashboard(mode):
        print("Fast mode selected; skipping dashboard server.")
        return

    port = _find_port(args.port)
    _start_dashboard_server(repo_root, port)
    url = _build_dashboard_url(port, run_dir, repo_root, args.refresh_ms)
    print(f"Dashboard URL: {url}")

    if not args.no_open:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BioSNN demo and open dashboard")
    parser.add_argument(
        "--demo",
        choices=["minimal", "network"],
        default=_default_demo(),
        help="which demo to run",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="run device; CUDA defaults to DelayedSparseMatmulSynapse in the demo",
    )
    parser.add_argument(
        "--mode",
        choices=["dashboard", "fast"],
        default="dashboard",
        help="run mode: dashboard (full artifacts) or fast (throughput-oriented)",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--n", type=int, default=100, help="number of neurons")
    parser.add_argument("--p", type=float, default=0.05, help="connection probability")
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-in", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--n-out", type=int, default=10)
    parser.add_argument("--input-pops", type=int, default=2)
    parser.add_argument("--input-depth", type=int, default=2)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--hidden-pops-per-layer", type=int, default=1)
    parser.add_argument("--output-pops", type=int, default=1)
    parser.add_argument("--input-cross", action="store_true")
    parser.add_argument("--p-in-hidden", type=float, default=None, help="legacy: use --relay-to-hidden-p")
    parser.add_argument("--p-hidden-out", type=float, default=None, help="legacy: use --hidden-to-output-p")
    parser.add_argument("--input-to-relay-p", type=float, default=0.35)
    parser.add_argument("--input-to-relay-weight-scale", type=float, default=1.5)
    parser.add_argument("--relay-to-hidden-p", type=float, default=0.20)
    parser.add_argument("--relay-to-hidden-weight-scale", type=float, default=1.0)
    parser.add_argument("--hidden-to-output-p", type=float, default=0.20)
    parser.add_argument("--hidden-to-output-weight-scale", type=float, default=1.0)
    parser.add_argument("--input-skip-to-hidden", action="store_true")
    parser.add_argument("--input-skip-p", type=float, default=0.03)
    parser.add_argument("--input-skip-weight-scale", type=float, default=0.5)
    parser.add_argument("--relay-cross", action="store_true")
    parser.add_argument("--relay-cross-p", type=float, default=0.05)
    parser.add_argument("--relay-cross-weight-scale", type=float, default=0.2)
    parser.add_argument("--relay-lateral", action="store_true")
    parser.add_argument("--hidden-lateral", action="store_true")
    parser.add_argument("--weight-init", type=float, default=0.05)
    parser.add_argument("--input-drive", type=float, default=1.0)
    parser.add_argument("--drive-monitor", action="store_true", help="write drive.csv diagnostics")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--refresh-ms", type=int, default=1200)
    return parser.parse_args(argv)


def _make_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}"
    if run_dir.exists():
        run_dir = base / f"run_{stamp}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_device(torch: Any, device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_demo() -> str:
    try:
        from biosnn.simulation.engine import TorchNetworkEngine

        _ = TorchNetworkEngine
        return "network"
    except Exception:
        return "minimal"


def _should_launch_dashboard(mode: str) -> bool:
    return mode.lower().strip() != "fast"


def _start_dashboard_server(repo_root: Path, port: int) -> None:
    def handler(*args, **kwargs):
        return SimpleHTTPRequestHandler(*args, directory=str(repo_root), **kwargs)

    server = ThreadingHTTPServer(("localhost", port), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def _build_dashboard_url(port: int, run_dir: Path, repo_root: Path, refresh_ms: int) -> str:
    base = f"http://localhost:{port}/docs/dashboard/"
    query = {
        "topology": _dashboard_param(repo_root, run_dir, "topology.json"),
        "neuron": _dashboard_param(repo_root, run_dir, "neuron.csv"),
        "synapse": _dashboard_param(repo_root, run_dir, "synapse.csv"),
        "spikes": _dashboard_param(repo_root, run_dir, "spikes.csv"),
        "metrics": _dashboard_param(repo_root, run_dir, "metrics.csv"),
        "refresh": str(refresh_ms),
    }
    weights_path = run_dir / "weights.csv"
    if weights_path.exists():
        query["weights"] = _dashboard_param(repo_root, run_dir, "weights.csv")

    query_str = "&".join(f"{key}={quote(value, safe='/:')}" for key, value in query.items())
    return f"{base}?{query_str}"


def _dashboard_param(repo_root: Path, run_dir: Path, filename: str) -> str:
    try:
        rel = run_dir.relative_to(repo_root)
    except ValueError:
        return (run_dir / filename).as_posix()
    return (Path("/") / rel / filename).as_posix()


def _find_port(requested: int | None) -> int:
    if requested is not None:
        if _port_available(requested):
            return requested
        raise RuntimeError(f"Port {requested} is not available")

    for port in range(8000, 8011):
        if _port_available(port):
            return port
    raise RuntimeError("No available port in range 8000-8010")


def _port_available(port: int) -> bool:
    try:
        server = ThreadingHTTPServer(("localhost", port), SimpleHTTPRequestHandler)
        server.server_close()
        return True
    except OSError:
        return False


if __name__ == "__main__":
    main()
