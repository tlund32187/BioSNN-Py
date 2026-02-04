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

    if args.demo == "network":
        run_demo_network(
            DemoNetworkConfig(
                out_dir=run_dir,
                steps=steps,
                dt=dt,
                seed=args.seed,
                device=device,
                n_in=args.n_in,
                n_hidden=args.n_hidden,
                n_out=args.n_out,
                p_in_hidden=args.p_in_hidden,
                p_hidden_out=args.p_hidden_out,
                weight_init=args.weight_init,
                input_drive=args.input_drive,
            )
        )
    else:
        run_demo_minimal(
            DemoMinimalConfig(
                out_dir=run_dir,
                n_neurons=args.n,
                p_connect=args.p,
                steps=steps,
                dt=dt,
                seed=args.seed,
                device=device,
            )
        )

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
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--n", type=int, default=100, help="number of neurons")
    parser.add_argument("--p", type=float, default=0.05, help="connection probability")
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-in", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--n-out", type=int, default=10)
    parser.add_argument("--p-in-hidden", type=float, default=0.25)
    parser.add_argument("--p-hidden-out", type=float, default=0.25)
    parser.add_argument("--weight-init", type=float, default=0.05)
    parser.add_argument("--input-drive", type=float, default=1.0)
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
