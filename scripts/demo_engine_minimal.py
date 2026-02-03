"""Minimal end-to-end engine demo (GLIF + delayed synapses)."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from biosnn.biophysics.models._torch_utils import require_torch
from biosnn.experiments.demo_minimal import DemoMinimalConfig, run_demo_minimal


def main() -> None:
    args = _parse_args()
    torch = require_torch()

    repo_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "artifacts" / f"demo_minimal_{stamp}"

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DemoMinimalConfig(
        out_dir=out_dir,
        n_neurons=args.n,
        p_connect=args.p,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        device=device,
        neuron_model=args.neuron_model,
    )

    summary = run_demo_minimal(cfg)

    print(f"Wrote artifacts to: {summary['out_dir']}")
    url = _build_dashboard_url(out_dir, repo_root, port=8000)
    print("To view the dashboard, start a local server from the repo root, e.g.:")
    print("  python -m http.server 8000")
    print(f"Then open: {url}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal BioSNN demo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--n", type=int, default=100, help="number of neurons")
    parser.add_argument("--p", type=float, default=0.05, help="connection probability")
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--neuron-model", choices=["glif", "adex_2c"], default="glif")
    return parser.parse_args()


def _build_dashboard_url(out_dir: Path, repo_root: Path, *, port: int) -> str:
    base = f"http://localhost:{port}/docs/dashboard/"
    query = {
        "topology": _dashboard_param(repo_root, out_dir, "topology.json"),
        "neuron": _dashboard_param(repo_root, out_dir, "neuron.csv"),
        "synapse": _dashboard_param(repo_root, out_dir, "synapse.csv"),
        "spikes": _dashboard_param(repo_root, out_dir, "spikes.csv"),
        "metrics": _dashboard_param(repo_root, out_dir, "metrics.csv"),
    }
    weights_path = out_dir / "weights.csv"
    if weights_path.exists():
        query["weights"] = _dashboard_param(repo_root, out_dir, "weights.csv")

    query_str = "&".join(f"{key}={quote(value, safe='/:')}" for key, value in query.items())
    return f"{base}?{query_str}"


def _dashboard_param(repo_root: Path, out_dir: Path, filename: str) -> str:
    try:
        rel = out_dir.relative_to(repo_root)
    except ValueError:
        return (out_dir / filename).as_posix()
    return (Path("..") / ".." / rel / filename).as_posix()


if __name__ == "__main__":
    main()
