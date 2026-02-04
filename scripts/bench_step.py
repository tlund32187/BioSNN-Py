"""Lightweight step benchmark for TorchNetworkEngine."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("torch is required to run this benchmark") from exc

try:
    import biosnn  # noqa: F401
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "biosnn is not installed. Run: pip install -e \".[torch]\" from the repo root."
    ) from exc

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import build_bipartite_erdos_renyi_topology
from biosnn.connectivity.sparse_rebuild import rebuild_sparse_delay_mats
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.synapses import ISynapseModel
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)


class _NoOpMonitor:
    name = "noop"

    def __init__(self) -> None:
        self._sum = 0.0

    def on_step(self, event) -> None:
        if event.scalars:
            value = event.scalars.get("spike_count_total", 0.0)
            try:
                self._sum += float(value)
            except Exception:
                return

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((pct / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[max(0, min(idx, len(values_sorted) - 1))]


def _build_engine(
    device: str,
    *,
    backend: str,
    fast_mode: bool,
    compiled_mode: bool,
    seed: int | None,
) -> TorchNetworkEngine:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    torch_device = torch.device(device)
    n_in = 64
    n_hidden = 128
    n_out = 32

    input_pop = PopulationSpec(name="Input", model=GLIFModel(), n=n_in)
    hidden_pop = PopulationSpec(name="Hidden", model=AdEx2CompModel(), n=n_hidden)
    output_pop = PopulationSpec(name="Output", model=GLIFModel(), n=n_out)

    pre_pos = torch.rand((n_in, 3), device=torch_device)
    hid_pos = torch.rand((n_hidden, 3), device=torch_device)
    out_pos = torch.rand((n_out, 3), device=torch_device)

    topo_in_hidden = build_bipartite_erdos_renyi_topology(
        n_pre=n_in,
        n_post=n_hidden,
        p=0.2,
        device=device,
        dt=1e-3,
        pre_pos=pre_pos,
        post_pos=hid_pos,
        weight_init=0.05,
    )
    topo_hidden_out = build_bipartite_erdos_renyi_topology(
        n_pre=n_hidden,
        n_post=n_out,
        p=0.2,
        device=device,
        dt=1e-3,
        pre_pos=hid_pos,
        post_pos=out_pos,
        weight_init=0.05,
    )

    synapse: ISynapseModel
    if backend == "sparse":
        synapse = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=0.05))
    elif backend == "event":
        synapse = DelayedCurrentSynapse(
            DelayedCurrentParams(
                init_weight=0.05,
                event_driven=True,
                adaptive_event_driven=False,
            )
        )
    elif backend == "dense":
        synapse = DelayedCurrentSynapse(
            DelayedCurrentParams(
                init_weight=0.05,
                event_driven=False,
                adaptive_event_driven=False,
            )
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'")

    projections = [
        ProjectionSpec(
            name="Input->Hidden",
            synapse=synapse,
            topology=topo_in_hidden,
            pre="Input",
            post="Hidden",
        ),
        ProjectionSpec(
            name="Hidden->Output",
            synapse=synapse,
            topology=topo_hidden_out,
            pre="Hidden",
            post="Output",
        ),
    ]

    engine = TorchNetworkEngine(
        populations=[input_pop, hidden_pop, output_pop],
        projections=projections,
        fast_mode=fast_mode,
        compiled_mode=compiled_mode,
    )
    return engine


def run_benchmark_case(
    case_name: str,
    build_engine_fn: Callable[[], TorchNetworkEngine],
    *,
    steps: int,
    warmup: int,
    device: str,
    fast_mode: bool,
    compiled_mode: bool,
    monitors_on: bool,
    rebuild_sparse_every: int,
    seed: int | None,
) -> dict[str, Any]:
    engine = build_engine_fn()
    if monitors_on:
        engine.attach_monitors([_NoOpMonitor()])

    engine.reset(config=SimulationConfig(dt=1e-3, device=device, seed=seed))

    for _ in range(warmup):
        engine.step()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    step_times: list[float] = []
    rebuild_total_ms = 0.0
    rebuild_count = 0
    start = time.perf_counter()
    for _ in range(steps):
        step_start = time.perf_counter()
        engine.step()
        step_times.append((time.perf_counter() - step_start) * 1000.0)
        if rebuild_sparse_every > 0 and (engine._step % rebuild_sparse_every) == 0:
            rebuild_start = time.perf_counter()
            for proj in engine._proj_specs:
                reqs = None
                if hasattr(proj.synapse, "compilation_requirements"):
                    try:
                        reqs = proj.synapse.compilation_requirements()
                    except Exception:
                        reqs = None
                if not isinstance(reqs, Mapping):
                    continue
                if not bool(reqs.get("needs_sparse_delay_mats", False)):
                    continue
                build_bucket = bool(reqs.get("needs_bucket_edge_mapping", False))
                rebuild_sparse_delay_mats(
                    proj.topology,
                    device=engine._device,
                    dtype=engine._dtype,
                    build_bucket_edge_mapping=build_bucket,
                )
            rebuild_total_ms += (time.perf_counter() - rebuild_start) * 1000.0
            rebuild_count += 1

    total = time.perf_counter() - start
    if device == "cuda":
        torch.cuda.synchronize()

    steps_per_sec = steps / total if total > 0 else 0.0
    avg_ms = (total / steps) * 1000.0 if steps else 0.0
    p50_ms = _percentile(step_times, 50.0)
    p95_ms = _percentile(step_times, 95.0)

    metrics: dict[str, Any] = {
        "backend": case_name,
        "steps": steps,
        "warmup": warmup,
        "device": device,
        "fast_mode": fast_mode,
        "compiled_mode": compiled_mode,
        "monitors": monitors_on,
        "rebuild_sparse_every": rebuild_sparse_every,
        "rebuild_sparse_count": rebuild_count,
        "rebuild_sparse_total_ms": rebuild_total_ms,
        "rebuild_sparse_avg_ms": (rebuild_total_ms / rebuild_count) if rebuild_count else 0.0,
        "steps_per_sec": steps_per_sec,
        "avg_step_ms": avg_ms,
        "p50_step_ms": p50_ms,
        "p95_step_ms": p95_ms,
        "total_time_s": total,
    }

    if device == "cuda":
        metrics["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        metrics["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)

    return metrics


def _write_csv_row(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TorchNetworkEngine step throughput.")
    parser.add_argument("--steps", type=int, default=5000, help="Timed steps to run.")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup steps to run before timing.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run on.")
    parser.add_argument("--fast-mode", action="store_true", help="Enable fast_mode.")
    parser.add_argument("--compiled-mode", action="store_true", help="Enable compiled_mode.")
    parser.add_argument(
        "--backends",
        type=str,
        default="all",
        choices=["all", "dense", "event", "sparse"],
        help="Synapse backend(s) to benchmark.",
    )
    parser.add_argument(
        "--monitors",
        type=str,
        default="off",
        choices=["off", "on"],
        help="Enable simple monitors.",
    )
    parser.add_argument("--out", type=str, default="bench_results.json", help="Output JSON/CSV path.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for topology.")
    parser.add_argument(
        "--rebuild-sparse-every",
        type=int,
        default=0,
        help="Rebuild sparse delay matrices every N steps (0 disables).",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    backends = ["dense", "event", "sparse"] if args.backends == "all" else [args.backends]
    results: list[dict[str, Any]] = []

    def _make_builder(name: str) -> Callable[[], TorchNetworkEngine]:
        def _builder() -> TorchNetworkEngine:
            return _build_engine(
                args.device,
                backend=name,
                fast_mode=bool(args.fast_mode),
                compiled_mode=bool(args.compiled_mode),
                seed=args.seed,
            )

        return _builder

    for backend in backends:
        metrics = run_benchmark_case(
            backend,
            _make_builder(backend),
            steps=args.steps,
            warmup=args.warmup,
            device=args.device,
            fast_mode=bool(args.fast_mode),
            compiled_mode=bool(args.compiled_mode),
            monitors_on=args.monitors == "on",
            rebuild_sparse_every=int(args.rebuild_sparse_every) if backend == "sparse" else 0,
            seed=args.seed,
        )
        results.append(metrics)

    print("backend  steps/sec  avg_ms  p50_ms  p95_ms  peak_alloc_mb  peak_res_mb")
    for metrics in results:
        peak_alloc = metrics.get("cuda_peak_allocated_mb", 0.0)
        peak_res = metrics.get("cuda_peak_reserved_mb", 0.0)
        print(
            f"{metrics['backend']:<7} "
            f"{metrics['steps_per_sec']:>9.2f} "
            f"{metrics['avg_step_ms']:>7.4f} "
            f"{metrics['p50_step_ms']:>7.4f} "
            f"{metrics['p95_step_ms']:>7.4f} "
            f"{peak_alloc:>14.2f} "
            f"{peak_res:>12.2f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    if out_path.suffix.lower() == ".csv":
        for metrics in results:
            row = {
                "timestamp": timestamp,
                "device": args.device,
                "backend": metrics["backend"],
                "steps": metrics["steps"],
                "warmup": metrics["warmup"],
                "steps_per_sec": metrics["steps_per_sec"],
                "avg_ms": metrics["avg_step_ms"],
                "p50_ms": metrics["p50_step_ms"],
                "p95_ms": metrics["p95_step_ms"],
                "peak_alloc_mb": metrics.get("cuda_peak_allocated_mb", 0.0),
                "peak_res_mb": metrics.get("cuda_peak_reserved_mb", 0.0),
            }
            _write_csv_row(out_path, row)
        json_path = out_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": {
                        "device": args.device,
                        "steps": args.steps,
                        "warmup": args.warmup,
                        "fast_mode": bool(args.fast_mode),
                        "compiled_mode": bool(args.compiled_mode),
                        "monitors": args.monitors,
                        "seed": args.seed,
                    },
                    "results": results,
                },
                handle,
                indent=2,
            )
    else:
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": {
                        "device": args.device,
                        "steps": args.steps,
                        "warmup": args.warmup,
                        "fast_mode": bool(args.fast_mode),
                        "compiled_mode": bool(args.compiled_mode),
                        "monitors": args.monitors,
                        "seed": args.seed,
                    },
                    "results": results,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
