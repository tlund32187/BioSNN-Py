"""Lightweight step benchmark for TorchNetworkEngine."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("torch is required to run this benchmark") from exc

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import build_bipartite_erdos_renyi_topology
from biosnn.contracts.simulation import SimulationConfig
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse


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


def _build_engine(device: str, *, fast_mode: bool, compiled_mode: bool) -> TorchNetworkEngine:
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

    syn_params = DelayedCurrentParams(init_weight=0.05)

    projections = [
        ProjectionSpec(
            name="Input->Hidden",
            synapse=DelayedCurrentSynapse(syn_params),
            topology=topo_in_hidden,
            pre="Input",
            post="Hidden",
        ),
        ProjectionSpec(
            name="Hidden->Output",
            synapse=DelayedCurrentSynapse(syn_params),
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


def _run_bench(
    *,
    steps: int,
    warmup: int,
    device: str,
    fast_mode: bool,
    compiled_mode: bool,
    monitors_on: bool,
) -> dict[str, Any]:
    engine = _build_engine(device, fast_mode=fast_mode, compiled_mode=compiled_mode)
    if monitors_on:
        engine.attach_monitors([_NoOpMonitor()])

    engine.reset(config=SimulationConfig(dt=1e-3, device=device))

    for _ in range(warmup):
        engine.step()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    step_times: list[float] = []
    start = time.perf_counter()
    for _ in range(steps):
        step_start = time.perf_counter()
        engine.step()
        step_times.append((time.perf_counter() - step_start) * 1000.0)
    if device == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - start

    steps_per_sec = steps / total if total > 0 else 0.0
    avg_ms = (total / steps) * 1000.0 if steps else 0.0
    p50_ms = _percentile(step_times, 50.0)
    p95_ms = _percentile(step_times, 95.0)

    metrics: dict[str, Any] = {
        "steps": steps,
        "warmup": warmup,
        "device": device,
        "fast_mode": fast_mode,
        "compiled_mode": compiled_mode,
        "monitors": monitors_on,
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
        "--monitors",
        type=str,
        default="off",
        choices=["off", "on"],
        help="Enable simple monitors.",
    )
    parser.add_argument("--out", type=str, default="bench_results.json", help="Output JSON/CSV path.")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    metrics = _run_bench(
        steps=args.steps,
        warmup=args.warmup,
        device=args.device,
        fast_mode=bool(args.fast_mode),
        compiled_mode=bool(args.compiled_mode),
        monitors_on=args.monitors == "on",
    )

    summary = (
        f"device={metrics['device']} steps={metrics['steps']} "
        f"fast_mode={metrics['fast_mode']} compiled_mode={metrics['compiled_mode']} "
        f"steps/sec={metrics['steps_per_sec']:.2f} avg_ms={metrics['avg_step_ms']:.4f} "
        f"p50_ms={metrics['p50_step_ms']:.4f} p95_ms={metrics['p95_step_ms']:.4f}"
    )
    print(summary)
    if metrics["device"] == "cuda":
        print(
            "cuda_peak_allocated_mb="
            f"{metrics['cuda_peak_allocated_mb']:.2f} "
            "cuda_peak_reserved_mb="
            f"{metrics['cuda_peak_reserved_mb']:.2f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        _write_csv_row(out_path, metrics)
        json_path = out_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    else:
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
