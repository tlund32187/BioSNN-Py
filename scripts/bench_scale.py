"""Scaling benchmark: topology compile + step throughput + monitor overhead."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biosnn.connectivity.builders.bipartite_topology import (
    build_bipartite_erdos_renyi_topology,
)
from biosnn.contracts.monitors import IMonitor
from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.monitors.csv import NeuronCSVMonitor
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)


@dataclass(slots=True)
class FixedState:
    spikes: Tensor
    v: Tensor


class FixedSpikesModel(INeuronModel):
    name = "fixed_spikes"
    compartments = frozenset({Compartment.SOMA})

    def __init__(self, spike_prob: float) -> None:
        self._spike_prob = spike_prob

    def init_state(self, n: int, *, ctx: StepContext) -> FixedState:
        torch = require_torch()
        device = torch.device(ctx.device) if ctx.device else None
        spikes = torch.rand((n,), device=device) < float(self._spike_prob)
        v = torch.zeros((n,), device=device, dtype=torch.float32)
        return FixedState(spikes=spikes, v=v)

    def reset_state(
        self,
        state: FixedState,
        *,
        ctx: StepContext,
        indices: Tensor | None = None,
    ) -> FixedState:
        _ = ctx
        if indices is None:
            state.spikes.zero_()
            state.v.zero_()
        else:
            state.spikes[indices] = False
            state.v[indices] = 0.0
        return state

    def step(
        self,
        state: FixedState,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[FixedState, NeuronStepResult]:
        _ = (inputs, dt, t, ctx)
        return state, NeuronStepResult(spikes=state.spikes)

    def state_tensors(self, state: FixedState):
        return {"v": state.v}


def main() -> None:
    args = _parse_args()
    torch = require_torch()

    devices = _resolve_devices(torch, args.device)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.out_dir or "artifacts") / f"bench_scale_{run_id}"
    base_dir.mkdir(parents=True, exist_ok=True)

    for device in devices:
        print(f"\nDevice: {device}")
        for mode in _iter_monitor_modes(args.monitor_modes):
            out_dir = base_dir / f"{device}_{mode}"
            out_dir.mkdir(parents=True, exist_ok=True)
            result = _run_bench(
                device=device,
                n_pre=args.n_pre,
                n_post=args.n_post,
                p=args.p,
                dt=args.dt,
                steps=args.steps,
                warmup=args.warmup,
                spike_prob=args.spike_prob,
                monitor_mode=mode,
                sample_n=args.sample_n,
                out_dir=out_dir,
            )
            print(
                f"  mode={mode:8s} compile={result.compile_s:.4f}s "
                f"steps/sec={result.steps_per_sec:,.1f}"
            )

            if args.profile and mode == "none":
                profile_path = out_dir / "profile.json"
                _run_profile(
                    engine=result.engine,
                    steps=args.profile_steps,
                    device=device,
                    out_path=profile_path,
                )
                print(f"    profile: {profile_path}")


@dataclass(slots=True)
class BenchResult:
    compile_s: float
    steps_per_sec: float
    engine: TorchNetworkEngine


def _run_bench(
    *,
    device: str,
    n_pre: int,
    n_post: int,
    p: float,
    dt: float,
    steps: int,
    warmup: int,
    spike_prob: float,
    monitor_mode: str,
    sample_n: int,
    out_dir: Path,
) -> BenchResult:
    torch = require_torch()
    dtype = "float32"
    torch.manual_seed(0)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    pre_model = FixedSpikesModel(spike_prob)
    post_model = FixedSpikesModel(0.0)

    pre_pop = PopulationSpec(name="Pre", model=pre_model, n=n_pre)
    post_pop = PopulationSpec(name="Post", model=post_model, n=n_post)

    device_obj = torch.device(device)
    pre_pos = torch.rand((n_pre, 3), device=device_obj, dtype=torch.float32)
    post_pos = torch.rand((n_post, 3), device=device_obj, dtype=torch.float32)

    topology = build_bipartite_erdos_renyi_topology(
        n_pre=n_pre,
        n_post=n_post,
        p=p,
        device=device,
        dtype=dtype,
        dt=dt,
        pre_positions=pre_pos,
        post_positions=post_pos,
    )

    synapse = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=0.01))
    proj = ProjectionSpec(name="Pre->Post", synapse=synapse, topology=topology, pre="Pre", post="Post")

    engine = TorchNetworkEngine(populations=[pre_pop, post_pop], projections=[proj])
    monitors = _build_monitors(out_dir, monitor_mode, sample_n)
    engine.attach_monitors(monitors)

    compile_start = time.perf_counter()
    engine.reset(config=SimulationConfig(dt=dt, device=device, dtype=dtype))
    _sync_device(torch, device)
    compile_s = time.perf_counter() - compile_start

    for _ in range(max(0, warmup)):
        engine.step()
    _sync_device(torch, device)

    t0 = time.perf_counter()
    for _ in range(max(1, steps)):
        engine.step()
    _sync_device(torch, device)
    elapsed = time.perf_counter() - t0
    steps_per_sec = float(steps) / elapsed if elapsed > 0 else 0.0

    for monitor in monitors:
        monitor.close()

    return BenchResult(compile_s=compile_s, steps_per_sec=steps_per_sec, engine=engine)


def _build_monitors(out_dir: Path, mode: str, sample_n: int) -> list[IMonitor]:
    if mode == "none":
        return []

    async_gpu = mode in {"buffered", "async"}
    async_io = mode == "async"
    sample_indices = list(range(sample_n)) if sample_n > 0 else None
    return [
        NeuronCSVMonitor(
            out_dir / "neuron.csv",
            tensor_keys=("v",),
            include_spikes=True,
            sample_indices=sample_indices,
            flush_every=25,
            async_gpu=async_gpu,
            async_io=async_io,
        )
    ]


def _iter_monitor_modes(raw: str) -> Iterable[str]:
    modes = [mode.strip().lower() for mode in raw.split(",") if mode.strip()]
    allowed = {"none", "sync", "buffered", "async"}
    for mode in modes:
        if mode not in allowed:
            raise ValueError(f"Unknown monitor mode: {mode}")
    return modes


def _resolve_devices(torch: Any, device: str) -> list[str]:
    if device == "all":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        return devices
    if device == "cuda" and not torch.cuda.is_available():
        return ["cpu"]
    return [device]


def _sync_device(torch: Any, device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_profile(*, engine: TorchNetworkEngine, steps: int, device: str, out_path: Path) -> None:
    torch = require_torch()
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        print("    profiler unavailable")
        return

    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities) as prof:
        for _ in range(max(1, steps)):
            engine.step()
        _sync_device(torch, device)
    prof.export_chrome_trace(str(out_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaling benchmark for BioSNN")
    parser.add_argument("--n-pre", type=int, default=2000)
    parser.add_argument("--n-post", type=int, default=2000)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--spike-prob", type=float, default=0.05)
    parser.add_argument("--sample-n", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda", "all"], default="all")
    parser.add_argument(
        "--monitor-modes",
        default="none,sync,buffered,async",
        help="comma-separated: none,sync,buffered,async",
    )
    parser.add_argument("--out-dir", default=None, help="base output directory (default artifacts)")
    parser.add_argument("--profile", action="store_true", help="enable torch profiler for a short run")
    parser.add_argument("--profile-steps", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main()
