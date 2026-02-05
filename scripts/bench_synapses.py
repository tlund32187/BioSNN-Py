"""Tiny synapse throughput benchmark (dev utility)."""

from __future__ import annotations

import argparse
import time
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

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)


def _make_topology(
    *,
    n: int,
    e: int,
    n_delays: int,
    device: str,
    dtype: str,
    build_edges_by_delay: bool = False,
    build_sparse_delay_mats: bool = False,
    fuse_delay_buckets: bool = True,
) -> SynapseTopology:
    pre_idx = torch.randint(0, n, (e,), dtype=torch.long)
    post_idx = torch.randint(0, n, (e,), dtype=torch.long)
    delay_steps = torch.randint(0, n_delays, (e,), dtype=torch.int32)
    weights = torch.rand((e,), dtype=torch.float32) * 0.1 + 0.01
    topo = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        weights=weights,
        target_compartment=Compartment.SOMA,
    )
    return compile_topology(
        topo,
        device=device,
        dtype=dtype,
        build_edges_by_delay=build_edges_by_delay,
        build_sparse_delay_mats=build_sparse_delay_mats,
        fuse_delay_buckets=fuse_delay_buckets,
    )


def _make_spike_seq(
    *,
    steps: int,
    n: int,
    device: str,
    dtype: Any,
    rate: float,
) -> torch.Tensor:
    spikes = torch.rand((steps, n), device=device)
    return (spikes < rate).to(dtype=dtype)


def _bench_synapse(
    *,
    synapse,
    topology: SynapseTopology,
    pre_seq: torch.Tensor,
    ctx: StepContext,
    steps: int,
    warmup: int,
) -> float:
    state = synapse.init_state(topology.pre_idx.numel(), ctx=ctx)
    if topology.weights is not None:
        state.weights.copy_(topology.weights)

    for _ in range(warmup):
        state, _ = synapse.step(
            state,
            topology,
            SynapseInputs(pre_spikes=pre_seq[0]),
            dt=1e-3,
            t=0.0,
            ctx=ctx,
        )

    if ctx.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for step in range(steps):
        state, _ = synapse.step(
            state,
            topology,
            SynapseInputs(pre_spikes=pre_seq[step]),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
    if ctx.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return steps / max(elapsed, 1e-9)


def _run_case(
    *,
    label: str,
    synapse,
    topology: SynapseTopology,
    device: str,
    dtype: str,
    steps: int,
    warmup: int,
    rate: float,
) -> None:
    ctx = StepContext(device=device, dtype=dtype)
    meta = topology.meta
    if meta is None:
        raise RuntimeError("Topology meta missing; compile_topology must be called before benchmarking.")
    pre_seq = _make_spike_seq(
        steps=steps,
        n=int(meta["n_pre"]),
        device=device,
        dtype=torch.float32,
        rate=rate,
    )
    throughput = _bench_synapse(
        synapse=synapse,
        topology=topology,
        pre_seq=pre_seq,
        ctx=ctx,
        steps=steps,
        warmup=warmup,
    )
    print(f"{label:40s} steps/sec: {throughput:10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny synapse throughput benchmark")
    parser.add_argument("--n", type=int, default=2000, help="number of neurons")
    parser.add_argument("--e", type=int, default=20000, help="number of edges")
    parser.add_argument("--delays", type=int, default=5, help="number of delay buckets")
    parser.add_argument("--steps", type=int, default=200, help="timed steps")
    parser.add_argument("--warmup", type=int, default=20, help="warmup steps")
    parser.add_argument("--rate", type=float, default=0.05, help="spike rate (0..1)")
    args = parser.parse_args()

    n = max(1, int(args.n))
    e = max(1, int(args.e))
    n_delays = max(1, int(args.delays))
    steps = max(1, int(args.steps))
    warmup = max(0, int(args.warmup))
    rate = max(0.0, min(1.0, float(args.rate)))

    print(f"Graph: n={n} edges={e} delays={n_delays}")

    cpu_topo_current = _make_topology(
        n=n,
        e=e,
        n_delays=n_delays,
        device="cpu",
        dtype="float32",
        build_edges_by_delay=True,
    )
    _run_case(
        label="CPU | DelayedCurrentSynapse",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.05)),
        topology=cpu_topo_current,
        device="cpu",
        dtype="float32",
        steps=steps,
        warmup=warmup,
        rate=rate,
    )

    for fused in (True, False):
        topo = _make_topology(
            n=n,
            e=e,
            n_delays=n_delays,
            device="cpu",
            dtype="float32",
            build_sparse_delay_mats=True,
            fuse_delay_buckets=fused,
        )
        label = f"CPU | DelayedSparseMatmulSynapse | fused={fused}"
        _run_case(
            label=label,
            synapse=DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=0.05)),
            topology=topo,
            device="cpu",
            dtype="float32",
            steps=steps,
            warmup=warmup,
            rate=rate,
        )

    if torch.cuda.is_available():
        for fused in (True, False):
            topo = _make_topology(
                n=n,
                e=e,
                n_delays=n_delays,
                device="cuda",
                dtype="float32",
                build_sparse_delay_mats=True,
                fuse_delay_buckets=fused,
            )
            label = f"CUDA | DelayedSparseMatmulSynapse | fused={fused}"
            _run_case(
                label=label,
                synapse=DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=0.05)),
                topology=topo,
                device="cuda",
                dtype="float32",
                steps=steps,
                warmup=warmup,
                rate=rate,
            )
    else:
        print("CUDA not available; skipping CUDA benchmarks.")


if __name__ == "__main__":
    main()
