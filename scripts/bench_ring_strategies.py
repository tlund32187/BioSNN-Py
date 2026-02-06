"""Benchmark delayed-current ring strategies (dense vs event-list prototype)."""

from __future__ import annotations

import argparse
import time

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("torch is required to run this benchmark") from exc

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse


def _build_topology(
    *,
    n_pre: int,
    n_post: int,
    edges: int,
    max_delay: int,
    device: str,
    dtype: str,
) -> SynapseTopology:
    pre_idx = torch.randint(0, n_pre, (edges,), dtype=torch.long, device=device)
    post_idx = torch.randint(0, n_post, (edges,), dtype=torch.long, device=device)
    delay_steps = torch.randint(0, max_delay + 1, (edges,), dtype=torch.int32, device=device)
    weights = torch.rand((edges,), dtype=torch.float32, device=device)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        weights=weights,
        target_compartment=Compartment.DENDRITE,
    )
    return compile_topology(
        topology,
        device=device,
        dtype=dtype,
        build_pre_adjacency=True,
    )


def _make_spike_sequence(
    *,
    steps: int,
    n_pre: int,
    device: str,
    rate: float,
) -> list[torch.Tensor]:
    spikes = torch.rand((steps, n_pre), device=device) < rate
    return [spikes[i].to(dtype=torch.bool) for i in range(steps)]


def _ring_bytes_dense(state: object) -> int:
    ring_bytes = 0
    post_ring = getattr(state, "post_ring", None)
    if isinstance(post_ring, dict):
        for ring in post_ring.values():
            ring_bytes += int(ring.numel() * ring.element_size())
    return ring_bytes


def _event_bytes(state: object) -> int:
    ring_bytes = 0
    post_event_ring = getattr(state, "post_event_ring", None)
    if isinstance(post_event_ring, dict):
        for ring in post_event_ring.values():
            ring_bytes += int(ring.due_row.numel() * ring.due_row.element_size())
            ring_bytes += int(ring.post_idx.numel() * ring.post_idx.element_size())
            ring_bytes += int(ring.values.numel() * ring.values.element_size())
    return ring_bytes


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = int((len(values) - 1) * pct / 100.0)
    return float(values[pos])


def _bench(
    *,
    device: str,
    dtype: str,
    n_pre: int,
    n_post: int,
    edges: int,
    max_delay: int,
    steps: int,
    warmup: int,
    rate: float,
    ring_strategy: str,
    ring_dtype: str | None,
    max_events: int,
) -> None:
    topology = _build_topology(
        n_pre=n_pre,
        n_post=n_post,
        edges=edges,
        max_delay=max_delay,
        device=device,
        dtype=dtype,
    )
    params = DelayedCurrentParams(
        init_weight=1.0,
        event_driven=True,
        ring_strategy=ring_strategy,  # type: ignore[arg-type]
        ring_dtype=ring_dtype,
        event_list_max_events=max_events,
    )
    synapse = DelayedCurrentSynapse(params)
    ctx = StepContext(device=device, dtype=dtype)
    state = synapse.init_state(topology.pre_idx.numel(), ctx=ctx)
    if topology.weights is not None:
        state.weights.copy_(topology.weights)

    pre_seq = _make_spike_sequence(steps=steps + warmup, n_pre=n_pre, device=device, rate=rate)

    for step in range(warmup):
        state, _ = synapse.step(
            state,
            topology,
            SynapseInputs(pre_spikes=pre_seq[step]),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    durations: list[float] = []
    peak_event_bytes = 0
    for step in range(warmup, warmup + steps):
        start = time.perf_counter()
        state, _ = synapse.step(
            state,
            topology,
            SynapseInputs(pre_spikes=pre_seq[step]),
            dt=1e-3,
            t=step * 1e-3,
            ctx=ctx,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000.0)
        if ring_strategy == "event_list_proto":
            peak_event_bytes = max(peak_event_bytes, _event_bytes(state))

    median_ms = _percentile(durations, 50.0)
    p90_ms = _percentile(durations, 90.0)
    steps_per_sec = 1000.0 / max(median_ms, 1e-9)

    if ring_strategy == "event_list_proto":
        ring_bytes = peak_event_bytes
        mem_label = "event-list bytes (peak)"
    else:
        ring_bytes = _ring_bytes_dense(state)
        mem_label = "dense ring bytes"

    print(
        f"strategy={ring_strategy} ring_dtype={ring_dtype or 'match'} device={device} dtype={dtype}"
    )
    print(
        f"n_pre={n_pre} n_post={n_post} edges={edges} max_delay={max_delay} rate={rate:.4f}"
    )
    print(f"{mem_label}: {ring_bytes} ({ring_bytes / (1024 * 1024):.2f} MiB)")
    print(
        f"timing: median={median_ms:.3f} ms  p90={p90_ms:.3f} ms  steps/sec={steps_per_sec:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark delay ring strategies (dense vs event list)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--ring-strategy",
        choices=["dense", "event_list_proto"],
        default="dense",
    )
    parser.add_argument(
        "--ring-dtype",
        choices=["none", "float32", "float16", "bfloat16"],
        default="none",
    )
    parser.add_argument("--n-pre", type=int, default=2000)
    parser.add_argument("--n-post", type=int, default=2000)
    parser.add_argument("--edges", type=int, default=20000)
    parser.add_argument("--max-delay", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument("--max-events", type=int, default=1_000_000)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    ring_dtype = None if args.ring_dtype == "none" else args.ring_dtype

    _bench(
        device=args.device,
        dtype=args.dtype,
        n_pre=max(1, int(args.n_pre)),
        n_post=max(1, int(args.n_post)),
        edges=max(1, int(args.edges)),
        max_delay=max(0, int(args.max_delay)),
        steps=max(1, int(args.steps)),
        warmup=max(0, int(args.warmup)),
        rate=max(0.0, min(1.0, float(args.rate))),
        ring_strategy=args.ring_strategy,
        ring_dtype=ring_dtype,
        max_events=max(1, int(args.max_events)),
    )


if __name__ == "__main__":
    main()
