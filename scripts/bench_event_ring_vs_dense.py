"""Benchmark dense ring vs bucketed event ring for delayed synapse backends.

Compares:
- dense ring with float32 ring buffer
- dense ring with float16/bfloat16 ring buffer (if supported)
- bucketed event ring

Backends:
- delayed current (`DelayedCurrentSynapse`)
- delayed sparse matmul (`DelayedSparseMatmulSynapse`, event-driven mode)
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any, Literal, cast

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("torch is required to run this benchmark") from exc

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext, Tensor
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

BackendName = Literal["current", "sparse"]
RingStrategy = Literal["dense", "event_bucketed"]
DelayDistribution = Literal["uniform", "short", "long", "fixed"]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    backend: Literal["current", "sparse", "both"]
    device: str
    dtype: str
    n_pre: int
    n_post: int
    edges: int
    ring_len: int
    spikes_per_step: int
    delay_distribution: DelayDistribution
    fixed_delay: int
    steps: int
    warmup: int
    seed: int
    ring_capacity_max: int
    dense_ring_dtypes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseSpec:
    backend: BackendName
    label: str
    ring_strategy: RingStrategy
    ring_dtype: str | None


@dataclass(frozen=True, slots=True)
class CaseResult:
    spec: CaseSpec
    ring_mib: float
    out_mib: float
    median_ms: float
    p90_ms: float
    steps_per_sec: float


def _sample_delays(
    *,
    edges: int,
    ring_len: int,
    distribution: DelayDistribution,
    fixed_delay: int,
    device: str,
) -> Tensor:
    max_delay = max(0, ring_len - 1)
    if distribution == "fixed":
        delay = max(0, min(max_delay, int(fixed_delay)))
        delays = torch.full((edges,), delay, device=device, dtype=torch.int32)
    elif distribution == "uniform":
        delays = torch.randint(0, max_delay + 1, (edges,), device=device, dtype=torch.int32)
    elif distribution == "short":
        u = torch.rand((edges,), device=device, dtype=torch.float32)
        delays = torch.floor((u * u) * max_delay).to(dtype=torch.int32)
    else:  # long
        u = torch.rand((edges,), device=device, dtype=torch.float32)
        delays = torch.floor((1.0 - (1.0 - u) * (1.0 - u)) * max_delay).to(dtype=torch.int32)

    # Keep effective ring length exactly as requested.
    if edges >= 1:
        delays[0] = 0
    if edges >= 2:
        delays[1] = max_delay
    return delays


def _build_base_topology(cfg: BenchmarkConfig) -> SynapseTopology:
    pre_idx = torch.randint(0, cfg.n_pre, (cfg.edges,), dtype=torch.long, device=cfg.device)
    post_idx = torch.randint(0, cfg.n_post, (cfg.edges,), dtype=torch.long, device=cfg.device)
    delay_steps = _sample_delays(
        edges=cfg.edges,
        ring_len=cfg.ring_len,
        distribution=cfg.delay_distribution,
        fixed_delay=cfg.fixed_delay,
        device=cfg.device,
    )
    weights = torch.rand((cfg.edges,), dtype=torch.float32, device=cfg.device)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        weights=weights,
        target_compartment=Compartment.DENDRITE,
    )


def _compile_for_case(topology: SynapseTopology, spec: CaseSpec, cfg: BenchmarkConfig) -> SynapseTopology:
    build_edges_by_delay = spec.backend == "current" and spec.ring_strategy == "dense"
    build_pre_adjacency = (
        (spec.backend == "current" and spec.ring_strategy == "event_bucketed")
        or (spec.backend == "sparse" and spec.ring_strategy == "event_bucketed")
    )
    build_sparse_delay_mats = spec.backend == "sparse" and spec.ring_strategy == "dense"
    return compile_topology(
        topology,
        device=cfg.device,
        dtype=cfg.dtype,
        build_edges_by_delay=build_edges_by_delay,
        build_pre_adjacency=build_pre_adjacency,
        build_sparse_delay_mats=build_sparse_delay_mats,
    )


def _make_case_synapse(spec: CaseSpec, cfg: BenchmarkConfig) -> Any:
    if spec.backend == "current":
        params = DelayedCurrentParams(
            init_weight=1.0,
            event_driven=spec.ring_strategy == "event_bucketed",
            ring_strategy=spec.ring_strategy,
            ring_dtype=spec.ring_dtype,
            ring_capacity_max=cfg.ring_capacity_max,
            event_list_max_events=cfg.ring_capacity_max,
        )
        return DelayedCurrentSynapse(params)

    if spec.ring_strategy == "event_bucketed":
        params_sparse = DelayedSparseMatmulParams(
            init_weight=1.0,
            backend="event_driven",
            ring_strategy="event_bucketed",
            ring_dtype=spec.ring_dtype,
            ring_capacity_max=cfg.ring_capacity_max,
        )
    else:
        params_sparse = DelayedSparseMatmulParams(
            init_weight=1.0,
            backend="spmm_fused",
            ring_strategy="dense",
            ring_dtype=spec.ring_dtype,
        )
    return DelayedSparseMatmulSynapse(params_sparse)


def _make_spike_sequence(cfg: BenchmarkConfig) -> tuple[Tensor, ...]:
    total_steps = cfg.warmup + cfg.steps
    seq = torch.zeros((total_steps, cfg.n_pre), device=cfg.device, dtype=torch.bool)
    k = max(0, min(cfg.n_pre, cfg.spikes_per_step))
    if k == 0:
        return cast(tuple[Tensor, ...], tuple(seq.unbind(0)))
    gen = torch.Generator(device=cfg.device)
    gen.manual_seed(cfg.seed + 1337)
    for step in range(total_steps):
        idx = torch.randperm(cfg.n_pre, generator=gen, device=cfg.device)[:k]
        seq[step].index_fill_(0, idx, True)
    return cast(tuple[Tensor, ...], tuple(seq.unbind(0)))


def _tensor_bytes(tensor: Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _state_ring_bytes(state: Any) -> int:
    ring_bytes = 0
    post_ring = getattr(state, "post_ring", None)
    if isinstance(post_ring, dict):
        for ring in post_ring.values():
            ring_bytes += _tensor_bytes(cast(Tensor, ring))

    post_event_ring = getattr(state, "post_event_ring", None)
    if isinstance(post_event_ring, dict):
        for ring in post_event_ring.values():
            ring_bytes += _tensor_bytes(getattr(ring, "due_slot", None))
            ring_bytes += _tensor_bytes(getattr(ring, "post_idx", None))
            ring_bytes += _tensor_bytes(getattr(ring, "values", None))
            ring_bytes += _tensor_bytes(getattr(ring, "slot_counts", None))
            ring_bytes += _tensor_bytes(getattr(ring, "slot_offsets", None))
    return ring_bytes


def _state_out_bytes(state: Any) -> int:
    out_bytes = 0
    post_out = getattr(state, "post_out", None)
    if isinstance(post_out, dict):
        for out in post_out.values():
            out_bytes += _tensor_bytes(cast(Tensor, out))
    return out_bytes


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = int((len(sorted_values) - 1) * pct / 100.0)
    return float(sorted_values[pos])


def _maybe_sync_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _run_case(
    *,
    cfg: BenchmarkConfig,
    base_topology: SynapseTopology,
    pre_seq: tuple[Tensor, ...],
    spec: CaseSpec,
) -> CaseResult | None:
    compiled = _compile_for_case(base_topology, spec, cfg)
    synapse = _make_case_synapse(spec, cfg)
    ctx = StepContext(device=cfg.device, dtype=cfg.dtype)
    state = synapse.init_state(compiled.pre_idx.numel(), ctx=ctx)
    if compiled.weights is not None:
        state.weights.copy_(compiled.weights)

    try:
        for step in range(cfg.warmup):
            state, _ = synapse.step(
                state,
                compiled,
                SynapseInputs(pre_spikes=pre_seq[step]),
                dt=1e-3,
                t=step * 1e-3,
                ctx=ctx,
            )

        _maybe_sync_cuda(cfg.device)
        durations_ms: list[float] = []
        offset = cfg.warmup
        for i in range(cfg.steps):
            step_idx = offset + i
            _maybe_sync_cuda(cfg.device)
            t0 = time.perf_counter()
            state, _ = synapse.step(
                state,
                compiled,
                SynapseInputs(pre_spikes=pre_seq[step_idx]),
                dt=1e-3,
                t=step_idx * 1e-3,
                ctx=ctx,
            )
            _maybe_sync_cuda(cfg.device)
            durations_ms.append((time.perf_counter() - t0) * 1000.0)
    except RuntimeError as exc:
        print(f"skip {spec.backend}/{spec.label}: {exc}")
        return None

    ring_bytes = _state_ring_bytes(state)
    out_bytes = _state_out_bytes(state)
    median_ms = float(statistics.median(durations_ms))
    p90_ms = _percentile(durations_ms, 90.0)
    steps_per_sec = 1000.0 / max(median_ms, 1e-9)
    return CaseResult(
        spec=spec,
        ring_mib=ring_bytes / (1024.0 * 1024.0),
        out_mib=out_bytes / (1024.0 * 1024.0),
        median_ms=median_ms,
        p90_ms=p90_ms,
        steps_per_sec=steps_per_sec,
    )


def _case_specs_for_backend(
    *,
    backend: BackendName,
    dense_ring_dtypes: tuple[str, ...],
) -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    seen: set[str] = set()

    # Dense ring float32 baseline.
    specs.append(
        CaseSpec(
            backend=backend,
            label="dense/float32",
            ring_strategy="dense",
            ring_dtype=None,
        )
    )
    seen.add("float32")

    for dtype in dense_ring_dtypes:
        if dtype in seen:
            continue
        seen.add(dtype)
        specs.append(
            CaseSpec(
                backend=backend,
                label=f"dense/{dtype}",
                ring_strategy="dense",
                ring_dtype=dtype,
            )
        )

    specs.append(
        CaseSpec(
            backend=backend,
            label="bucketed/match",
            ring_strategy="event_bucketed",
            ring_dtype=None,
        )
    )
    return specs


def _print_results(backend: BackendName, results: list[CaseResult]) -> None:
    if not results:
        print(f"\nBackend={backend}: no runnable cases")
        return

    baseline = next((r for r in results if r.spec.label == "dense/float32"), results[0])
    print(f"\nBackend={backend}")
    print(
        "case                 ring_mib  out_mib  median_ms  p90_ms  steps/s  speedup  mem_ratio"
    )
    print(
        "-------------------  --------  -------  ---------  ------  -------  -------  ---------"
    )
    for result in results:
        speedup = baseline.median_ms / max(result.median_ms, 1e-9)
        mem_ratio = result.ring_mib / max(baseline.ring_mib, 1e-9)
        print(
            f"{result.spec.label:<19}  "
            f"{result.ring_mib:8.2f}  "
            f"{result.out_mib:7.2f}  "
            f"{result.median_ms:9.3f}  "
            f"{result.p90_ms:6.3f}  "
            f"{result.steps_per_sec:7.1f}  "
            f"{speedup:7.2f}  "
            f"{mem_ratio:9.2f}"
        )


def _parse_dense_ring_dtypes(value: str) -> tuple[str, ...]:
    allowed = {"float32", "float16", "bfloat16"}
    parts = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not parts:
        return ("float16", "bfloat16")
    invalid = [item for item in parts if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid dense ring dtypes: {', '.join(invalid)}. Allowed: float32,float16,bfloat16"
        )
    return tuple(parts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dense ring vs bucketed event ring (current/sparse backends)."
    )
    parser.add_argument("--backend", choices=["current", "sparse", "both"], default="both")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--n-pre", type=int, default=2048)
    parser.add_argument("--n-post", type=int, default=2048)
    parser.add_argument("--edges", type=int, default=32768)
    parser.add_argument("--ring-len", type=int, default=16)
    parser.add_argument("--spikes-per-step", type=int, default=64)
    parser.add_argument(
        "--delay-distribution",
        choices=["uniform", "short", "long", "fixed"],
        default="uniform",
    )
    parser.add_argument("--fixed-delay", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ring-capacity-max", type=int, default=2_000_000)
    parser.add_argument(
        "--dense-ring-dtypes",
        type=_parse_dense_ring_dtypes,
        default=("float16", "bfloat16"),
        help="Comma-separated subset of float32,float16,bfloat16 for extra dense-ring cases.",
    )
    return parser.parse_args()


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    return requested


def main() -> None:
    args = _parse_args()
    device = _select_device(cast(str, args.device))

    cfg = BenchmarkConfig(
        backend=cast(Literal["current", "sparse", "both"], args.backend),
        device=device,
        dtype=cast(str, args.dtype),
        n_pre=max(1, int(args.n_pre)),
        n_post=max(1, int(args.n_post)),
        edges=max(2, int(args.edges)),
        ring_len=max(1, int(args.ring_len)),
        spikes_per_step=max(0, int(args.spikes_per_step)),
        delay_distribution=cast(DelayDistribution, args.delay_distribution),
        fixed_delay=max(0, int(args.fixed_delay)),
        steps=max(1, int(args.steps)),
        warmup=max(0, int(args.warmup)),
        seed=int(args.seed),
        ring_capacity_max=max(1, int(args.ring_capacity_max)),
        dense_ring_dtypes=cast(tuple[str, ...], args.dense_ring_dtypes),
    )

    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    print("Benchmark configuration")
    print(
        f"device={cfg.device} dtype={cfg.dtype} n_pre={cfg.n_pre} n_post={cfg.n_post} "
        f"edges={cfg.edges} ring_len={cfg.ring_len}"
    )
    print(
        f"spikes_per_step={cfg.spikes_per_step} delay_distribution={cfg.delay_distribution} "
        f"steps={cfg.steps} warmup={cfg.warmup} seed={cfg.seed}"
    )

    backend_order: tuple[BackendName, ...]
    if cfg.backend == "both":
        backend_order = ("current", "sparse")
    else:
        backend_order = (cast(BackendName, cfg.backend),)

    for backend in backend_order:
        base_topology = _build_base_topology(cfg)
        pre_seq = _make_spike_sequence(cfg)
        specs = _case_specs_for_backend(
            backend=backend,
            dense_ring_dtypes=cfg.dense_ring_dtypes,
        )

        results: list[CaseResult] = []
        for spec in specs:
            result = _run_case(
                cfg=cfg,
                base_topology=base_topology,
                pre_seq=pre_seq,
                spec=spec,
            )
            if result is not None:
                results.append(result)
        _print_results(backend, results)


if __name__ == "__main__":
    main()
