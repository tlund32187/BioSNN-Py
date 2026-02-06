"""Micro-benchmark fused sparse SpMM for delay buckets (COO vs CSR)."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterable

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("torch is required to run this benchmark") from exc


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    pct = min(max(pct, 0.0), 100.0)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


def _bytes_per_value(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    return 4


def _format_bytes(value: int) -> str:
    mib = value / (1024.0 * 1024.0)
    return f"{value} B ({mib:.2f} MiB)"


def _supports_sparse_mm_cpu(dtype: torch.dtype) -> bool:
    try:
        values = torch.ones((1,), dtype=dtype, device="cpu")
        indices = torch.zeros((2, 1), dtype=torch.long, device="cpu")
        mat = torch.sparse_coo_tensor(indices, values, (1, 1), device="cpu", dtype=dtype)
        vec = torch.ones((1, 1), dtype=dtype, device="cpu")
        _ = torch.sparse.mm(mat, vec)
    except Exception:
        return False
    return True


def _make_fused_matrix(
    *,
    n_pre: int,
    n_post: int,
    delays: int,
    nnz: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, int]:
    torch.manual_seed(seed)
    delay_blocks = torch.randint(0, delays, (nnz,), dtype=torch.long, device=device)
    post_idx = torch.randint(0, n_post, (nnz,), dtype=torch.long, device=device)
    pre_idx = torch.randint(0, n_pre, (nnz,), dtype=torch.long, device=device)
    row = delay_blocks * n_post + post_idx
    indices = torch.stack([row, pre_idx], dim=0)
    values = torch.rand((nnz,), device=device, dtype=dtype)
    size = (delays * n_post, n_pre)
    fused = torch.sparse_coo_tensor(indices, values, size=size, device=device, dtype=dtype)
    fused = fused.coalesce()
    return fused, int(fused._nnz())


def _make_pre_activity(
    *,
    n_pre: int,
    device: str,
    dtype: torch.dtype,
    spike_rate: float | None,
    k_active: int | None,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed + 1)
    vec = torch.zeros((n_pre,), device=device, dtype=dtype)
    if k_active is not None:
        k = max(0, min(int(k_active), n_pre))
        if k > 0:
            idx = torch.randperm(n_pre, device=device)[:k]
            vec[idx] = 1.0
        return vec
    rate = 0.0 if spike_rate is None else min(max(float(spike_rate), 0.0), 1.0)
    mask = torch.rand((n_pre,), device=device) < rate
    vec[mask] = 1.0
    return vec


def _time_spmm(
    mat: torch.Tensor,
    vec_col: torch.Tensor,
    *,
    iters: int,
    warmup: int,
    device: str,
) -> list[float]:
    for _ in range(warmup):
        _ = torch.sparse.mm(mat, vec_col)
    if device == "cuda":
        torch.cuda.synchronize()
        starts: list[torch.cuda.Event] = []
        ends: list[torch.cuda.Event] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = torch.sparse.mm(mat, vec_col)
            end.record()
            starts.append(start)
            ends.append(end)
        torch.cuda.synchronize()
        return [float(s.elapsed_time(e)) for s, e in zip(starts, ends, strict=True)]
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = torch.sparse.mm(mat, vec_col)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def _bench_layouts(
    *,
    device: str,
    dtype: torch.dtype,
    layout: str,
    n_pre: int,
    n_post: int,
    delays: int,
    nnz: int,
    spike_rate: float | None,
    k_active: int | None,
    warmup: int,
    iters: int,
    seed: int,
) -> None:
    fused_coo, actual_nnz = _make_fused_matrix(
        n_pre=n_pre,
        n_post=n_post,
        delays=delays,
        nnz=nnz,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    vec = _make_pre_activity(
        n_pre=n_pre,
        device=device,
        dtype=dtype,
        spike_rate=spike_rate,
        k_active=k_active,
        seed=seed,
    )
    vec_col = vec.reshape(-1, 1)

    layouts: Iterable[str] = (
        ("csr", "coo") if device == "cpu" else ("coo",)
    ) if layout == "auto" else (layout,)

    rows = delays * n_post
    bytes_val = _bytes_per_value(dtype)
    coo_index_bytes = 2 * actual_nnz * 8
    coo_value_bytes = actual_nnz * bytes_val
    csr_crow_bytes = (rows + 1) * 8
    csr_col_bytes = actual_nnz * 8
    csr_value_bytes = actual_nnz * bytes_val

    print("layout  device  dtype      shape(rows x cols)        nnz      delays  activity")
    for layout_choice in layouts:
        if layout_choice == "coo":
            mat = fused_coo
        elif layout_choice == "csr":
            if not hasattr(torch.Tensor, "to_sparse_csr"):
                print("csr     (skipped: torch has no CSR support)")
                continue
            try:
                mat = fused_coo.to_sparse_csr()
            except Exception as exc:  # pragma: no cover - device dependent
                print(f"csr     (skipped: {exc})")
                continue
        else:
            raise ValueError(f"Unknown layout {layout_choice}")

        try:
            timings = _time_spmm(mat, vec_col, iters=iters, warmup=warmup, device=device)
        except Exception as exc:  # pragma: no cover - device dependent
            print(f"{layout_choice:6s} (skipped: {exc})")
            continue

        timings_sorted = sorted(timings)
        median_ms = _percentile(timings_sorted, 50.0)
        p90_ms = _percentile(timings_sorted, 90.0)
        steps_per_sec = 1000.0 / max(median_ms, 1e-9)
        activity = f"k={k_active}" if k_active is not None else f"rate={spike_rate:.4f}"
        print(
            f"{layout_choice:6s}  {device:6s}  {str(dtype).replace('torch.', ''):8s}  "
            f"{rows:8d} x {n_pre:<8d}  {actual_nnz:8d}  {delays:6d}  {activity}"
        )
        print(
            f"  timing: median={median_ms:8.3f} ms  p90={p90_ms:8.3f} ms  "
            f"steps/sec={steps_per_sec:10.1f}"
        )

    print("memory estimates (actual nnz after coalesce)")
    print(f"  COO indices: {_format_bytes(coo_index_bytes)}")
    print(f"  COO values : {_format_bytes(coo_value_bytes)}")
    print(f"  CSR crow   : {_format_bytes(csr_crow_bytes)}")
    print(f"  CSR col    : {_format_bytes(csr_col_bytes)}")
    print(f"  CSR values : {_format_bytes(csr_value_bytes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fused sparse SpMM (COO vs CSR)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype",
        choices=list(_DTYPE_MAP.keys()),
        default="float32",
        help="float16/bfloat16 require CUDA or CPU support",
    )
    parser.add_argument("--layout", choices=["auto", "coo", "csr"], default="auto")
    parser.add_argument("--n_pre", type=int, default=200_000)
    parser.add_argument("--n_post", type=int, default=200_000)
    parser.add_argument("--nnz", type=int, default=2_000_000)
    parser.add_argument("--delays", type=int, default=8)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--spike_rate", type=float, default=0.01)
    group.add_argument("--k_active", type=int)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    dtype = _DTYPE_MAP[args.dtype]
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16) and not _supports_sparse_mm_cpu(dtype):
        raise SystemExit(
            f"{args.dtype} sparse mm not supported on CPU in this build. "
            "Use --dtype float32 or run on CUDA."
        )

    n_pre = max(1, int(args.n_pre))
    n_post = max(1, int(args.n_post))
    nnz = max(1, int(args.nnz))
    delays = max(1, int(args.delays))
    iters = max(1, int(args.iters))
    warmup = max(0, int(args.warmup))

    print("Fused SpMM benchmark")
    print(f"device={device} dtype={args.dtype} layout={args.layout}")

    _bench_layouts(
        device=device,
        dtype=dtype,
        layout=args.layout,
        n_pre=n_pre,
        n_post=n_post,
        delays=delays,
        nnz=nnz,
        spike_rate=None if args.k_active is not None else float(args.spike_rate),
        k_active=args.k_active,
        warmup=warmup,
        iters=iters,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
