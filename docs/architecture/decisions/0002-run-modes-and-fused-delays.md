# ADR 0002: Run Modes and Fused Delay Buckets

**Status:** Accepted  
**Date:** 2026-02-05

## Context
We need a clear way to run demos in two different operating modes:
- a dashboard-friendly mode that produces full CSV artifacts for visualization
- a throughput-oriented mode that minimizes per-step Python and monitoring overhead

At the same time, the sparse synapse backend uses per-delay buckets that can introduce Python loops
at runtime. We want a CUDA-friendly, vectorized path without losing a fallback for debugging.

## Decision
- Add `--mode {dashboard,fast}` to demo runs.
- In `fast` mode, use `fast_mode=True` and `compiled_mode=True`, and attach only lightweight monitors.
- Always write `topology.json` so runs remain inspectable.
- Enable monitor safety rails by default to avoid unbounded CSV output on large networks.
- For the sparse backend, compile fused delay buckets by default. The fused representation stacks
  non-empty delay buckets into a single sparse matrix per compartment and uses a vectorized scatter
  into the delay ring buffer at runtime.
- Keep the legacy per-delay path as a fallback if fused artifacts are not available.
- Add an optional `--profile` flag to capture a short `torch.profiler` trace after a demo run.

## Consequences
- Users can choose between artifact-rich runs and throughput-oriented runs with a single flag.
- Large network demos avoid accidental CSV blowups unless safety rails are explicitly disabled.
- Profiling is available without changing code paths, which helps detect GPU sync regressions.
- CUDA runs avoid avoidable Python loops in the sparse delay path by default.
- Debugging remains possible via the legacy per-delay path.
