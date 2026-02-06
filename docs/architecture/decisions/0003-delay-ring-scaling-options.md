# ADR 0003: Delay Ring Scaling Options

**Status:** Accepted  
**Date:** 2026-02-06

## Context
Delay-based synapses currently use a dense ring buffer per projection. This is simple and fast
for moderate sizes, but it scales linearly with `(max_delay_steps + 1) * n_post` and can become
the dominant memory cost for large networks. We need to document the scaling tradeoffs and
alternative designs for future work.

## Decision
We will keep the dense ring as the default, but document alternative strategies and the
invariants any replacement must satisfy. The options below are design candidates for scaling
large models without changing core semantics.

## Option A: Dense Ring (current)
Cost model:
`bytes ~= (max_delay_steps + 1) * n_post * n_compartments * bytes_per_value`

Benefits:
- Simple indexing: `slot = ring[cursor]`, `schedule -> ring[(cursor + delay) % ring_len]`.
- Deterministic and easy to reason about.
- High throughput for dense activity and moderate delay windows.

Tradeoffs:
- Memory scales with `max_delay_steps`, even if activity is sparse.
- Large `n_post` projections can allocate very large rings.

## Option B: Unique-Delay Slots (delay buckets)
Idea:
- Keep only `D` unique delay buckets (one row per unique delay).
- Still must map each bucket to a time slot every step because `(cursor + delay)` changes.

Notes:
- This is not a memory win by itself unless paired with time-sparse representation.
- It reduces matrix sizes for fused paths, but does not avoid time indexing.

## Option C: Sparse-Event Ring (event driven)
Idea:
- Store events as `(t_due, post_idx, value)` tuples.
- On each step, pop events with `t_due == t`.

Best for:
- Very sparse spiking.
- Event-driven synapses that already enumerate active edges.

Tradeoffs:
- Needs CUDA-safe storage and bucketing.
- Avoid Python dicts or lists keyed by GPU values.
- Prefer tensor-based queues, radix/bucket sort, or fixed-size staging buffers.

## Option D: Chunked/Block Ring (lazy blocks)
Idea:
- Partition `post_idx` into blocks.
- Allocate blocks lazily per time slot only when activity touches them.

Best for:
- Large `n_post` where each step touches only a small fraction of posts.

Tradeoffs:
- More complex bookkeeping to track allocated blocks per slot.
- Must ensure deterministic allocation/clearing order across runs.

## Invariants Any Strategy Must Preserve
- Exact delay alignment: a spike with delay `d` must land at step `t + d` with the same semantics.
- Deterministic pop/clear: once a slot is consumed, it must be cleared predictably.
- Safe accumulation: multiple contributions to the same `(time, post_idx, compartment)` must sum.
- Device safety: no CPU sync in the hot path; no Python-side structures for CUDA data.

## Consequences
- Dense ring remains the default for simplicity and performance.
- Alternative designs are documented for future scaling work.
- Any new ring strategy must satisfy the invariants above and remain compatible with
  fused delay bucketing semantics.
