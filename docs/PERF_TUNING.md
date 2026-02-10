# BioSNN Performance Tuning

This page summarizes practical defaults for large-network runs.

## Synapse Backend Choice

- Use `spmm_fused` when activity is moderate/high and projections are large.
- Use `event_driven` when presynaptic spikes are very sparse and most rows are inactive per step.
- Keep `fused_layout=auto` unless profiling shows a stable win from `csr` or `coo`.

## Ring Strategy and Memory

- Dense ring memory scales as:
  - `bytes ~= ring_len * n_post * bytes_per_element` per projected compartment.
- Event-bucketed ring memory scales with scheduled events:
  - `bytes ~= capacity_max * (value_bytes + post_idx_bytes + slot_idx_bytes)` plus small metadata.
- For very large delayed projections with sparse events, prefer `ring_strategy=event_bucketed`.

## Multi-Receptor State Memory (`g[R, N]`)

- Receptor state memory scales as:
  - `bytes ~= R * N_post * bytes_per_element` per compartment.
- Recommended defaults:
  - CPU: keep dtype aligned with weights (`float32` typically).
  - CUDA: use `float16` receptor state unless precision issues are observed.
- Override with CLI:
  - `--receptor-state-dtype {none,float32,float16,bfloat16}`

## Modulator Grid Memory (`grid[K, H, W]`)

- Grid memory scales as:
  - `bytes ~= K * H * W * bytes_per_element`.
- Keep dashboard export grids bounded:
  - `--modgrid-max-side`
  - `--modgrid-max-elements`
- Keep vision frame export bounded:
  - `--vision-max-side`
  - `--vision-max-elements`

## Large Network Safety Mode

Enable with:

```bash
python -m biosnn.runners.cli --large-network-safety ...
```

Safety mode applies conservative defaults:

- Forces monitor safety rails on.
- Defaults CUDA monitor sync to off unless explicitly requested.
- Caps monitor export payload sizes for vision/modulator JSON monitors.
- Defaults multi-receptor state dtype to `float16` on CUDA when not overridden.

## Profiling

- Profiling traces are written only when `--profile` is passed.
- Output file: `profile.json` in the run directory.
- Useful knobs:
  - `--profile`
  - `--profile-steps N`

## Million-Neuron-Scale Guidelines

- Keep monitors minimal (`--mode fast` or sparse monitor set).
- Avoid per-step heavyweight CSV/JSON exports.
- Prefer sparse/event-driven pathways where spike sparsity is high.
- Use CUDA, bounded ring memory (`--max-ring-mib`), and monitor-safe defaults.
