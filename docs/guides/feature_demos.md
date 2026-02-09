# Feature Demos

These CLI demos exercise advanced network features end-to-end and write dashboard artifacts.

## `propagation_impulse`
- Purpose: deterministic spike propagation across `Input -> Relay -> Hidden -> Out`.
- Artifacts: `topology.json`, `neuron.csv`, `tap.csv`, `spikes.csv`, `synapse.csv`, `weights.csv`, `metrics.csv`.

```bash
python -m biosnn.runners.cli --demo propagation_impulse --device cpu --steps 20 --mode dashboard
```

## `delay_impulse`
- Purpose: verify exact delay alignment for `1 -> 1` impulse propagation.
- Key arg: `--delay_steps`.
- Artifacts include `synapse.csv` with projection-drive traces and expected arrival step.

```bash
python -m biosnn.runners.cli --demo delay_impulse --device cpu --steps 20 --delay_steps 5 --mode dashboard
```

## `learning_gate`
- Purpose: deterministic co-spike schedule for a single learned edge.
- Key arg: `--learning_lr`.
- `weights.csv` should show non-decreasing weight updates.

```bash
python -m biosnn.runners.cli --demo learning_gate --device cpu --steps 20 --learning_lr 0.1 --mode dashboard
```

## `dopamine_plasticity`
- Purpose: three-factor plasticity with dopamine-gated learning.
- Uses an off->on dopamine phase in one run:
  - before `--da_step`: no DA release
  - from `--da_step`: DA releases enabled
- Key args: `--da_amount`, `--da_step`.

```bash
python -m biosnn.runners.cli --demo dopamine_plasticity --device cpu --steps 30 --da_amount 1.0 --da_step 10 --mode dashboard
```
