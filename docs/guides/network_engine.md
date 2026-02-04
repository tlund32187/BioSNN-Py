# Network Engine Guide

This guide covers the multi-population Torch network engine usage via the public
API. The recommended path is to build a network with `NetworkBuilder` and execute
it with `Trainer`.

## Concepts
- NetworkBuilder: fluent network construction.
- Trainer: orchestrates engine creation and stepping.

## Minimal example

```python
from biosnn.api import (
    NetworkBuilder,
    Trainer,
    ErdosRenyi,
    GLIFModel,
    DelayedCurrentParams,
    DelayedCurrentSynapse,
)

net = (
    NetworkBuilder()
    .device("cpu")
    .dtype("float32")
    .population("A", n=50, neuron=GLIFModel())
    .population("B", n=25, neuron=GLIFModel())
    .projection(
        "A",
        "B",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1e-9)),
        topology=ErdosRenyi(p=0.1),
    )
    .build()
)

Trainer(net).run(steps=200)
```

## Advanced features
Advanced hooks (initial spikes, external drive, modulators, custom monitoring)
are supported by the underlying engine and can be wired via internal APIs.
For a stable, public workflow, prefer `NetworkBuilder` + `Trainer`.

## Learning rules
The engine can update synapse weights using ILearningRule implementations.
The provided `ThreeFactorHebbianRule` supports dopamine gating.

```python
from biosnn.api import ThreeFactorHebbianParams, ThreeFactorHebbianRule

learning = ThreeFactorHebbianRule(ThreeFactorHebbianParams(lr=1e-3))
proj = ProjectionSpec(
    name="A_to_B",
    synapse=synapse,
    topology=topology,
    pre="A",
    post="B",
    learning=learning,
)
```

Weights are updated in-place on the synapse state. If you need clamping,
set `clamp_min`/`clamp_max` in projection.meta or synapse params.

## Monitoring
Monitoring is supported, but keep it optional for performance-critical runs.
Use the engine directly if you need custom monitor wiring.

## Tips
- Use positions (shape [N,3]) if you plan to sample modulators by position.
- For deterministic runs, pass `seed` in SimulationConfig.
- The engine emits a single StepEvent with population slices in meta.
