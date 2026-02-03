# Network Engine Guide

This guide covers the multi-population Torch network engine, modulators, and learning
rules. It is a minimal, contract-first engine designed to connect existing neuron
models, synapse models, and monitors without adding any file I/O inside models.

## Concepts
- PopulationSpec: one neuron model + N neurons + optional positions.
- ProjectionSpec: one synapse model + topology + pre/post population names.
- ModulatorSpec: one modulator field + which kinds it provides.
- TorchNetworkEngine: advances all populations/projections in a consistent order
  and emits a single StepEvent per step.

## Minimal example

```python
import torch

from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import build_erdos_renyi_topology
from biosnn.contracts.simulation import SimulationConfig
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics import DelayedCurrentParams, DelayedCurrentSynapse

# Populations
pop_a = PopulationSpec(name="A", model=GLIFModel(), n=50)
pop_b = PopulationSpec(name="B", model=GLIFModel(), n=25)

# Topology (A -> B)
topology = build_erdos_renyi_topology(n=50, p=0.1)

# Synapses
synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1e-9))
proj = ProjectionSpec(name="A_to_B", synapse=synapse, topology=topology, pre="A", post="B")

engine = TorchNetworkEngine(populations=[pop_a, pop_b], projections=[proj])
engine.reset(config=SimulationConfig(dt=1e-3))
engine.run(steps=200)
```

## Initial spikes
You can seed initial spikes via SimulationConfig.meta:

```python
config = SimulationConfig(
    dt=1e-3,
    meta={
        "initial_spikes_by_pop": {"A": [0, 3, 7]},
    },
)
```

Note: list/tuple values are treated as index lists. If you want a full spike
vector, pass a tensor of shape [N].

## External drive
To inject currents externally, provide an `external_drive_fn`:

```python
from biosnn.contracts.neurons import Compartment

def drive_fn(t, step, pop_name, ctx):
    if pop_name == "A":
        return {Compartment.SOMA: torch.full((50,), 1e-10)}
    return {}

engine = TorchNetworkEngine(
    populations=[pop_a, pop_b],
    projections=[proj],
    external_drive_fn=drive_fn,
)
```

## Modulators
The engine supports modulator fields (e.g., dopamine). The provided
`GlobalScalarField` is a minimal implementation.

```python
from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.neuromodulators import GlobalScalarField, GlobalScalarParams
from biosnn.simulation.network import ModulatorSpec

field = GlobalScalarField(
    kinds=(ModulatorKind.DOPAMINE,),
    params=GlobalScalarParams(decay_tau=1.0),
)
mod = ModulatorSpec(name="da", field=field, kinds=(ModulatorKind.DOPAMINE,))

engine = TorchNetworkEngine(
    populations=[pop_a, pop_b],
    projections=[proj],
    modulators=[mod],
)

# Optional release function

def releases_fn(t, step, ctx):
    return [
        ModulatorRelease(
            kind=ModulatorKind.DOPAMINE,
            positions=torch.zeros((1, 3)),
            amount=torch.tensor([1.0]),
        )
    ]
```

## Learning rules
The engine can update synapse weights using ILearningRule implementations.
The provided `ThreeFactorHebbianRule` supports dopamine gating.

```python
from biosnn.learning import ThreeFactorHebbianParams, ThreeFactorHebbianRule

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
Attach monitors to capture spikes or state tensors:

```python
from pathlib import Path
from biosnn.monitors.csv import NeuronCSVMonitor, SynapseCSVMonitor

out_dir = Path("artifacts")
out_dir.mkdir(parents=True, exist_ok=True)

engine.attach_monitors([
    NeuronCSVMonitor(out_dir / "neurons.csv"),
    SynapseCSVMonitor(out_dir / "synapses.csv"),
])
```

## Tips
- Use positions (shape [N,3]) if you plan to sample modulators by position.
- For deterministic runs, pass `seed` in SimulationConfig.
- The engine emits a single StepEvent with population slices in meta.
