# Synapses Guide

This guide shows how to build a synapse topology, compute per-edge delays, and
step a simple delayed-current synapse model.

## Concepts
- Topology: fixed pre/post edge lists plus optional per-edge annotations
  (delay_steps, receptors, target compartments, weights, positions).
- Delay steps: computed from 3D distance and myelin factors.
- Ring buffer: handles integer delays efficiently on GPU/CPU.

## Minimal usage example

```python
import torch

from biosnn.api import (
    Compartment,
    StepContext,
    ReceptorKind,
    SynapseInputs,
    SynapseTopology,
    DelayedCurrentParams,
    DelayedCurrentSynapse,
)

# 1) Define a tiny graph (2 pre -> 2 post)
pre_idx = torch.tensor([0, 1], dtype=torch.long)
post_idx = torch.tensor([0, 1], dtype=torch.long)

# 2) Provide positions and optional myelin factors
pre_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
post_pos = torch.tensor([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
myelin = torch.tensor([0.0, 1.0])

# 3) Provide delay steps (dt in seconds); compute externally if needed
delay_steps = torch.tensor([1, 2], dtype=torch.long)

# 4) Optional per-edge receptor and target compartments
receptor = torch.tensor([0, 1], dtype=torch.long)  # AMPA for edge0, GABA for edge1
receptor_kinds = (ReceptorKind.AMPA, ReceptorKind.GABA)
target_compartments = torch.tensor([0, 1], dtype=torch.long)  # soma, dendrite

# 5) Build topology
syn_topology = SynapseTopology(
    pre_idx=pre_idx,
    post_idx=post_idx,
    delay_steps=delay_steps,
    target_compartments=target_compartments,
    receptor=receptor,
    receptor_kinds=receptor_kinds,
    pre_pos=pre_pos,
    post_pos=post_pos,
    myelin=myelin,
)

# 6) Create synapse model + state
ctx = StepContext(device="cpu", dtype="float32")
model = DelayedCurrentSynapse(
    DelayedCurrentParams(init_weight=1e-9, receptor_scale={ReceptorKind.GABA: -1.0})
)
state = model.init_state(e=pre_idx.numel(), ctx=ctx)

# 7) Step (pre spikes -> post drive)
pre_spikes = torch.tensor([1.0, 1.0], dtype=state.weights.dtype)
state, out = model.step(
    state,
    syn_topology,
    SynapseInputs(pre_spikes=pre_spikes),
    dt=1e-3,
    t=0.0,
    ctx=ctx,
)

post_drive = out.post_drive
print(post_drive[Compartment.SOMA])
print(post_drive[Compartment.DENDRITE])
```

## Notes
- target_compartments and receptor are integer-coded per edge.
- If target_compartments is not provided, target_compartment is used for all edges.
- To validate input shapes, set ctx.extras["validate_shapes"] = True (default).
- To require inputs already on the same device (avoid CPU->GPU copies), set
  ctx.extras["require_inputs_on_device"] = True.

## Monitoring
Monitoring is supported via engine monitors; wire these through `Trainer` or the engine
directly when you need CSV output.
