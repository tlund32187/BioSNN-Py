# Adding Neuron Models

This guide shows how to create a custom neuron model that integrates cleanly
with the BioSNN engine.

## Recommended structure (step ordering)

1) Validate inputs (compartment drives, shapes).
2) Compute membrane updates.
3) Determine spikes (bool tensor).
4) Apply resets/refractory effects.
5) Return `NeuronStepResult` with `spikes` and optional `membrane`.

## Device/dtype rules

- All state tensors must live on the same device/dtype as the engine.
- Inputs are provided on that device/dtype.
- Avoid `.item()` or Python loops inside `step()`.

## Spikes

- Always return a `torch.bool` tensor of shape `[N]`.
- Avoid casting to float; monitors can convert if needed.

## Using the base class

`NeuronModelBase` provides a guided scaffold and validation:

```python
from biosnn.api import NeuronModelBase, StateTensorSpec, Compartment, NeuronStepResult

class MyNeuron(NeuronModelBase):
    name = "my_neuron"

    def state_tensors_spec(self):
        return {"v_soma": StateTensorSpec(shape=(None,), dtype="float")}

    def init_state_tensors(self, n, *, device, dtype):
        ...

    def state_tensors(self, state):
        return {"v_soma": state.v_soma}

    def step_state(self, state, drive, *, dt, t, ctx):
        ...
        return state, NeuronStepResult(spikes=spikes, membrane={Compartment.SOMA: v_next})
```

See `src/biosnn/biophysics/models/template_neuron.py` for a minimal working example.

## Common pitfalls

- Missing compartment drive key (e.g., `Compartment.SOMA`)
- Returning spikes as float instead of bool
- Mixing devices (CPU vs CUDA) between state tensors
- Allocating new tensors every step instead of reusing state buffers
