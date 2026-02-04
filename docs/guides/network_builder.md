# NetworkBuilder

`NetworkBuilder` is the fluent, minimal front door for constructing a network
without wiring specs by hand. It produces the existing spec objects used by
`TorchNetworkEngine`.

## Minimal two-pop example

```python
from biosnn.api import NetworkBuilder, ErdosRenyi, GLIFModel, DelayedCurrentSynapse, Trainer

builder = (
    NetworkBuilder()
    .device("cpu")
    .dtype("float32")
    .seed(123)
    .population("sensory", n=32, neuron=GLIFModel())
    .population("motor", n=8, neuron=GLIFModel())
    .projection(
        "sensory",
        "motor",
        synapse=DelayedCurrentSynapse(),
        topology=ErdosRenyi(p=0.1),
    )
)

net = builder.build()
Trainer(net).run(steps=100)
```

## Delayed synapse + weight init

```python
from biosnn.api import NetworkBuilder, ErdosRenyi, Init, GLIFModel, DelayedCurrentParams, DelayedCurrentSynapse

net = (
    NetworkBuilder()
    .device("cuda")
    .dtype("float32")
    .seed(7)
    .population("input", n=64, neuron=GLIFModel())
    .population("output", n=10, neuron=GLIFModel())
    .projection(
        "input",
        "output",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.01)),
        topology=ErdosRenyi(p=0.2),
        weights=Init.normal(mean=0.02, std=0.005),
    )
    .build()
)
```

Notes:
- `build()` returns a lightweight `NetworkSpec` with `populations`, `projections`,
  and `modulators` tuples.
- The builder validates names, endpoints, and topology dimensions and raises
  `ValueError` with clear messages when something is wrong.
