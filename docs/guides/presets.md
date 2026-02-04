# Presets

`biosnn.api.presets` provides opinionated, small network templates and safe engine defaults.

## Two-pop delayed demo

```python
from biosnn.api import Trainer, presets

net = presets.make_two_pop_delayed_demo(device="cuda")
runner = Trainer(net).run(steps=1000)
```

## Minimal logic gate

```python
from biosnn.api import Trainer, presets

net = presets.make_minimal_logic_gate_network("xor", device="cpu", seed=7)
engine = Trainer(net).run(steps=250)
```

## Default engine config

```python
from biosnn.api import presets

cfg = presets.default_engine_config(compiled_mode=False)
```

Notes:
- Presets return a `NetworkSpec` (populations + projections).
- `Trainer` is a small convenience wrapper around `TorchNetworkEngine`.
