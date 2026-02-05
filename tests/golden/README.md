# Golden Snapshots

Goldens are CPU-only and deterministic. They capture step-by-step traces for
small scenarios and are used for regression testing.

Regenerate (one at a time):

```sh
python -m tests.golden.generate_golden --name prop_chain --overwrite
python -m tests.golden.generate_golden --name delay_impulse --overwrite
python -m tests.golden.generate_golden --name learning_gate --overwrite
```

Commit updated NPZ files whenever behavioral changes are intentional.
