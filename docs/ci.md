# CI Notes

BioSNN-Py CI runs on Python `3.13` and installs CPU-compatible dependencies explicitly.

## Dependency install in CI

Each CI job installs:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install "numpy>=1.26.0"
python -m pip install "torch" --index-url https://download.pytorch.org/whl/cpu
```

This is scoped to CI jobs and does not change local CUDA installations.

## Default test selection

The default test job runs:

```bash
pytest -m "not cuda and not slow"
```

This keeps hosted-runner CI fast and deterministic while excluding optional CUDA/slow suites.

## CUDA tests in CI

- CUDA tests are optional and run only in the workflow-dispatch CUDA job.
- Hosted runners are typically CPU-only; when no CUDA device is present, the CUDA job reports a skip note and exits cleanly.
- Full CUDA invariant coverage requires a GPU-capable runner.

## Running CUDA tests locally

Use a Python environment with a CUDA-enabled PyTorch build and a visible GPU, then run:

```bash
pytest -m "cuda"
```

For full local coverage:

```bash
pytest -m "cuda or not cuda"
```
