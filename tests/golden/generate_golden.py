from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from biosnn.contracts.simulation import SimulationConfig

from tests.support.determinism import set_deterministic_cpu
from tests.support.tap_monitor import TapMonitor
from tests.support.scenarios import (
    build_delay_impulse_engine,
    build_learning_gate_engine,
    build_prop_chain_engine,
)

STEPS = 20
SEED = 123
DT = 1e-3
DTYPE = "float64"
DEVICE = "cpu"


def main() -> None:
    args = _parse_args()
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{args.name}_v1.npz"
    if npz_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {npz_path}")

    runner = _SCENARIOS[args.name]
    with set_deterministic_cpu(SEED):
        engine, tap_keys = runner()
        engine.reset(config=SimulationConfig(dt=DT, device=DEVICE, dtype=DTYPE, seed=SEED))
        tap = TapMonitor(tap_keys, cpu_copy=True)
        engine.attach_monitors([tap])
        engine.run(STEPS)

    data = tap.to_numpy_dict()
    data["steps"] = np.arange(len(tap.t), dtype=np.int64)

    encoded = {_encode_key(key): np.asarray(value) for key, value in data.items()}
    np.savez(npz_path, **encoded)

    if args.csv:
        csv_dir = out_dir / f"{args.name}_v1"
        csv_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(csv_dir, encoded)

    print(f"Wrote {npz_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate golden snapshots for tests.")
    parser.add_argument(
        "--name",
        required=True,
        choices=sorted(_SCENARIOS.keys()),
        help="Scenario name to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ file.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write per-key CSV files for human diff.",
    )
    return parser.parse_args()


def _encode_key(key: str) -> str:
    return key.replace("/", "__")


def _write_csv(out_dir: Path, data: dict[str, np.ndarray]) -> None:
    for key, array in sorted(data.items()):
        path = out_dir / f"{key}.csv"
        arr = _prepare_csv_array(array)
        np.savetxt(path, arr, fmt="%.9e", delimiter=",")


def _prepare_csv_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype.kind in {"b", "i", "u"}:
        arr = arr.astype(np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _run_prop_chain() -> tuple[object, tuple[str, ...]]:
    engine, tap_keys, _ = build_prop_chain_engine(compiled_mode=False)
    return engine, tap_keys


def _run_delay_impulse() -> tuple[object, tuple[str, ...]]:
    engine, proj_name, _ = build_delay_impulse_engine(delay_steps=3, compiled_mode=False)
    tap_keys = (
        "pop/Input/spikes",
        "pop/Post/spikes",
        "pop/Post/v_soma_raw",
        "pop/Post/last_drive_dendrite",
        f"proj/{proj_name}/weights",
    )
    return engine, tap_keys


def _run_learning_gate() -> tuple[object, tuple[str, ...]]:
    engine, proj_name, _, _ = build_learning_gate_engine(dopamine_on=True, compiled_mode=False)
    tap_keys = (
        "pop/Pre/spikes",
        "pop/Post/spikes",
        f"proj/{proj_name}/weights",
        f"learn/{proj_name}/last_mean_dw",
        "mod/dopamine/levels",
    )
    return engine, tap_keys


_SCENARIOS: dict[str, Callable[[], tuple[object, tuple[str, ...]]]] = {
    "prop_chain": _run_prop_chain,
    "delay_impulse": _run_delay_impulse,
    "learning_gate": _run_learning_gate,
}


if __name__ == "__main__":
    main()
