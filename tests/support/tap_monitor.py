from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from biosnn.contracts.monitors import IMonitor, StepEvent
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch


class TapMonitor(IMonitor):
    """Test-only monitor that records selected tensors/scalars in memory."""

    name = "tap_monitor"

    def __init__(self, keys: Sequence[str], cpu_copy: bool = True) -> None:
        self._keys = tuple(keys)
        self._cpu_copy = bool(cpu_copy)
        self._series: dict[str, list[Any]] = {key: [] for key in self._keys}
        self.t: list[float] = []
        self.dt: list[float] = []

    def on_step(self, event: StepEvent) -> None:
        self.t.append(float(event.t))
        self.dt.append(float(event.dt))
        tensors = event.tensors or {}
        scalars = event.scalars or {}

        for key in self._keys:
            if key in tensors:
                self._series[key].append(_clone_tensor(tensors[key], self._cpu_copy))
                continue
            if key in scalars:
                self._series[key].append(_clone_scalar(scalars[key], self._cpu_copy))

    def get_series(self, key: str) -> list[Any]:
        if key == "t":
            return self.t
        if key == "dt":
            return self.dt
        return self._series[key]

    def to_numpy_dict(self) -> dict[str, np.ndarray]:
        data: dict[str, np.ndarray] = {
            "t": np.asarray(self.t),
            "dt": np.asarray(self.dt),
        }
        for key, series in self._series.items():
            data[key] = _series_to_numpy(series)
        return data

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _clone_tensor(value: Tensor, cpu_copy: bool) -> Tensor:
    if cpu_copy:
        return value.detach().to("cpu").clone()
    return value.detach().clone()


def _clone_scalar(value: Any, cpu_copy: bool) -> Any:
    if _is_torch_tensor(value):
        return _clone_tensor(value, cpu_copy)
    return value


def _series_to_numpy(series: Sequence[Any]) -> np.ndarray:
    if not series:
        return np.asarray([])
    first = series[0]
    if _is_torch_tensor(first):
        torch = require_torch()
        stacked = torch.stack([item.detach().to("cpu") for item in series])
        return stacked.numpy()
    return np.asarray(series)


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "to") and hasattr(value, "cpu")
