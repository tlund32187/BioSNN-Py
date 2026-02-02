"""CSV monitors for neuron model outputs."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from biosnn.contracts.monitors import IMonitor, StepEvent

_DEFAULT_STATS: tuple[str, ...] = ("mean", "min", "max")


class NeuronCSVMonitor(IMonitor):
    """Write per-step neuron summaries to CSV.

    The monitor expects StepEvent.tensors to include the requested tensor keys.
    By default, it writes mean/min/max statistics for each tensor and basic spike stats.
    """

    name = "csv_neuron"

    def __init__(
        self,
        path: str | Path,
        *,
        tensor_keys: Sequence[str] | None = None,
        stats: Sequence[str] | None = None,
        include_spikes: bool = True,
        include_scalars: bool = True,
        sample_indices: Sequence[int] | None = None,
        flush_every: int = 1,
    ) -> None:
        self._path = Path(path)
        self._tensor_keys = list(tensor_keys) if tensor_keys is not None else None
        self._stats = tuple(stats) if stats is not None else _DEFAULT_STATS
        self._include_spikes = include_spikes
        self._include_scalars = include_scalars
        self._sample_indices = list(sample_indices) if sample_indices is not None else None
        self._flush_every = max(1, flush_every)
        self._step_count = 0
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: list[str] | None = None

    def on_step(self, event: StepEvent) -> None:
        row: dict[str, Any] = {"t": event.t, "dt": event.dt}

        if self._include_spikes and event.spikes is not None:
            spike_count = _reduce(event.spikes, "sum")
            spike_fraction = _reduce(event.spikes, "mean")
            row["spike_count"] = spike_count
            row["spike_fraction"] = spike_fraction
            row["spike_rate_hz"] = (spike_fraction / event.dt) if event.dt else 0.0
            if self._sample_indices is not None:
                for idx, value in _sample(event.spikes, self._sample_indices):
                    row[f"spike_i{idx}"] = value

        if self._include_scalars and event.scalars:
            for key, value in sorted(event.scalars.items()):
                row[key] = float(value)

        tensors = event.tensors or {}
        if self._tensor_keys is None:
            self._tensor_keys = sorted(tensors.keys())

        for key in self._tensor_keys:
            tensor = tensors.get(key)
            if tensor is None:
                continue
            for stat in self._stats:
                row[f"{key}_{stat}"] = _reduce(tensor, stat)
            if self._sample_indices is not None:
                for idx, value in _sample(tensor, self._sample_indices):
                    row[f"{key}_i{idx}"] = value

        self._write_row(row)

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
        self._writer = None
        self._fieldnames = None

    def _write_row(self, row: Mapping[str, Any]) -> None:
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._step_count += 1
        if self._step_count % self._flush_every == 0:
            self._file.flush()


class GLIFCSVMonitor(NeuronCSVMonitor):
    """CSV monitor preset for GLIF tensors."""

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        super().__init__(
            path,
            tensor_keys=("v_soma", "refrac_left", "spike_hold_left", "theta"),
            **kwargs,
        )


class AdEx2CompCSVMonitor(NeuronCSVMonitor):
    """CSV monitor preset for AdEx 2-compartment tensors."""

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        super().__init__(
            path,
            tensor_keys=("v_soma", "v_dend", "w", "refrac_left", "spike_hold_left"),
            **kwargs,
        )


def _reduce(values: Any, stat: str) -> float:
    if values is None:
        return 0.0
    candidate = values
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "flatten"):
        candidate = candidate.flatten()
    reducer = getattr(candidate, stat, None)
    if callable(reducer):
        return _to_scalar(reducer())
    if isinstance(candidate, Iterable):
        flat = list(_flatten(candidate))
        if not flat:
            return 0.0
        if stat == "mean":
            return float(sum(flat) / len(flat))
        if stat == "sum":
            return float(sum(flat))
        if stat == "min":
            return float(min(flat))
        if stat == "max":
            return float(max(flat))
    return _to_scalar(candidate)


def _sample(values: Any, indices: Sequence[int]) -> list[tuple[int, float]]:
    if values is None:
        return []
    result: list[tuple[int, float]] = []
    for idx in indices:
        value = values[idx]
        result.append((idx, _to_scalar(value)))
    return result


def _flatten(values: Any) -> Iterable[float]:
    if isinstance(values, (list, tuple)):
        for item in values:
            yield from _flatten(item)
    else:
        yield _to_scalar(values)


def _to_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
