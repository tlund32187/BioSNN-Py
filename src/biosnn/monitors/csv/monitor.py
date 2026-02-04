"""CSV monitors for neuron and synapse outputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from biosnn.contracts.monitors import IMonitor, StepEvent
from biosnn.io.sinks import CsvSink
from biosnn.monitors.metrics import reduce_stat, reduce_tensor, sample_tensor, scalar_to_float

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
        every_n_steps: int = 1,
        append: bool = False,
    ) -> None:
        self._tensor_keys = list(tensor_keys) if tensor_keys is not None else None
        self._stats = tuple(stats) if stats is not None else _DEFAULT_STATS
        self._include_spikes = include_spikes
        self._include_scalars = include_scalars
        self._sample_indices = list(sample_indices) if sample_indices is not None else None
        self._flush_every = max(1, flush_every)
        self._every_n_steps = max(1, every_n_steps)
        self._event_count = 0
        self._sink = CsvSink(path, flush_every=self._flush_every, append=append)

    def on_step(self, event: StepEvent) -> None:
        self._event_count += 1
        if self._event_count % self._every_n_steps != 0:
            return

        row: dict[str, Any] = {"t": event.t, "dt": event.dt}

        if self._include_spikes and event.spikes is not None:
            spike_count = reduce_stat(event.spikes, "sum")
            spike_fraction = reduce_stat(event.spikes, "mean")
            row["spike_count"] = spike_count
            row["spike_fraction"] = spike_fraction
            row["spike_rate_hz"] = (spike_fraction / event.dt) if event.dt else 0.0
            if self._sample_indices is not None:
                for idx, sample_value in sample_tensor(event.spikes, self._sample_indices):
                    row[f"spike_i{idx}"] = sample_value

        if self._include_scalars and event.scalars:
            for key, scalar_value in sorted(event.scalars.items()):
                row[key] = scalar_to_float(scalar_value)

        tensors = event.tensors or {}
        if self._tensor_keys is None:
            self._tensor_keys = sorted(tensors.keys())

        for key in self._tensor_keys:
            tensor = tensors.get(key)
            if tensor is None:
                continue
            row.update(reduce_tensor(key, tensor, self._stats, self._sample_indices))

        self._sink.write_row(row)

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


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


class SynapseCSVMonitor(NeuronCSVMonitor):
    """CSV monitor preset for synapse tensors."""

    name = "csv_synapse"

    def __init__(
        self,
        path: str | Path,
        *,
        tensor_keys: Sequence[str] | None = None,
        stats: Sequence[str] | None = None,
        include_spikes: bool = False,
        include_scalars: bool = True,
        sample_indices: Sequence[int] | None = None,
        flush_every: int = 1,
        every_n_steps: int = 1,
        append: bool = False,
    ) -> None:
        super().__init__(
            path,
            tensor_keys=tensor_keys or ("weights",),
            stats=stats,
            include_spikes=include_spikes,
            include_scalars=include_scalars,
            sample_indices=sample_indices,
            flush_every=flush_every,
            every_n_steps=every_n_steps,
            append=append,
        )
