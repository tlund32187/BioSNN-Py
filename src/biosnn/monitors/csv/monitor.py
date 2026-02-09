"""CSV monitors for neuron and synapse outputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
from biosnn.io.sinks import AsyncCsvSink, BufferedCsvSink, CsvSink
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
        safe_sample: int | None = None,
        flush_every: int = 1,
        every_n_steps: int = 1,
        append: bool = False,
        async_gpu: bool = False,
        async_io: bool = False,
    ) -> None:
        self._tensor_keys = list(tensor_keys) if tensor_keys is not None else None
        self._stats = tuple(stats) if stats is not None else _DEFAULT_STATS
        self._include_spikes = include_spikes
        self._include_scalars = include_scalars
        self._sample_indices = list(sample_indices) if sample_indices is not None else None
        self._safe_sample = int(safe_sample) if safe_sample and safe_sample > 0 else None
        self._flush_every = max(1, flush_every)
        self._every_n_steps = max(1, every_n_steps)
        self._event_count = 0
        self._async_gpu = bool(async_gpu)
        self._async_io = bool(async_io)
        sink_cls: type[CsvSink] | type[BufferedCsvSink] | type[AsyncCsvSink]
        if self._async_io:
            sink_cls = AsyncCsvSink
        elif self._async_gpu:
            sink_cls = BufferedCsvSink
        else:
            sink_cls = CsvSink
        self._sink = sink_cls(path, flush_every=self._flush_every, append=append)

    def on_step(self, event: StepEvent) -> None:
        self._event_count += 1
        if self._event_count % self._every_n_steps != 0:
            return

        row: dict[str, Any] = {"t": event.t, "dt": event.dt}
        sample_indices = self._sample_indices
        if sample_indices is None and self._safe_sample is not None:
            sample_indices = self._maybe_set_safe_sample(event)

        if self._include_spikes and event.spikes is not None:
            spike_count = reduce_stat(event.spikes, "sum", as_tensor=self._async_gpu)
            spike_fraction = reduce_stat(event.spikes, "mean", as_tensor=self._async_gpu)
            row["spike_count"] = spike_count
            row["spike_fraction"] = spike_fraction
            row["spike_rate_hz"] = (spike_fraction / event.dt) if event.dt else 0.0
            if sample_indices is not None:
                for idx, sample_value in sample_tensor(
                    event.spikes, sample_indices, as_tensor=self._async_gpu
                ):
                    row[f"spike_i{idx}"] = sample_value

        if self._include_scalars and event.scalars:
            for key, scalar_value in sorted(event.scalars.items()):
                row[key] = _scalar_for_row(scalar_value, self._async_gpu)

        tensors = event.tensors or {}
        if self._tensor_keys is None:
            self._tensor_keys = sorted(tensors.keys())

        for key in self._tensor_keys:
            tensor = tensors.get(key)
            if tensor is None:
                continue
            row.update(
                reduce_tensor(
                    key,
                    tensor,
                    self._stats,
                    sample_indices,
                    as_tensor=self._async_gpu,
                )
            )

        self._sink.write_row(row)

    def _maybe_set_safe_sample(self, event: StepEvent) -> list[int] | None:
        if self._safe_sample is None:
            return None
        sample_target = _infer_sample_target(event)
        if sample_target is None:
            return None
        if sample_target <= self._safe_sample:
            return None
        sample_indices = list(range(self._safe_sample))
        self._sample_indices = sample_indices
        return sample_indices

    def requirements(self) -> MonitorRequirements:
        if self._tensor_keys is None:
            return MonitorRequirements.all()

        needs_population_state = False
        needs_v_soma = False
        needs_projection_weights = False
        needs_synapse_state = False
        needs_modulators = False
        needs_learning_state = False
        needs_homeostasis_state = False

        for key in self._tensor_keys:
            key_str = str(key)
            if key_str == "weights":
                needs_projection_weights = True
                needs_synapse_state = True
                continue
            if key_str == "v_soma" or key_str.endswith("/v_soma"):
                needs_v_soma = True
                needs_population_state = True
                continue
            if key_str.startswith("proj/"):
                if key_str.endswith("/weights"):
                    needs_projection_weights = True
                else:
                    needs_synapse_state = True
                continue
            if key_str.startswith("learn/"):
                needs_learning_state = True
                continue
            if key_str.startswith("mod/"):
                needs_modulators = True
                continue
            if key_str.startswith("homeostasis/"):
                needs_homeostasis_state = True
                continue
            needs_population_state = True

        return MonitorRequirements(
            needs_spikes=bool(self._include_spikes),
            needs_v_soma=needs_v_soma,
            needs_projection_weights=needs_projection_weights,
            needs_synapse_state=needs_synapse_state,
            needs_modulators=needs_modulators,
            needs_learning_state=needs_learning_state,
            needs_homeostasis_state=needs_homeostasis_state,
            needs_population_state=needs_population_state,
            needs_scalars=bool(self._include_scalars),
            needs_population_slices=False,
        )

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
        async_gpu: bool = False,
        async_io: bool = False,
    ) -> None:
        super().__init__(
            path,
            tensor_keys=tensor_keys or ("weights",),
            stats=stats,
            include_spikes=include_spikes,
            include_scalars=include_scalars,
            sample_indices=sample_indices,
            safe_sample=None,
            flush_every=flush_every,
            every_n_steps=every_n_steps,
            append=append,
            async_gpu=async_gpu,
            async_io=async_io,
        )


def _scalar_for_row(value: Any, async_gpu: bool) -> Any:
    if not async_gpu:
        return scalar_to_float(value)
    if hasattr(value, "detach"):
        return value.detach()
    return value


def _infer_sample_target(event: StepEvent) -> int | None:
    spikes = event.spikes
    if spikes is not None:
        return _tensor_len(spikes)
    tensors = event.tensors or {}
    for tensor in tensors.values():
        target = _tensor_len(tensor)
        if target is not None:
            return target
    return None


def _tensor_len(value: Any) -> int | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            pass
    if hasattr(value, "numel"):
        try:
            return int(value.numel())
        except (TypeError, ValueError):
            return None
    return None
