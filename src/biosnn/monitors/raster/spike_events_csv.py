"""Spike events CSV monitor."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.io.sinks import AsyncCsvSink, CsvSink
from biosnn.monitors.metrics.scalar_utils import scalar_to_float


class SpikeEventsCSVMonitor(IMonitor):
    """Write sparse spike events to CSV.

    Columns: step, t, pop, neuron
    """

    name = "spike_events_csv"

    def __init__(
        self,
        path: str,
        *,
        stride: int = 1,
        max_spikes_per_step: int | None = None,
        neuron_sample: int | None = None,
        safe_neuron_sample: int | None = None,
        allow_cuda_sync: bool = False,
        append: bool = False,
        flush_every: int = 1,
        async_io: bool = False,
    ) -> None:
        if neuron_sample is None and safe_neuron_sample:
            neuron_sample = int(safe_neuron_sample)
        sink_cls = AsyncCsvSink if async_io else CsvSink
        self._sink = sink_cls(
            path,
            fieldnames=["step", "t", "pop", "neuron"],
            append=append,
            flush_every=flush_every,
        )
        self._stride = max(1, stride)
        self._max_spikes = max_spikes_per_step
        self._neuron_sample = neuron_sample
        self._sample_indices: dict[str, Tensor] = {}
        self._sample_mask: dict[str, Tensor] = {}
        self._allow_cuda_sync = bool(allow_cuda_sync)
        self._warned_cuda = False

    def on_step(self, event: StepEvent) -> None:
        if event.spikes is None:
            return
        torch = require_torch()
        if (
            hasattr(event.spikes, "device")
            and getattr(event.spikes.device, "type", None) == "cuda"
            and not self._allow_cuda_sync
        ):
            if not self._warned_cuda:
                warnings.warn(
                    "SpikeEventsCSVMonitor disabled on CUDA unless allow_cuda_sync=True",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_cuda = True
            return
        step = int(scalar_to_float(event.scalars.get("step", 0))) if event.scalars else 0
        if step % self._stride != 0:
            return

        spikes = event.spikes
        population_slices = _population_slices(event.meta)

        if population_slices:
            for pop, (start, end) in population_slices.items():
                pop_spikes = spikes[start:end]
                if getattr(pop_spikes, "dtype", None) == torch.bool:
                    indices = pop_spikes.nonzero(as_tuple=False).flatten()
                else:
                    indices = (pop_spikes > 0.5).nonzero(as_tuple=False).flatten()
                indices = self._filter_indices(pop, indices, end - start)
                indices = _cap_indices(indices, self._max_spikes)
                rows = _indices_to_rows(indices, step=step, t=event.t, pop=pop)
                _write_rows(self._sink, rows)
            return

        if getattr(spikes, "dtype", None) == torch.bool:
            indices = spikes.nonzero(as_tuple=False).flatten()
        else:
            indices = (spikes > 0.5).nonzero(as_tuple=False).flatten()
        indices = self._filter_indices("pop0", indices, int(spikes.shape[0]))
        indices = _cap_indices(indices, self._max_spikes)
        rows = _indices_to_rows(indices, step=step, t=event.t, pop="pop0")
        _write_rows(self._sink, rows)

    def _filter_indices(self, pop: str, indices: Tensor, n: int) -> Tensor:
        if self._neuron_sample is None:
            return indices
        torch = require_torch()
        if pop not in self._sample_indices:
            sample_n = min(self._neuron_sample, n)
            perm = torch.randperm(n, device=indices.device)
            sample_idx = perm[:sample_n]
            self._sample_indices[pop] = sample_idx
            mask = torch.zeros((n,), device=indices.device, dtype=torch.bool)
            mask[sample_idx] = True
            self._sample_mask[pop] = mask
        mask = self._sample_mask[pop]
        return indices[mask[indices]]

    def requirements(self) -> MonitorRequirements:
        return MonitorRequirements(
            needs_spikes=True,
            needs_scalars=True,
            needs_population_slices=True,
        )

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


def _population_slices(meta: Mapping[str, Any] | None) -> dict[str, tuple[int, int]]:
    if not meta:
        return {}
    raw = meta.get("population_slices")
    if isinstance(raw, Mapping):
        return {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}
    return {}


def _cap_indices(indices: Tensor, cap: int | None) -> Tensor:
    if cap is None:
        return indices
    if indices.numel() <= cap:
        return indices
    torch = require_torch()
    perm = torch.randperm(indices.numel(), device=indices.device)
    return indices[perm[:cap]]


def _indices_to_rows(indices: Tensor, *, step: int, t: float, pop: str) -> list[dict[str, Any]]:
    if (
        hasattr(indices, "device")
        and getattr(indices.device, "type", None) != "cpu"
        and hasattr(indices, "to")
    ):
        indices = indices.to("cpu", non_blocking=True)
    indices_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    return [{"step": step, "t": t, "pop": pop, "neuron": int(idx)} for idx in indices_list]


def _write_rows(sink: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    if hasattr(sink, "write_rows"):
        sink.write_rows(rows)
        return
    for row in rows:
        sink.write_row(row)


__all__ = ["SpikeEventsCSVMonitor"]
