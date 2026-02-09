"""Homeostasis metrics CSV monitor."""

from __future__ import annotations

from typing import Any

from biosnn.contracts.monitors import IMonitor, StepEvent
from biosnn.io.sinks import AsyncCsvSink, BufferedCsvSink, CsvSink
from biosnn.monitors.metrics.scalar_utils import scalar_to_float


class HomeostasisCSVMonitor(IMonitor):
    """Write per-population homeostasis summaries to CSV."""

    name = "homeostasis_csv"

    def __init__(
        self,
        path: str,
        *,
        stride: int = 1,
        append: bool = False,
        flush_every: int = 256,
        async_gpu: bool = False,
        async_io: bool = False,
    ) -> None:
        self._async_gpu = bool(async_gpu)
        self._async_io = bool(async_io)
        sink_cls: type[CsvSink] | type[BufferedCsvSink] | type[AsyncCsvSink]
        if self._async_io:
            sink_cls = AsyncCsvSink
        elif self._async_gpu:
            sink_cls = BufferedCsvSink
        else:
            sink_cls = CsvSink
        self._sink = sink_cls(
            path,
            fieldnames=[
                "step",
                "t",
                "population",
                "rate_mean",
                "target_rate",
                "control_mean",
            ],
            append=append,
            flush_every=flush_every,
        )
        self._stride = max(1, int(stride))

    def on_step(self, event: StepEvent) -> None:
        step = _step_from_event(event, async_gpu=self._async_gpu)
        if step % self._stride != 0:
            return
        if not event.scalars:
            return

        pops = _extract_populations(event)
        if not pops:
            return

        for pop_name in pops:
            row = {
                "step": step,
                "t": float(event.t),
                "population": pop_name,
                "rate_mean": _scalar(
                    event,
                    f"homeostasis/rate_mean/{pop_name}",
                    async_gpu=self._async_gpu,
                ),
                "target_rate": _scalar(
                    event,
                    f"homeostasis/target_rate/{pop_name}",
                    async_gpu=self._async_gpu,
                ),
                "control_mean": _scalar(
                    event,
                    f"homeostasis/control_mean/{pop_name}",
                    async_gpu=self._async_gpu,
                ),
            }
            self._sink.write_row(row)

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


def _extract_populations(event: StepEvent) -> list[str]:
    if not event.scalars:
        return []
    pops: set[str] = set()
    for key in event.scalars:
        prefix = "homeostasis/rate_mean/"
        if key.startswith(prefix):
            pops.add(key[len(prefix) :])
    return sorted(pops)


def _scalar(event: StepEvent, key: str, *, async_gpu: bool) -> Any:
    if not event.scalars or key not in event.scalars:
        return ""
    value = event.scalars[key]
    if async_gpu:
        if hasattr(value, "detach"):
            return value.detach()
        return value
    return scalar_to_float(value)


def _step_from_event(event: StepEvent, *, async_gpu: bool) -> int:
    if not event.scalars or "step" not in event.scalars:
        return 0
    step_val = event.scalars["step"]
    if isinstance(step_val, (int, float)):
        return int(step_val)
    if async_gpu:
        return 0
    return int(scalar_to_float(step_val))


__all__ = ["HomeostasisCSVMonitor"]

