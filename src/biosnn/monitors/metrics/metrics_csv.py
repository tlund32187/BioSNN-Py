"""Metrics CSV monitor."""

from __future__ import annotations

from typing import Any

from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
from biosnn.io.sinks import AsyncCsvSink, BufferedCsvSink, CsvSink
from biosnn.monitors.metrics.scalar_utils import scalar_to_float
from biosnn.monitors.metrics.tensor_reducer import reduce_stat


class MetricsCSVMonitor(IMonitor):
    """Write scalar metrics to CSV."""

    name = "metrics_csv"

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
                "spike_count_total",
                "spike_fraction_total",
                "train_accuracy",
                "eval_accuracy",
                "loss",
                "reward",
            ],
            append=append,
            flush_every=flush_every,
        )
        self._stride = max(1, stride)

    def on_step(self, event: StepEvent) -> None:
        step = _step_from_event(event, async_gpu=self._async_gpu)
        if step % self._stride != 0:
            return

        spike_count = _value_from_scalars(event, "spike_count_total", as_tensor=self._async_gpu)
        spike_fraction = _value_from_scalars(
            event, "spike_fraction_total", as_tensor=self._async_gpu
        )

        if (
            (spike_count is None or spike_fraction is None)
            and event.spikes is not None
            and hasattr(event.spikes, "numel")
        ):
            if spike_count is None:
                spike_count = reduce_stat(event.spikes, "sum", as_tensor=self._async_gpu)
            if spike_fraction is None:
                spike_fraction = reduce_stat(event.spikes, "mean", as_tensor=self._async_gpu)

        row = {
            "step": step,
            "t": event.t,
            "spike_count_total": _safe_value(spike_count, as_tensor=self._async_gpu),
            "spike_fraction_total": _safe_value(spike_fraction, as_tensor=self._async_gpu),
            "train_accuracy": _safe_value(
                _value_from_scalars(event, "train_accuracy", as_tensor=self._async_gpu),
                as_tensor=self._async_gpu,
            ),
            "eval_accuracy": _safe_value(
                _value_from_scalars(event, "eval_accuracy", as_tensor=self._async_gpu),
                as_tensor=self._async_gpu,
            ),
            "loss": _safe_value(
                _value_from_scalars(event, "loss", as_tensor=self._async_gpu),
                as_tensor=self._async_gpu,
            ),
            "reward": _safe_value(
                _value_from_scalars(event, "reward", as_tensor=self._async_gpu),
                as_tensor=self._async_gpu,
            ),
        }
        self._sink.write_row(row)

    def record_eval(self, *, step: int, t: float, eval_accuracy: float) -> None:
        self._sink.write_row(
            {
                "step": int(step),
                "t": float(t),
                "spike_count_total": "",
                "spike_fraction_total": "",
                "train_accuracy": "",
                "eval_accuracy": float(eval_accuracy),
                "loss": "",
                "reward": "",
            }
        )

    def requirements(self) -> MonitorRequirements:
        return MonitorRequirements(needs_scalars=True)

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


def _value_from_scalars(event: StepEvent, key: str, *, as_tensor: bool) -> Any | None:
    if event.scalars and key in event.scalars:
        value = event.scalars[key]
        if as_tensor:
            _detach = getattr(value, "detach", None)
            if callable(_detach):
                return _detach()
            return value
        return scalar_to_float(value)
    return None


def _safe_value(value: Any | None, *, as_tensor: bool) -> Any:
    if value is None:
        return ""
    if as_tensor:
        return value
    if isinstance(value, (int, float)):
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


__all__ = ["MetricsCSVMonitor"]
