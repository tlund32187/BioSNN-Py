"""Metrics CSV monitor."""

from __future__ import annotations

from typing import Any

from biosnn.contracts.monitors import IMonitor, StepEvent
from biosnn.io.sinks.csv_sink import CsvSink
from biosnn.monitors.metrics.scalar_utils import scalar_to_float


class MetricsCSVMonitor(IMonitor):
    """Write scalar metrics to CSV."""

    name = "metrics_csv"

    def __init__(
        self,
        path: str,
        *,
        stride: int = 1,
        append: bool = False,
        flush_every: int = 1,
    ) -> None:
        self._sink = CsvSink(
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
        step = int(scalar_to_float(event.scalars.get("step", 0))) if event.scalars else 0
        if step % self._stride != 0:
            return

        spike_count = _value_from_scalars(event, "spike_count_total")
        spike_fraction = _value_from_scalars(event, "spike_fraction_total")

        if (spike_count is None or spike_fraction is None) and event.spikes is not None and hasattr(
            event.spikes, "numel"
        ):
            spikes = event.spikes
            if (
                getattr(spikes, "dtype", None) is not None
                and "bool" in str(spikes.dtype).lower()
                and hasattr(spikes, "float")
            ):
                spikes = spikes.float()
            total = float(spikes.sum().item()) if spikes.numel() else 0.0
            frac = float(spikes.mean().item()) if spikes.numel() else 0.0
            spike_count = total if spike_count is None else spike_count
            spike_fraction = frac if spike_fraction is None else spike_fraction

        row = {
            "step": step,
            "t": event.t,
            "spike_count_total": _safe_value(spike_count),
            "spike_fraction_total": _safe_value(spike_fraction),
            "train_accuracy": _safe_value(_value_from_scalars(event, "train_accuracy")),
            "eval_accuracy": _safe_value(_value_from_scalars(event, "eval_accuracy")),
            "loss": _safe_value(_value_from_scalars(event, "loss")),
            "reward": _safe_value(_value_from_scalars(event, "reward")),
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

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


def _value_from_scalars(event: StepEvent, key: str) -> float | None:
    if event.scalars and key in event.scalars:
        return scalar_to_float(event.scalars[key])
    return None


def _safe_value(value: float | None) -> Any:
    if value is None:
        return ""
    return value


__all__ = ["MetricsCSVMonitor"]
