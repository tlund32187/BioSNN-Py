"""CSV monitor for receptor summary statistics."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
from biosnn.contracts.tensor import Tensor
from biosnn.io.sinks import CsvSink
from biosnn.monitors.metrics.scalar_utils import scalar_to_float

_RECEPTORS = ("ampa", "nmda", "gabaa", "gabab")
_STATS = ("mean", "max")

_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^proj/(?P<proj>[^/]+)/receptors/(?P<comp>[^/]+)/"
        r"(?P<receptor>ampa|nmda|gabaa|gabab)_(?P<stat>mean|max)$"
    ),
    re.compile(
        r"^proj/(?P<proj>[^/]+)/receptor/(?P<comp>[^/]+)/"
        r"(?P<receptor>ampa|nmda|gabaa|gabab)_(?P<stat>mean|max)$"
    ),
    re.compile(
        r"^proj/(?P<proj>[^/]+)/(?P<comp>[^/]+)/"
        r"(?P<receptor>ampa|nmda|gabaa|gabab)_(?P<stat>mean|max)$"
    ),
    re.compile(
        r"^proj/(?P<proj>[^/]+)/receptor_(?P<comp>[^_]+)_"
        r"(?P<receptor>ampa|nmda|gabaa|gabab)_(?P<stat>mean|max)$"
    ),
)


class ReceptorSummaryCsvMonitor(IMonitor):
    """Write per-projection receptor summaries to `receptors.csv`."""

    name = "receptor_summary_csv"

    def __init__(
        self,
        path: str | Path,
        *,
        stride: int = 1,
        include_max: bool = True,
        allow_cuda_sync: bool = False,
        append: bool = False,
        flush_every: int = 25,
    ) -> None:
        self._stride = max(1, int(stride))
        self._include_max = bool(include_max)
        self._allow_cuda_sync = bool(allow_cuda_sync)
        self._warned_cuda = False
        fieldnames = [
            "step",
            "t",
            "proj",
            "comp",
            "ampa_mean",
            "nmda_mean",
            "gabaa_mean",
            "gabab_mean",
        ]
        if self._include_max:
            fieldnames.extend(["ampa_max", "nmda_max", "gabaa_max", "gabab_max"])
        self._sink = CsvSink(
            str(path),
            fieldnames=fieldnames,
            append=append,
            flush_every=max(1, int(flush_every)),
        )

    def on_step(self, event: StepEvent) -> None:
        step = int(scalar_to_float(event.scalars.get("step", 0))) if event.scalars else 0
        if step % self._stride != 0:
            return
        if not event.tensors:
            return

        rows_by_proj_comp: dict[tuple[str, str], dict[str, float]] = {}
        for key, tensor in event.tensors.items():
            parsed = _parse_receptor_key(key)
            if parsed is None:
                continue
            proj, comp, receptor, stat = parsed
            value = self._reduce_tensor(tensor, stat=stat)
            if value is None:
                continue
            rows_by_proj_comp.setdefault((proj, comp), {})[f"{receptor}_{stat}"] = value

        for (proj, comp), values in sorted(rows_by_proj_comp.items()):
            row: dict[str, Any] = {"step": step, "t": event.t, "proj": proj, "comp": comp}
            for receptor in _RECEPTORS:
                row[f"{receptor}_mean"] = values.get(f"{receptor}_mean", "")
            if self._include_max:
                for receptor in _RECEPTORS:
                    row[f"{receptor}_max"] = values.get(f"{receptor}_max", "")
            self._sink.write_row(row)

    def requirements(self) -> MonitorRequirements:
        return MonitorRequirements(
            needs_synapse_state=True,
            needs_scalars=True,
        )

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()

    def _reduce_tensor(self, value: Tensor, *, stat: str) -> float | None:
        if hasattr(value, "numel") and int(value.numel()) == 0:
            return None
        if (
            hasattr(value, "device")
            and getattr(value.device, "type", None) == "cuda"
            and not self._allow_cuda_sync
        ):
            if not self._warned_cuda:
                warnings.warn(
                    "ReceptorSummaryCsvMonitor disabled on CUDA unless allow_cuda_sync=True",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_cuda = True
            return None
        sample = value.detach()
        reduced = sample.max() if stat == "max" else sample.mean()
        return float(scalar_to_float(reduced))


def _parse_receptor_key(key: str) -> tuple[str, str, str, str] | None:
    for pattern in _KEY_PATTERNS:
        match = pattern.match(key)
        if match is None:
            continue
        proj = match.group("proj")
        comp = match.group("comp")
        receptor = match.group("receptor")
        stat = match.group("stat")
        if receptor not in _RECEPTORS:
            continue
        if stat not in _STATS:
            continue
        return proj, comp, receptor, stat
    return None


__all__ = ["ReceptorSummaryCsvMonitor"]
