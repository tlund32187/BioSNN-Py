"""Projection weights CSV monitor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from biosnn.contracts.monitors import IMonitor, StepEvent
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.io.sinks.csv_sink import CsvSink
from biosnn.monitors.metrics.scalar_utils import scalar_to_float


@dataclass(frozen=True, slots=True)
class _ProjectionMeta:
    name: str
    pre_idx: Tensor
    post_idx: Tensor


class ProjectionWeightsCSVMonitor(IMonitor):
    """Write projection weight snapshots to CSV."""

    name = "projection_weights_csv"

    def __init__(
        self,
        path: str,
        *,
        projections: Sequence[Any],
        stride: int = 1,
        max_edges_full: int = 20000,
        max_edges_sample: int = 20000,
        safe_max_edges_sample: int | None = None,
        append: bool = False,
        flush_every: int = 1,
    ) -> None:
        self._sink = CsvSink(
            path,
            fieldnames=["step", "t", "proj", "pre", "post", "w"],
            append=append,
            flush_every=flush_every,
        )
        self._stride = max(1, stride)
        self._max_edges_full = max_edges_full
        safe_cap = int(safe_max_edges_sample) if safe_max_edges_sample else None
        if safe_cap is not None and safe_cap > 0:
            self._max_edges_sample = min(max_edges_sample, safe_cap)
        else:
            self._max_edges_sample = max_edges_sample
        self._projections = [_normalize_projection(proj) for proj in projections]

    def on_step(self, event: StepEvent) -> None:
        step = int(scalar_to_float(event.scalars.get("step", 0))) if event.scalars else 0
        if step % self._stride != 0:
            return
        if not event.tensors:
            return

        torch = require_torch()
        for proj in self._projections:
            name = proj.name
            key = f"proj/{name}/weights"
            weights = event.tensors.get(key)
            if weights is None:
                continue
            pre_idx = proj.pre_idx
            post_idx = proj.post_idx
            edge_count = int(pre_idx.shape[0])
            if edge_count == 0:
                continue

            if edge_count <= self._max_edges_full:
                indices = torch.arange(edge_count, device=weights.device)
            else:
                cap = min(edge_count, self._max_edges_sample)
                perm = torch.randperm(edge_count, device=weights.device)
                indices = perm[:cap]

            w_sel = weights.index_select(0, indices).detach().cpu().tolist()
            pre_sel = pre_idx.index_select(0, indices).detach().cpu().tolist()
            post_sel = post_idx.index_select(0, indices).detach().cpu().tolist()

            for pre, post, weight in zip(pre_sel, post_sel, w_sel, strict=True):
                self._sink.write_row(
                    {
                        "step": step,
                        "t": event.t,
                        "proj": name,
                        "pre": int(pre),
                        "post": int(post),
                        "w": float(weight),
                    }
                )

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


def _normalize_projection(proj: Any) -> _ProjectionMeta:
    if hasattr(proj, "name") and hasattr(proj, "topology"):
        return _ProjectionMeta(
            name=str(proj.name),
            pre_idx=proj.topology.pre_idx,
            post_idx=proj.topology.post_idx,
        )
    if isinstance(proj, Mapping):
        topo = proj.get("topology")
        if topo is None:
            raise ValueError("Projection metadata must include topology")
        return _ProjectionMeta(
            name=str(proj.get("name") or proj.get("id") or "projection"),
            pre_idx=topo.pre_idx,
            post_idx=topo.post_idx,
        )
    raise ValueError("Unsupported projection metadata")


__all__ = ["ProjectionWeightsCSVMonitor"]
