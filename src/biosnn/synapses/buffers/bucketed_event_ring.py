"""CUDA-safe bucketed event ring with bounded flat storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch


@dataclass(slots=True)
class BucketedEventRing:
    """Sparse delayed-event ring using bounded flat tensors.

    Events are stored in fixed-capacity arrays on device, then grouped by slot using
    a stable argsort. Pop remains O(N) in this baseline implementation by filtering
    events matching the requested slot and compacting in place.
    """

    depth: int
    device: Any
    dtype: Any
    capacity_max: int
    max_events: int
    due_slot: Tensor
    post_idx: Tensor
    values: Tensor
    slot_counts: Tensor
    slot_offsets: Tensor
    active_count: int

    def __init__(
        self,
        *,
        depth: int,
        device: Any,
        dtype: Any,
        capacity_max: int | None = None,
        max_events: int | None = None,
    ) -> None:
        torch = require_torch()
        self.depth = int(depth)
        self.device = device
        self.dtype = dtype
        if capacity_max is None:
            if max_events is None:
                raise ValueError("Either capacity_max or max_events must be provided")
            capacity_max = int(max_events)
        self.capacity_max = int(capacity_max)
        self.max_events = self.capacity_max
        if self.capacity_max <= 0:
            raise ValueError("capacity_max must be > 0")
        if self.depth <= 0:
            raise ValueError("depth must be > 0")
        self.due_slot = torch.empty((self.capacity_max,), device=device, dtype=torch.long)
        self.post_idx = torch.empty((self.capacity_max,), device=device, dtype=torch.long)
        self.values = torch.empty((self.capacity_max,), device=device, dtype=dtype)
        self.slot_counts = torch.zeros((self.depth,), device=device, dtype=torch.long)
        self.slot_offsets = torch.zeros((self.depth + 1,), device=device, dtype=torch.long)
        self.active_count = 0

    def clear(self) -> None:
        if self.active_count == 0:
            return
        self.active_count = 0
        self.slot_counts.zero_()
        self.slot_offsets.zero_()

    def schedule(self, due_rows: Tensor, post_idx: Tensor, values: Tensor) -> None:
        if due_rows.numel() == 0:
            return
        torch = require_torch()
        due_rows = self._normalize_due_rows(due_rows, torch=torch)
        post_idx = self._normalize_post_idx(post_idx, torch=torch)
        values = self._normalize_values(values)

        if due_rows.dim() != 1 or post_idx.dim() != 1 or values.dim() != 1:
            raise ValueError("due_rows/post_idx/values must be 1D tensors")
        if due_rows.numel() != post_idx.numel() or due_rows.numel() != values.numel():
            raise ValueError("due_rows/post_idx/values must have matching lengths")

        add_count = int(due_rows.numel())
        new_total = self.active_count + add_count
        if new_total > self.capacity_max:
            raise RuntimeError(
                f"BucketedEventRing exceeded capacity_max={self.capacity_max} (attempted {new_total})."
            )

        start = self.active_count
        end = new_total
        self.due_slot[start:end].copy_(due_rows)
        self.post_idx[start:end].copy_(post_idx)
        self.values[start:end].copy_(values)
        self.active_count = new_total
        self._rebuild_slot_index(torch=torch)

    def pop_into(self, cursor_row: int, out_drive: Tensor) -> None:
        if self.active_count == 0:
            return

        torch = require_torch()
        cursor = int(cursor_row) % self.depth
        active_due = self.due_slot[: self.active_count]
        match = active_due == cursor
        selected = match.nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            return

        posts = self.post_idx[: self.active_count].index_select(0, selected)
        vals = self.values[: self.active_count].index_select(0, selected)
        if vals.dtype != out_drive.dtype:
            vals = vals.to(dtype=out_drive.dtype)
        out_drive.index_add_(0, posts, vals)

        keep = (~match).nonzero(as_tuple=False).flatten()
        remaining = int(keep.numel())
        if remaining > 0:
            self.due_slot[:remaining].copy_(active_due.index_select(0, keep))
            self.post_idx[:remaining].copy_(self.post_idx[: self.active_count].index_select(0, keep))
            self.values[:remaining].copy_(self.values[: self.active_count].index_select(0, keep))
        self.active_count = remaining
        self._rebuild_slot_index(torch=torch)

    def _rebuild_slot_index(self, *, torch: Any) -> None:
        self.slot_counts.zero_()
        self.slot_offsets.zero_()
        if self.active_count == 0:
            return

        active_due = self.due_slot[: self.active_count]
        order = torch.argsort(active_due, stable=True)
        if order.numel():
            sorted_due = active_due.index_select(0, order)
            sorted_post = self.post_idx[: self.active_count].index_select(0, order)
            sorted_vals = self.values[: self.active_count].index_select(0, order)
            self.due_slot[: self.active_count].copy_(sorted_due)
            self.post_idx[: self.active_count].copy_(sorted_post)
            self.values[: self.active_count].copy_(sorted_vals)
            active_due = self.due_slot[: self.active_count]

        counts = torch.bincount(active_due, minlength=self.depth)
        self.slot_counts.copy_(counts)
        self.slot_offsets[1:].copy_(torch.cumsum(counts, dim=0))

    def _normalize_due_rows(self, due_rows: Tensor, *, torch: Any) -> Tensor:
        if due_rows.device != self.device or due_rows.dtype != torch.long:
            due_rows = due_rows.to(device=self.device, dtype=torch.long)
        return cast(Tensor, torch.remainder(due_rows, self.depth))

    def _normalize_post_idx(self, post_idx: Tensor, *, torch: Any) -> Tensor:
        if post_idx.device != self.device or post_idx.dtype != torch.long:
            post_idx = post_idx.to(device=self.device, dtype=torch.long)
        return post_idx

    def _normalize_values(self, values: Tensor) -> Tensor:
        if values.device != self.device or values.dtype != self.dtype:
            values = values.to(device=self.device, dtype=self.dtype)
        return values


__all__ = ["BucketedEventRing"]
