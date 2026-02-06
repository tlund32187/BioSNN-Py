"""Prototype event-list ring buffer for sparse delayed events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch


@dataclass(slots=True)
class EventListRing:
    depth: int
    device: Any
    dtype: Any
    max_events: int
    due_row: Tensor
    post_idx: Tensor
    values: Tensor

    def __init__(self, *, depth: int, device: Any, dtype: Any, max_events: int) -> None:
        torch = require_torch()
        self.depth = int(depth)
        self.device = device
        self.dtype = dtype
        self.max_events = int(max_events)
        self.due_row = torch.empty((0,), device=device, dtype=torch.long)
        self.post_idx = torch.empty((0,), device=device, dtype=torch.long)
        self.values = torch.empty((0,), device=device, dtype=dtype)

    def clear(self) -> None:
        if self.due_row.numel() == 0:
            return
        self.due_row = self.due_row[:0]
        self.post_idx = self.post_idx[:0]
        self.values = self.values[:0]

    def schedule(self, due_rows: Tensor, post_idx: Tensor, values: Tensor) -> None:
        if due_rows.numel() == 0:
            return
        torch = require_torch()
        if due_rows.device != self.device or due_rows.dtype != torch.long:
            due_rows = due_rows.to(device=self.device, dtype=torch.long)
        if post_idx.device != self.device or post_idx.dtype != torch.long:
            post_idx = post_idx.to(device=self.device, dtype=torch.long)
        if values.device != self.device or values.dtype != self.dtype:
            values = values.to(device=self.device, dtype=self.dtype)
        new_total = int(self.due_row.numel() + due_rows.numel())
        if new_total > self.max_events:
            raise RuntimeError(
                f"EventListRing exceeded max_events_total={self.max_events} (attempted {new_total})."
            )
        self.due_row = torch.cat([self.due_row, due_rows], dim=0)
        self.post_idx = torch.cat([self.post_idx, post_idx], dim=0)
        self.values = torch.cat([self.values, values], dim=0)

    def pop_into(self, cursor_row: int, out_drive: Tensor) -> None:
        if self.due_row.numel() == 0:
            return
        mask = self.due_row == int(cursor_row)
        idx = mask.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        posts = self.post_idx.index_select(0, idx)
        vals = self.values.index_select(0, idx)
        if vals.dtype != out_drive.dtype:
            vals = vals.to(dtype=out_drive.dtype)
        out_drive.index_add_(0, posts, vals)
        keep = ~mask
        self.due_row = self.due_row[keep]
        self.post_idx = self.post_idx[keep]
        self.values = self.values[keep]


__all__ = ["EventListRing"]
