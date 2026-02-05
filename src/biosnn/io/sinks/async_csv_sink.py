"""Async CSV sink writing rows on a background thread."""

from __future__ import annotations

import csv
import queue
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:  # optional torch
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

_SENTINEL = object()


class AsyncCsvSink:
    """Write CSV rows on a background thread using a bounded queue."""

    def __init__(
        self,
        path: str | Path,
        fieldnames: Iterable[str] | None = None,
        *,
        max_queue: int = 10_000,
        flush_every: int = 1000,
        append: bool = False,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = max(1, int(flush_every))
        self._append = append
        self._fieldnames = list(fieldnames) if fieldnames is not None else None
        self._writer: csv.DictWriter[str] | None = None
        self._header_written = False
        self._should_write_header = True
        self._buffer: list[dict[str, Any]] = []
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(
            maxsize=max(1, int(max_queue))
        )
        self._closed = False
        self._lock = threading.Lock()

        mode = "a" if append else "w"
        self._file = self._path.open(mode, newline="", encoding="utf-8")

        if append and self._path.exists():
            try:
                self._should_write_header = self._path.stat().st_size == 0
            except OSError:
                self._should_write_header = True

        if self._fieldnames is not None:
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            if self._should_write_header:
                self._writer.writeheader()
                self._header_written = True

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def write_row(self, row: Mapping[str, Any]) -> None:
        if self._closed:
            return
        self._queue.put(dict(row))

    def flush(self) -> None:
        if self._closed:
            return
        self._queue.join()
        with self._lock:
            self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_SENTINEL)
        self._thread.join()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                break
            if not isinstance(item, dict):
                self._queue.task_done()
                continue
            self._buffer.append(item)
            if len(self._buffer) >= self._flush_every:
                self._flush_buffer_locked()
            self._queue.task_done()

        if self._buffer:
            self._flush_buffer_locked()
        with self._lock:
            self._file.flush()
            self._file.close()
            self._writer = None

    def _flush_buffer_locked(self) -> None:
        if not self._buffer:
            return
        with self._lock:
            if self._writer is None:
                self._fieldnames = list(self._buffer[0].keys())
                self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
                if self._should_write_header and not self._header_written:
                    self._writer.writeheader()
                    self._header_written = True

            fieldnames = self._fieldnames or []
            columns = {name: [row.get(name) for row in self._buffer] for name in fieldnames}
            converted = {name: _column_to_list(values) for name, values in columns.items()}

            for idx in range(len(self._buffer)):
                row = {name: converted[name][idx] for name in fieldnames}
                self._writer.writerow(row)

            self._buffer.clear()
            self._file.flush()


def _column_to_list(values: list[Any]) -> list[Any]:
    if torch is not None and _all_tensor_scalars(values):
        stacked = torch.stack([val.detach() for val in values], dim=0)
        if stacked.device.type != "cpu":
            stacked = stacked.cpu()
        if hasattr(stacked, "tolist"):
            return stacked.tolist()
    return [_to_scalar(value) for value in values]


def _all_tensor_scalars(values: list[Any]) -> bool:
    if not values:
        return False
    return all(_is_tensor_scalar(value) for value in values)


def _is_tensor_scalar(value: Any) -> bool:
    if torch is None or not isinstance(value, torch.Tensor):
        return False
    try:
        return value.numel() == 1 and value.dim() == 0
    except Exception:
        return False


def _to_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


__all__ = ["AsyncCsvSink"]
