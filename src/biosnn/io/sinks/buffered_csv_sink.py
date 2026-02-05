"""Buffered CSV sink for batched GPU-friendly writes."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:  # optional torch
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


class BufferedCsvSink:
    """Write CSV rows in batches, minimizing GPU->CPU transfers."""

    def __init__(
        self,
        path: str | Path,
        *,
        fieldnames: Iterable[str] | None = None,
        flush_every: int = 100,
        append: bool = False,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = max(1, flush_every)
        self._append = append
        mode = "a" if append else "w"
        self._file = self._path.open(mode, newline="", encoding="utf-8")
        self._write_count = 0
        self._header_written = False
        self._fieldnames = list(fieldnames) if fieldnames is not None else None
        self._writer: csv.DictWriter[str] | None = None
        self._should_write_header = True
        self._buffer: list[dict[str, Any]] = []

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

    def write_row(self, row: Mapping[str, Any]) -> None:
        self._buffer.append(dict(row))
        if len(self._buffer) >= self._flush_every:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return

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
            self._write_count += 1

        self._buffer.clear()
        self._file.flush()

    def flush(self) -> None:
        self._flush_buffer()
        self._file.flush()

    def close(self) -> None:
        self._flush_buffer()
        self._file.flush()
        self._file.close()
        self._writer = None


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


__all__ = ["BufferedCsvSink"]
