"""CSV sink utilities for writing rows to disk."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class CsvSink:
    """Write CSV rows with a stable header and optional flushing."""

    def __init__(
        self,
        path: str | Path,
        *,
        fieldnames: Iterable[str] | None = None,
        flush_every: int = 1,
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
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            if self._should_write_header and not self._header_written:
                self._writer.writeheader()
                self._header_written = True

        self._writer.writerow(row)
        self._write_count += 1
        if self._write_count % self._flush_every == 0:
            self._file.flush()

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()
        self._writer = None


__all__ = ["CsvSink"]
