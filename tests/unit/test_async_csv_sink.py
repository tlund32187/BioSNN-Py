from __future__ import annotations

import csv

from biosnn.io.sinks import AsyncCsvSink


def test_async_csv_sink_writes_rows(tmp_path):
    path = tmp_path / "async.csv"
    sink = AsyncCsvSink(path, fieldnames=["a", "b"], flush_every=2, max_queue=10)
    for idx in range(5):
        sink.write_row({"a": idx, "b": idx + 1})
    sink.close()

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert rows[0]["a"] == "0"
    assert rows[-1]["b"] == "5"


def test_async_csv_sink_close_idempotent(tmp_path):
    path = tmp_path / "async_close.csv"
    sink = AsyncCsvSink(path, fieldnames=["x"], flush_every=1, max_queue=2)
    sink.write_row({"x": 1})
    sink.close()
    sink.close()
