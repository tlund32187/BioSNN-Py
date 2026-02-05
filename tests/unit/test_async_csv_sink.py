from __future__ import annotations

import csv
import threading
import time

import pytest

from biosnn.io.sinks import AsyncCsvSink

pytestmark = pytest.mark.unit

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


def test_async_csv_sink_close_drains_queue(tmp_path):
    path = tmp_path / "async_race.csv"
    sink = AsyncCsvSink(path, fieldnames=["i"], flush_every=3, max_queue=4)
    written = []
    errors = []

    def writer():
        for idx in range(12):
            try:
                sink.write_row({"i": idx})
                written.append(idx)
            except RuntimeError:
                errors.append(idx)
            time.sleep(0.01)

    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.03)
    sink.close()
    thread.join()

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(written)
