from __future__ import annotations

import csv
import math

import pytest

from biosnn.experiments.demo_network import DemoNetworkConfig, run_demo_network
from biosnn.learning.homeostasis import RateEmaThresholdHomeostasisConfig

pytestmark = pytest.mark.acceptance


def test_demo_network_homeostasis_stability_500_steps(tmp_path) -> None:
    pytest.importorskip("torch")
    out_dir = tmp_path / "network_homeostasis"
    cfg = DemoNetworkConfig(
        out_dir=out_dir,
        mode="fast",
        steps=500,
        dt=1e-3,
        seed=123,
        device="cpu",
        n_in=12,
        n_hidden=48,
        n_out=6,
        input_to_relay_p=0.45,
        relay_to_hidden_p=0.35,
        hidden_to_output_p=0.35,
        input_drive=1.5,
        enable_homeostasis=True,
        homeostasis=RateEmaThresholdHomeostasisConfig(
            alpha=0.02,
            eta=2e-3,
            r_target=0.08,
            clamp_min=0.0,
            clamp_max=0.05,
            scope="per_neuron",
        ),
        homeostasis_export_every=10,
    )

    summary = run_demo_network(cfg)
    assert summary["out_dir"] == out_dir
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "homeostasis.csv").exists()

    spike_fractions = _read_float_column(out_dir / "metrics.csv", "spike_fraction_total")
    assert spike_fractions
    assert all(value < 0.99 for value in spike_fractions)
    assert any(value > 1e-4 for value in spike_fractions)

    homeostasis_rows = _read_csv_rows(out_dir / "homeostasis.csv")
    assert homeostasis_rows
    rate_values = [float(row["rate_mean"]) for row in homeostasis_rows if row["rate_mean"]]
    assert rate_values
    assert all(0.0 <= value <= 1.0 for value in rate_values)

    control_values = [
        float(row["control_mean"])
        for row in homeostasis_rows
        if row["control_mean"] and row["control_mean"].lower() != "nan"
    ]
    assert any(math.isfinite(value) for value in control_values)


def _read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_float_column(path, key):
    values = []
    for row in _read_csv_rows(path):
        raw = row.get(key, "")
        if raw in {"", None}:  # noqa: PLC1901
            continue
        values.append(float(raw))
    return values

