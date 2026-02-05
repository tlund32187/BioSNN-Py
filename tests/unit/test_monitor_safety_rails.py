from __future__ import annotations

import pytest

from biosnn.contracts.monitors import StepEvent
from biosnn.contracts.synapses import SynapseTopology
from biosnn.monitors.csv import NeuronCSVMonitor
from biosnn.monitors.raster.spike_events_csv import SpikeEventsCSVMonitor
from biosnn.monitors.weights.projection_weights_csv import ProjectionWeightsCSVMonitor

torch = pytest.importorskip("torch")


def test_neuron_csv_monitor_safe_sample_applies(tmp_path):
    monitor = NeuronCSVMonitor(
        tmp_path / "neuron.csv",
        tensor_keys=("v",),
        include_spikes=False,
        sample_indices=None,
        safe_sample=4,
    )
    event = StepEvent(t=0.0, dt=1e-3, tensors={"v": torch.zeros((10,), dtype=torch.float32)})
    monitor.on_step(event)
    assert monitor._sample_indices == list(range(4))
    monitor.close()


def test_neuron_csv_monitor_safe_sample_skips_small(tmp_path):
    monitor = NeuronCSVMonitor(
        tmp_path / "neuron_small.csv",
        tensor_keys=("v",),
        include_spikes=False,
        sample_indices=None,
        safe_sample=8,
    )
    event = StepEvent(t=0.0, dt=1e-3, tensors={"v": torch.zeros((4,), dtype=torch.float32)})
    monitor.on_step(event)
    assert monitor._sample_indices is None
    monitor.close()


def test_spike_events_safe_neuron_sample(tmp_path):
    monitor = SpikeEventsCSVMonitor(
        str(tmp_path / "spikes.csv"),
        neuron_sample=None,
        safe_neuron_sample=32,
    )
    assert monitor._neuron_sample == 32
    monitor.close()

    explicit = SpikeEventsCSVMonitor(
        str(tmp_path / "spikes_explicit.csv"),
        neuron_sample=8,
        safe_neuron_sample=32,
    )
    assert explicit._neuron_sample == 8
    explicit.close()


def test_projection_weights_safe_edge_sample(tmp_path):
    topology = SynapseTopology(
        pre_idx=torch.tensor([0, 1], dtype=torch.long),
        post_idx=torch.tensor([0, 1], dtype=torch.long),
    )
    monitor = ProjectionWeightsCSVMonitor(
        str(tmp_path / "weights.csv"),
        projections=[{"name": "P", "topology": topology}],
        max_edges_sample=50000,
        safe_max_edges_sample=20000,
    )
    assert monitor._max_edges_sample == 20000
    monitor.close()

    monitor_small = ProjectionWeightsCSVMonitor(
        str(tmp_path / "weights_small.csv"),
        projections=[{"name": "P", "topology": topology}],
        max_edges_sample=100,
        safe_max_edges_sample=20000,
    )
    assert monitor_small._max_edges_sample == 100
    monitor_small.close()
