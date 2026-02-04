"""Reusable multi-population network demo experiment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import build_bipartite_erdos_renyi_topology
from biosnn.contracts.monitors import IMonitor, StepEvent
from biosnn.contracts.neurons import Compartment
from biosnn.contracts.simulation import SimulationConfig
from biosnn.core.torch_utils import require_torch
from biosnn.io.dashboard_export import export_population_topology_json
from biosnn.io.sinks import CsvSink
from biosnn.monitors.csv import NeuronCSVMonitor
from biosnn.monitors.metrics.metrics_csv import MetricsCSVMonitor
from biosnn.monitors.metrics.scalar_utils import scalar_to_float
from biosnn.monitors.raster.spike_events_csv import SpikeEventsCSVMonitor
from biosnn.monitors.weights.projection_weights_csv import ProjectionWeightsCSVMonitor
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import (
    DelayedCurrentParams,
    DelayedCurrentSynapse,
)


@dataclass(slots=True)
class DemoNetworkConfig:
    out_dir: Path
    steps: int = 800
    dt: float = 1e-3
    seed: int | None = None
    device: str = "cuda"
    n_in: int = 16
    n_hidden: int = 64
    n_out: int = 10
    p_in_hidden: float = 0.25
    p_hidden_out: float = 0.25
    weight_init: float = 0.05
    input_drive: float = 1.0
    neuron_sample: int = 32
    synapse_sample: int = 64
    spike_stride: int = 2
    spike_cap: int = 5000
    weights_stride: int = 10
    weights_cap: int = 20000


def run_demo_network(cfg: DemoNetworkConfig) -> dict[str, Any]:
    """Run a multi-population demo network and write CSV/JSON artifacts."""

    torch = require_torch()
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dtype = "float32"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pop_input = PopulationSpec(
        name="Input",
        model=GLIFModel(),
        n=cfg.n_in,
        meta={"layer": 0},
    )
    pop_hidden = PopulationSpec(
        name="Hidden",
        model=AdEx2CompModel(),
        n=cfg.n_hidden,
        meta={"layer": 1},
    )
    pop_output = PopulationSpec(
        name="Output",
        model=GLIFModel(),
        n=cfg.n_out,
        meta={"layer": 2},
    )

    syn_params = DelayedCurrentParams(init_weight=cfg.weight_init)
    topo_in_hidden = build_bipartite_erdos_renyi_topology(
        n_pre=cfg.n_in,
        n_post=cfg.n_hidden,
        p=cfg.p_in_hidden,
        device=device,
        dtype=dtype,
        weight_init=cfg.weight_init,
    )
    topo_hidden_out = build_bipartite_erdos_renyi_topology(
        n_pre=cfg.n_hidden,
        n_post=cfg.n_out,
        p=cfg.p_hidden_out,
        device=device,
        dtype=dtype,
        weight_init=cfg.weight_init,
    )

    proj_in_hidden = ProjectionSpec(
        name="Input->Hidden",
        synapse=DelayedCurrentSynapse(syn_params),
        topology=topo_in_hidden,
        pre="Input",
        post="Hidden",
    )
    proj_hidden_out = ProjectionSpec(
        name="Hidden->Output",
        synapse=DelayedCurrentSynapse(syn_params),
        topology=topo_hidden_out,
        pre="Hidden",
        post="Output",
    )

    def external_drive_fn(t: float, step: int, pop_name: str, ctx) -> dict[Compartment, Any]:
        _ = (t, step)
        if pop_name != "Input":
            return {}
        device_obj = torch.device(ctx.device) if ctx.device else None
        dtype_obj = getattr(torch, ctx.dtype) if isinstance(ctx.dtype, str) else ctx.dtype
        drive = torch.full((cfg.n_in,), cfg.input_drive, device=device_obj, dtype=dtype_obj)
        return {Compartment.SOMA: drive}

    engine = TorchNetworkEngine(
        populations=[pop_input, pop_hidden, pop_output],
        projections=[proj_in_hidden, proj_hidden_out],
        external_drive_fn=external_drive_fn,
    )

    total_neurons = cfg.n_in + cfg.n_hidden + cfg.n_out
    neuron_sample = _clamp_sample(cfg.neuron_sample, total_neurons, cap=64)
    synapse_sample = _clamp_sample(
        cfg.synapse_sample,
        min(_edge_count(topo_in_hidden), _edge_count(topo_hidden_out)),
        cap=64,
    )
    spike_stride = max(2, cfg.spike_stride)
    spike_cap = min(cfg.spike_cap, 5000)
    weights_stride = max(cfg.weights_stride, 10)
    weights_cap = min(cfg.weights_cap, 20000)

    weight_keys = [
        f"proj/{proj_in_hidden.name}/weights",
        f"proj/{proj_hidden_out.name}/weights",
    ]

    neuron_tensor_keys = (
        "v_soma",
        "v_dend",
        "w",
        "refrac_left",
        "spike_hold_left",
        "theta",
    )

    monitors: list[IMonitor] = [
        NeuronCSVMonitor(
            out_dir / "neuron.csv",
            tensor_keys=neuron_tensor_keys,
            include_spikes=True,
            sample_indices=list(range(neuron_sample)) if neuron_sample > 0 else None,
            flush_every=25,
        ),
        _AggregatedSynapseCSVMonitor(
            out_dir / "synapse.csv",
            weight_keys=weight_keys,
            sample_n=synapse_sample,
            stats=("mean", "std"),
            flush_every=25,
        ),
        SpikeEventsCSVMonitor(
            str(out_dir / "spikes.csv"),
            stride=spike_stride,
            max_spikes_per_step=spike_cap,
            append=False,
            flush_every=25,
        ),
        MetricsCSVMonitor(
            str(out_dir / "metrics.csv"),
            stride=1,
            append=False,
            flush_every=25,
        ),
        ProjectionWeightsCSVMonitor(
            str(out_dir / "weights.csv"),
            projections=[proj_in_hidden, proj_hidden_out],
            stride=weights_stride,
            max_edges_sample=weights_cap,
            append=False,
            flush_every=25,
        ),
    ]

    engine.attach_monitors(monitors)
    engine.reset(
        config=SimulationConfig(
            dt=cfg.dt,
            device=device,
            dtype=dtype,
            seed=cfg.seed,
        )
    )
    engine.run(steps=cfg.steps)

    topology_path = out_dir / "topology.json"
    export_population_topology_json(
        [pop_input, pop_hidden, pop_output],
        [proj_in_hidden, proj_hidden_out],
        path=topology_path,
        include_neuron_topology=True,
    )

    return {
        "out_dir": out_dir,
        "topology": topology_path,
        "neuron_csv": out_dir / "neuron.csv",
        "synapse_csv": out_dir / "synapse.csv",
        "spikes_csv": out_dir / "spikes.csv",
        "metrics_csv": out_dir / "metrics.csv",
        "weights_csv": out_dir / "weights.csv",
        "steps": cfg.steps,
        "device": device,
    }


class _AggregatedSynapseCSVMonitor(IMonitor):
    name = "csv_synapse_aggregate"

    def __init__(
        self,
        path: Path,
        *,
        weight_keys: Iterable[str],
        sample_n: int,
        stats: Iterable[str],
        flush_every: int = 25,
        every_n_steps: int = 1,
        append: bool = False,
    ) -> None:
        self._weight_keys = list(weight_keys)
        self._sample_n = max(0, sample_n)
        self._stats = tuple(stats)
        self._flush_every = max(1, flush_every)
        self._every_n_steps = max(1, every_n_steps)
        self._event_count = 0
        self._sink = CsvSink(path, flush_every=self._flush_every, append=append)

    def on_step(self, event: StepEvent) -> None:
        self._event_count += 1
        if self._event_count % self._every_n_steps != 0:
            return

        row: dict[str, Any] = {"t": event.t, "dt": event.dt}
        if event.scalars:
            for key, value in sorted(event.scalars.items()):
                row[key] = scalar_to_float(value)

        weights_list = _collect_weight_tensors(event, self._weight_keys)
        if weights_list:
            row.update(_aggregate_weight_stats(weights_list, self._stats))
            if self._sample_n > 0:
                for idx, value in _sample_weight_values(weights_list, self._sample_n):
                    row[f"weights_i{idx}"] = value

        self._sink.write_row(row)

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


def _collect_weight_tensors(event: StepEvent, keys: Iterable[str]) -> list[Any]:
    if not event.tensors:
        return []
    return [event.tensors[key] for key in keys if key in event.tensors]


def _aggregate_weight_stats(weights_list: list[Any], stats: Iterable[str]) -> dict[str, float]:
    torch = require_torch()
    stats_set = set(stats)

    total = 0
    sum_vals = None
    sum_sq = None
    min_val = None
    max_val = None

    for weights in weights_list:
        if hasattr(weights, "detach"):
            weights = weights.detach()
        count = int(weights.numel()) if hasattr(weights, "numel") else len(weights)
        if count == 0:
            continue
        total += count
        if "mean" in stats_set or "std" in stats_set:
            part_sum = weights.sum()
            sum_vals = part_sum if sum_vals is None else sum_vals + part_sum
        if "std" in stats_set:
            part_sq = (weights * weights).sum()
            sum_sq = part_sq if sum_sq is None else sum_sq + part_sq
        if "min" in stats_set:
            part_min = weights.min()
            min_val = part_min if min_val is None else torch.minimum(min_val, part_min)
        if "max" in stats_set:
            part_max = weights.max()
            max_val = part_max if max_val is None else torch.maximum(max_val, part_max)

    if total == 0:
        return {f"weights_{stat}": 0.0 for stat in stats_set}

    row: dict[str, float] = {}
    if "mean" in stats_set and sum_vals is not None:
        mean = sum_vals / float(total)
        row["weights_mean"] = float(mean.item())
    if "std" in stats_set and sum_vals is not None and sum_sq is not None:
        mean_val = sum_vals / float(total)
        var = sum_sq / float(total) - mean_val * mean_val
        var = torch.clamp(var, min=0.0)
        row["weights_std"] = float(torch.sqrt(var).item())
    if "min" in stats_set and min_val is not None:
        row["weights_min"] = float(min_val.item())
    if "max" in stats_set and max_val is not None:
        row["weights_max"] = float(max_val.item())

    return row


def _sample_weight_values(weights_list: list[Any], sample_n: int) -> list[tuple[int, float]]:
    torch = require_torch()
    remaining = sample_n
    offset = 0
    samples: list[tuple[int, float]] = []
    for weights in weights_list:
        if remaining <= 0:
            break
        count = int(weights.numel()) if hasattr(weights, "numel") else len(weights)
        if count == 0:
            continue
        take = min(remaining, count)
        idx = torch.arange(take, device=weights.device)
        values = weights.index_select(0, idx).detach().cpu().tolist()
        samples.extend((offset + i, float(value)) for i, value in enumerate(values))
        offset += count
        remaining -= take
    return samples


def _clamp_sample(requested: int, total: int, *, cap: int) -> int:
    if requested <= 0 or total <= 0:
        return 0
    return min(requested, total, cap)


def _edge_count(topology: Any) -> int:
    pre_idx = getattr(topology, "pre_idx", None)
    if pre_idx is None:
        return 0
    if hasattr(pre_idx, "numel"):
        return int(pre_idx.numel())
    try:
        return len(pre_idx)
    except TypeError:
        return 0


__all__ = ["DemoNetworkConfig", "run_demo_network"]
