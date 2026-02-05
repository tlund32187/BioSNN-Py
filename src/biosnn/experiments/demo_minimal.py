"""Reusable minimal demo experiment for single-population simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import build_erdos_renyi_topology
from biosnn.contracts.monitors import IMonitor
from biosnn.contracts.neurons import INeuronModel
from biosnn.contracts.simulation import SimulationConfig
from biosnn.core.torch_utils import require_torch
from biosnn.io.dashboard_export import export_topology_json
from biosnn.monitors.csv import (
    AdEx2CompCSVMonitor,
    GLIFCSVMonitor,
    NeuronCSVMonitor,
    SynapseCSVMonitor,
)
from biosnn.monitors.metrics.metrics_csv import MetricsCSVMonitor
from biosnn.monitors.raster.spike_events_csv import SpikeEventsCSVMonitor
from biosnn.simulation.engine import TorchSimulationEngine
from biosnn.synapses.dynamics.delayed_current import (
    DelayedCurrentParams,
    DelayedCurrentSynapse,
)


@dataclass(slots=True)
class DemoMinimalConfig:
    out_dir: Path
    mode: str = "dashboard"
    n_neurons: int = 100
    p_connect: float = 0.05
    steps: int = 500
    dt: float = 1e-3
    seed: int | None = None
    device: str = "cuda"
    profile: bool = False
    profile_steps: int = 20
    allow_cuda_monitor_sync: bool = False
    monitor_safe_defaults: bool = True
    monitor_neuron_sample: int = 512
    monitor_edge_sample: int = 20000
    neuron_model: str = "glif"
    synapse_model: str = "delayed_current"
    neuron_sample: int = 32
    synapse_sample: int = 64
    spike_stride: int = 2
    spike_cap: int = 5000
    weights_stride: int = 10
    weights_cap: int = 20000


def run_demo_minimal(cfg: DemoMinimalConfig) -> dict[str, Any]:
    """Run a minimal single-population demo and write CSV/JSON artifacts."""

    torch = require_torch()
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    cuda_device = device == "cuda"
    allow_cuda_sync = bool(cfg.allow_cuda_monitor_sync)
    monitor_async_gpu = cuda_device and not allow_cuda_sync

    dtype = "float32"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    neuron_model: INeuronModel
    neuron_monitor_cls: type[NeuronCSVMonitor]
    neuron_model_name = cfg.neuron_model.lower().strip()
    if neuron_model_name == "glif":
        neuron_model = GLIFModel()
        neuron_monitor_cls = GLIFCSVMonitor
    elif neuron_model_name in {"adex_2c", "adex2c"}:
        neuron_model = AdEx2CompModel()
        neuron_monitor_cls = AdEx2CompCSVMonitor
    else:
        raise ValueError(f"Unsupported neuron model: {cfg.neuron_model}")

    if cfg.synapse_model.lower().strip() != "delayed_current":
        raise ValueError(f"Unsupported synapse model: {cfg.synapse_model}")

    syn_params = DelayedCurrentParams(init_weight=1e-9)
    synapse_model = DelayedCurrentSynapse(syn_params)

    topology = build_erdos_renyi_topology(
        n=cfg.n_neurons,
        p=cfg.p_connect,
        allow_self=False,
        device=device,
        dtype=dtype,
        dt=cfg.dt,
        weight_init=syn_params.init_weight,
    )

    engine = TorchSimulationEngine(
        neuron_model=neuron_model,
        synapse_model=synapse_model,
        topology=topology,
        n=cfg.n_neurons,
    )

    neuron_sample = _clamp_sample(cfg.neuron_sample, cfg.n_neurons, cap=64)
    synapse_sample = _clamp_sample(cfg.synapse_sample, _edge_count(topology), cap=64)
    spike_stride = max(2, cfg.spike_stride)
    spike_cap = min(cfg.spike_cap, 5000)
    safe_neuron_sample = cfg.monitor_neuron_sample if cfg.monitor_safe_defaults else None

    run_mode = cfg.mode.lower().strip()
    monitors: list[IMonitor]
    if run_mode == "fast":
        monitors = [
            MetricsCSVMonitor(
                str(out_dir / "metrics.csv"),
                stride=10,
                append=False,
                flush_every=25,
                async_gpu=monitor_async_gpu,
            ),
        ]
    else:
        monitors = [
            neuron_monitor_cls(
                out_dir / "neuron.csv",
                include_spikes=True,
                sample_indices=list(range(neuron_sample)) if neuron_sample > 0 else None,
                safe_sample=safe_neuron_sample,
                flush_every=25,
                async_gpu=monitor_async_gpu,
            ),
            SynapseCSVMonitor(
                out_dir / "synapse.csv",
                sample_indices=list(range(synapse_sample)) if synapse_sample > 0 else None,
                stats=("mean", "std"),
                flush_every=25,
                async_gpu=monitor_async_gpu,
            ),
        ]
        if not cuda_device or allow_cuda_sync:
            monitors.append(
                SpikeEventsCSVMonitor(
                    str(out_dir / "spikes.csv"),
                    stride=spike_stride,
                    max_spikes_per_step=spike_cap,
                    safe_neuron_sample=safe_neuron_sample,
                    allow_cuda_sync=allow_cuda_sync,
                    append=False,
                    flush_every=25,
                )
            )
        monitors.append(
            MetricsCSVMonitor(
                str(out_dir / "metrics.csv"),
                stride=1,
                append=False,
                flush_every=25,
                async_gpu=monitor_async_gpu,
            )
        )

    engine.attach_monitors(monitors)
    sim_config = SimulationConfig(
        dt=cfg.dt,
        device=device,
        dtype=dtype,
        seed=cfg.seed,
    )
    engine.reset(config=sim_config)
    engine.run(steps=cfg.steps)

    if cfg.profile:
        engine.attach_monitors([])
        engine.reset(config=sim_config)
        _run_profile(
            engine=engine,
            steps=cfg.profile_steps,
            device=device,
            out_path=out_dir / "profile.json",
        )

    topology_path = out_dir / "topology.json"
    export_topology_json(topology, path=topology_path)

    return {
        "out_dir": out_dir,
        "topology": topology_path,
        "neuron_csv": out_dir / "neuron.csv",
        "synapse_csv": out_dir / "synapse.csv",
        "spikes_csv": out_dir / "spikes.csv",
        "metrics_csv": out_dir / "metrics.csv",
        "steps": cfg.steps,
        "device": device,
    }


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


def _run_profile(*, engine: TorchSimulationEngine, steps: int, device: str, out_path: Path) -> None:
    torch = require_torch()
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        print("Profiler unavailable; skipping profile run.")
        return

    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities) as prof:
        for _ in range(max(1, steps)):
            engine.step()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_path))


__all__ = ["DemoMinimalConfig", "run_demo_minimal"]
