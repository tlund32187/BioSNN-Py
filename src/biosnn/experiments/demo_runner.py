"""Shared demo runner for network-style demos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from biosnn.contracts.monitors import IMonitor
from biosnn.contracts.simulation import SimulationConfig
from biosnn.core.torch_utils import require_torch
from biosnn.io.dashboard_export import export_population_topology_json
from biosnn.simulation.engine import TorchNetworkEngine

from .demo_types import DemoModelSpec, DemoRuntimeConfig


def run_demo_from_spec(
    model_spec: DemoModelSpec,
    runtime_config: DemoRuntimeConfig,
    monitors: Sequence[IMonitor],
) -> dict[str, Any]:
    """Run a demo network from a spec and write dashboard artifacts."""

    fast_mode = runtime_config.mode.lower().strip() == "fast"
    out_dir = Path(runtime_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = TorchNetworkEngine(
        populations=model_spec.populations,
        projections=model_spec.projections,
        modulators=model_spec.modulators,
        homeostasis=model_spec.homeostasis,
        external_drive_fn=model_spec.external_drive_fn,
        releases_fn=model_spec.releases_fn,
        fast_mode=fast_mode,
        compiled_mode=fast_mode,
        learning_use_scratch=fast_mode,
        parallel_compile=runtime_config.parallel_compile,
        parallel_compile_workers=runtime_config.parallel_compile_workers,
        parallel_compile_torch_threads=runtime_config.parallel_compile_torch_threads,
    )

    for monitor in monitors:
        bind_engine = getattr(monitor, "bind_engine", None)
        if callable(bind_engine):
            bind_engine(engine)

    engine.attach_monitors(monitors)
    sim_config = SimulationConfig(
        dt=runtime_config.dt,
        device=runtime_config.device,
        dtype="float32",
        seed=runtime_config.seed,
        max_ring_mib=runtime_config.max_ring_mib,
    )
    engine.reset(config=sim_config)
    engine.run(steps=runtime_config.steps)

    if runtime_config.profile:
        engine.attach_monitors([])
        engine.reset(config=sim_config)
        _run_profile(
            engine=engine,
            steps=runtime_config.profile_steps,
            device=runtime_config.device,
            out_path=out_dir / "profile.json",
        )

    topology_path = out_dir / "topology.json"
    export_population_topology_json(
        model_spec.populations,
        model_spec.projections,
        path=topology_path,
        include_neuron_topology=model_spec.include_neuron_topology,
    )

    return {
        "out_dir": out_dir,
        "topology": topology_path,
        "neuron_csv": out_dir / "neuron.csv",
        "synapse_csv": out_dir / "synapse.csv",
        "spikes_csv": out_dir / "spikes.csv",
        "metrics_csv": out_dir / "metrics.csv",
        "weights_csv": out_dir / "weights.csv",
        "homeostasis_csv": out_dir / "homeostasis.csv",
        "steps": runtime_config.steps,
        "device": runtime_config.device,
    }


def _run_profile(*, engine: TorchNetworkEngine, steps: int, device: str, out_path: Path) -> None:
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


__all__ = ["run_demo_from_spec"]
