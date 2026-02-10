"""Shared demo runner for network-style demos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from biosnn.contracts.monitors import IMonitor
from biosnn.contracts.simulation import SimulationConfig
from biosnn.io.dashboard_export import export_population_topology_json
from biosnn.simulation.engine import TorchNetworkEngine

from .demo_types import DemoModelSpec, DemoRuntimeConfig
from .profile_utils import maybe_write_profile_trace


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
        enable_pruning=bool(getattr(runtime_config, "enable_pruning", False)),
        prune_interval_steps=int(getattr(runtime_config, "prune_interval_steps", 250)),
        usage_alpha=float(getattr(runtime_config, "usage_alpha", 0.01)),
        w_min=float(getattr(runtime_config, "w_min", 0.01)),
        usage_min=float(getattr(runtime_config, "usage_min", 0.01)),
        k_min_out=int(getattr(runtime_config, "k_min_out", 0)),
        k_min_in=int(getattr(runtime_config, "k_min_in", 0)),
        max_prune_fraction_per_interval=float(
            getattr(runtime_config, "max_prune_fraction_per_interval", 0.1)
        ),
        pruning_verbose=bool(getattr(runtime_config, "pruning_verbose", False)),
        enable_neurogenesis=bool(getattr(runtime_config, "enable_neurogenesis", False)),
        growth_interval_steps=int(getattr(runtime_config, "growth_interval_steps", 500)),
        add_neurons_per_event=int(getattr(runtime_config, "add_neurons_per_event", 4)),
        newborn_plasticity_multiplier=float(
            getattr(runtime_config, "newborn_plasticity_multiplier", 1.5)
        ),
        newborn_duration_steps=int(getattr(runtime_config, "newborn_duration_steps", 250)),
        max_total_neurons=int(getattr(runtime_config, "max_total_neurons", 20000)),
        neurogenesis_verbose=bool(getattr(runtime_config, "neurogenesis_verbose", False)),
    )
    engine.reset(config=sim_config)
    engine.run(steps=runtime_config.steps)

    if runtime_config.profile:
        engine.attach_monitors([])
        engine.reset(config=sim_config)
        maybe_write_profile_trace(
            enabled=True,
            engine=engine,
            steps=runtime_config.profile_steps,
            device=runtime_config.device,
            out_path=out_dir / "profile.json",
        )

    topology_path = out_dir / "topology.json"
    populations_final = cast(
        Sequence[Any],
        getattr(engine, "_pop_specs", model_spec.populations),
    )
    projections_final = cast(
        Sequence[Any],
        getattr(engine, "_proj_specs", model_spec.projections),
    )
    export_population_topology_json(
        populations_final,
        projections_final,
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
        "edge_count_csv": out_dir / "edge_count.csv",
        "neurogenesis_csv": out_dir / "neurogenesis.csv",
        "steps": runtime_config.steps,
        "device": runtime_config.device,
    }

__all__ = ["run_demo_from_spec"]
