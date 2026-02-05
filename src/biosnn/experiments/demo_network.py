"""Reusable multi-population network demo experiment."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import (
    build_bipartite_distance_topology,
    build_bipartite_erdos_renyi_topology,
)
from biosnn.connectivity.delays import DelayParams, compute_delay_steps
from biosnn.contracts.monitors import IMonitor, StepEvent
from biosnn.contracts.neurons import Compartment
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.synapses import ISynapseModel
from biosnn.core.torch_utils import require_torch
from biosnn.io.dashboard_export import export_population_topology_json
from biosnn.io.sinks import CsvSink
from biosnn.monitors.csv import NeuronCSVMonitor
from biosnn.monitors.metrics.metrics_csv import MetricsCSVMonitor
from biosnn.monitors.metrics.scalar_utils import scalar_to_float
from biosnn.monitors.raster.spike_events_csv import SpikeEventsCSVMonitor
from biosnn.monitors.weights.projection_weights_csv import ProjectionWeightsCSVMonitor
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import (
    NeuronPosition,
    PopulationFrame,
    PopulationSpec,
    ProjectionSpec,
)
from biosnn.synapses.dynamics.delayed_current import (
    DelayedCurrentParams,
    DelayedCurrentSynapse,
)
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)


@dataclass(slots=True)
class DemoNetworkConfig:
    out_dir: Path
    mode: Literal["dashboard", "fast"] = "dashboard"
    steps: int = 800
    dt: float = 1e-3
    seed: int | None = None
    device: str = "cuda"
    profile: bool = False
    profile_steps: int = 20
    allow_cuda_monitor_sync: bool | None = None
    parallel_compile: Literal["auto", "on", "off"] = "auto"
    parallel_compile_workers: int | None = None
    parallel_compile_torch_threads: int = 1
    monitor_safe_defaults: bool = True
    monitor_neuron_sample: int = 512
    monitor_edge_sample: int = 20000
    n_in: int = 16
    n_hidden: int = 64
    n_out: int = 10
    input_pops: int = 2
    input_depth: int = 2
    hidden_layers: int = 1
    hidden_pops_per_layer: int = 1
    output_pops: int = 1
    input_cross: bool = False
    input_drive_mode: str = "all"
    p_in_hidden: float = 0.25
    p_hidden_out: float = 0.25
    input_to_relay_p: float = 0.35
    input_to_relay_weight_scale: float = 1.5
    relay_to_hidden_p: float = 0.20
    relay_to_hidden_weight_scale: float = 1.0
    hidden_to_output_p: float = 0.20
    hidden_to_output_weight_scale: float = 1.0
    input_skip_to_hidden: bool = False
    input_skip_p: float = 0.03
    input_skip_weight_scale: float = 0.5
    relay_cross: bool = False
    relay_cross_p: float = 0.05
    relay_cross_weight_scale: float = 0.2
    weight_init: float = 0.05
    feedforward_delay_from_distance: bool = False
    feedforward_delay_base_velocity: float = 1.0
    feedforward_delay_myelin_scale: float = 5.0
    feedforward_delay_myelin_mean: float = 0.6
    feedforward_delay_myelin_std: float = 0.2
    feedforward_delay_distance_scale: float = 1.0
    feedforward_delay_min: float = 0.0
    feedforward_delay_max: float | None = None
    feedforward_delay_use_ceil: bool = True
    input_drive: float = 1.0
    neuron_sample: int = 32
    synapse_sample: int = 64
    spike_stride: int = 2
    spike_cap: int = 5000
    weights_stride: int = 10
    weights_cap: int = 20000
    drive_monitor: bool = False
    relay_lateral: bool = False
    relay_lateral_sigma: float = 0.15
    relay_lateral_p0: float = 0.25
    relay_lateral_dist_p_mode: Literal["off", "gaussian"] = "gaussian"
    relay_lateral_dist_p_sigma: float = 0.20
    relay_lateral_weight_scale: float = 0.2
    relay_lateral_dist_weight_mode: Literal["off", "near_stronger", "near_weaker"] = "near_stronger"
    relay_lateral_dist_weight_alpha: float = 1.0
    relay_lateral_dist_weight_floor: float = 0.25
    relay_lateral_dist_weight_cap: float = 2.0
    relay_lateral_ei_mode: Literal["mostly_inhib", "balanced"] = "mostly_inhib"
    relay_lateral_inhib_frac: float = 0.8
    relay_lateral_delay_from_distance: bool = False
    relay_lateral_delay_per_unit_steps: float = 2.0
    hidden_lateral: bool = False
    hidden_lateral_sigma: float = 0.20
    hidden_lateral_p0: float = 0.05
    hidden_lateral_dist_p_mode: Literal["off", "gaussian"] = "gaussian"
    hidden_lateral_dist_p_sigma: float = 0.20
    hidden_lateral_weight_scale: float = 0.1
    hidden_lateral_dist_weight_mode: Literal["off", "near_stronger", "near_weaker"] = "near_stronger"
    hidden_lateral_dist_weight_alpha: float = 1.0
    hidden_lateral_dist_weight_floor: float = 0.25
    hidden_lateral_dist_weight_cap: float = 2.0
    hidden_lateral_ei_mode: Literal["balanced", "mostly_inhib"] = "balanced"
    hidden_lateral_delay_from_distance: bool = False
    hidden_lateral_delay_per_unit_steps: float = 2.0


def run_demo_network(cfg: DemoNetworkConfig) -> dict[str, Any]:
    """Run a multi-population demo network and write CSV/JSON artifacts."""

    torch = require_torch()
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    run_mode = cfg.mode.lower().strip()
    fast_mode = run_mode == "fast"
    cuda_device = device == "cuda"
    allow_cuda_sync = cfg.allow_cuda_monitor_sync
    if allow_cuda_sync is None:
        allow_cuda_sync = run_mode == "dashboard"
    allow_cuda_sync = bool(allow_cuda_sync)
    monitor_async_gpu = cuda_device and not allow_cuda_sync

    dtype = "float32"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_pops = max(1, int(cfg.input_pops))
    input_depth = 2 if cfg.input_depth == 2 else 1
    hidden_layers = max(0, int(cfg.hidden_layers))
    hidden_pops_per_layer = max(1, int(cfg.hidden_pops_per_layer))
    output_pops = max(1, int(cfg.output_pops))

    input_sizes = _split_counts(cfg.n_in, input_pops)
    hidden_sizes = _split_counts(cfg.n_hidden, hidden_layers * hidden_pops_per_layer) if hidden_layers else []
    output_sizes = _split_counts(cfg.n_out, output_pops)

    populations: list[PopulationSpec] = []
    layers: list[list[PopulationSpec]] = []

    input_layer: list[PopulationSpec] = []
    for idx, size in enumerate(input_sizes):
        pop = _build_population_spec(
            name=f"Input{idx}",
            model=GLIFModel(),
            n=size,
            layer=0,
            role="input",
            group=idx,
            device=device,
            dtype=dtype,
            seed=_seed_for(cfg.seed, 0, idx),
        )
        populations.append(pop)
        input_layer.append(pop)
    layers.append(input_layer)

    if input_depth == 2:
        relay_layer: list[PopulationSpec] = []
        for idx, size in enumerate(input_sizes):
            pop = _build_population_spec(
                name=f"InputRelay{idx}",
                model=GLIFModel(),
                n=size,
                layer=1,
                role="input_relay",
                group=idx,
                device=device,
                dtype=dtype,
                seed=_seed_for(cfg.seed, 1, idx),
            )
            populations.append(pop)
            relay_layer.append(pop)
        layers.append(relay_layer)

    hidden_layer_start = input_depth
    if hidden_layers:
        hidden_idx = 0
        for layer_idx in range(hidden_layers):
            layer_list: list[PopulationSpec] = []
            for pop_idx in range(hidden_pops_per_layer):
                size = hidden_sizes[hidden_idx]
                hidden_idx += 1
                pop = _build_population_spec(
                    name=f"HiddenL{layer_idx}P{pop_idx}",
                    model=AdEx2CompModel(),
                    n=size,
                    layer=hidden_layer_start + layer_idx,
                    role="hidden",
                    group=layer_idx,
                    device=device,
                    dtype=dtype,
                    seed=_seed_for(cfg.seed, hidden_layer_start + layer_idx, pop_idx),
                )
                populations.append(pop)
                layer_list.append(pop)
            layers.append(layer_list)

    output_layer_idx = hidden_layer_start + hidden_layers
    output_layer: list[PopulationSpec] = []
    for idx, size in enumerate(output_sizes):
        pop = _build_population_spec(
            name=f"Output{idx}",
            model=GLIFModel(),
            n=size,
            layer=output_layer_idx,
            role="output",
            group=idx,
            device=device,
            dtype=dtype,
            seed=_seed_for(cfg.seed, output_layer_idx, idx),
        )
        populations.append(pop)
        output_layer.append(pop)
    layers.append(output_layer)

    use_sparse_synapse = device == "cuda"
    make_synapse: Callable[[], ISynapseModel]
    if use_sparse_synapse:
        # CUDA default: use sparse matmul backend to avoid CPU fallback paths.
        syn_params_sparse = DelayedSparseMatmulParams(init_weight=cfg.weight_init)

        def make_synapse() -> ISynapseModel:
            return DelayedSparseMatmulSynapse(syn_params_sparse)

    else:
        syn_params_current = DelayedCurrentParams(init_weight=cfg.weight_init)

        def make_synapse() -> ISynapseModel:
            return DelayedCurrentSynapse(syn_params_current)
    projections: list[ProjectionSpec] = []
    topologies: list[Any] = []

    def add_projection(
        pre: PopulationSpec,
        post: PopulationSpec,
        *,
        p: float,
        weight_scale: float = 1.0,
        apply_distance_delay: bool = False,
    ) -> None:
        weight_init = float(cfg.weight_init) * float(weight_scale)
        topo = build_bipartite_erdos_renyi_topology(
            n_pre=pre.n,
            n_post=post.n,
            p=p,
            device=device,
            dtype=dtype,
            weight_init=weight_init,
            target_compartment=default_target_compartment_for_post(post.model),
            pre_pos=pre.positions,
            post_pos=post.positions,
        )
        if apply_distance_delay and cfg.feedforward_delay_from_distance:
            topo = _apply_feedforward_delays(
                topo,
                dt=cfg.dt,
                base_velocity=cfg.feedforward_delay_base_velocity,
                myelin_scale=cfg.feedforward_delay_myelin_scale,
                myelin_mean=cfg.feedforward_delay_myelin_mean,
                myelin_std=cfg.feedforward_delay_myelin_std,
                distance_scale=cfg.feedforward_delay_distance_scale,
                min_delay=cfg.feedforward_delay_min,
                max_delay=cfg.feedforward_delay_max,
                use_ceil=cfg.feedforward_delay_use_ceil,
                seed=_seed_for(cfg.seed, _stable_hash_text(f"{pre.name}->{post.name}")),
                device=device,
                dtype=dtype,
            )
        proj = ProjectionSpec(
            name=f"{pre.name}->{post.name}",
            synapse=make_synapse(),
            topology=topo,
            pre=pre.name,
            post=post.name,
        )
        projections.append(proj)
        topologies.append(topo)

    if input_depth == 2:
        relay_layer = layers[1]
        for i, pre in enumerate(input_layer):
            for j, post in enumerate(relay_layer):
                if (not cfg.input_cross) and i != j:
                    continue
                add_projection(
                    pre,
                    post,
                    p=cfg.input_to_relay_p,
                    weight_scale=cfg.input_to_relay_weight_scale,
                    apply_distance_delay=True,
                )

        if cfg.relay_cross and len(relay_layer) > 1:
            for i, pre in enumerate(relay_layer):
                for j, post in enumerate(relay_layer):
                    if i == j:
                        continue
                    add_projection(
                        pre,
                        post,
                        p=cfg.relay_cross_p,
                        weight_scale=cfg.relay_cross_weight_scale,
                        apply_distance_delay=False,
                    )

        if hidden_layers:
            first_hidden = layers[hidden_layer_start]
            for pre in relay_layer:
                for post in first_hidden:
                    add_projection(
                        pre,
                        post,
                        p=cfg.relay_to_hidden_p,
                        weight_scale=cfg.relay_to_hidden_weight_scale,
                        apply_distance_delay=True,
                    )
            if cfg.input_skip_to_hidden:
                for pre in input_layer:
                    for post in first_hidden:
                        add_projection(
                            pre,
                            post,
                            p=cfg.input_skip_p,
                            weight_scale=cfg.input_skip_weight_scale,
                            apply_distance_delay=True,
                        )
        else:
            for pre in relay_layer:
                for post in output_layer:
                    add_projection(
                        pre,
                        post,
                        p=cfg.hidden_to_output_p,
                        weight_scale=cfg.hidden_to_output_weight_scale,
                        apply_distance_delay=True,
                    )
    else:
        if hidden_layers:
            first_hidden = layers[hidden_layer_start]
            for pre in input_layer:
                for post in first_hidden:
                    add_projection(
                        pre,
                        post,
                        p=cfg.relay_to_hidden_p,
                        weight_scale=cfg.relay_to_hidden_weight_scale,
                        apply_distance_delay=True,
                    )
        else:
            for pre in input_layer:
                for post in output_layer:
                    add_projection(
                        pre,
                        post,
                        p=cfg.hidden_to_output_p,
                        weight_scale=cfg.hidden_to_output_weight_scale,
                        apply_distance_delay=True,
                    )

    if hidden_layers > 1:
        for layer_offset in range(hidden_layers - 1):
            pre_layer = layers[hidden_layer_start + layer_offset]
            post_layer = layers[hidden_layer_start + layer_offset + 1]
            for pre in pre_layer:
                for post in post_layer:
                    add_projection(
                        pre,
                        post,
                        p=cfg.relay_to_hidden_p,
                        weight_scale=cfg.relay_to_hidden_weight_scale,
                        apply_distance_delay=True,
                    )

    if hidden_layers:
        last_hidden = layers[hidden_layer_start + hidden_layers - 1]
        for pre in last_hidden:
            for post in output_layer:
                add_projection(
                    pre,
                    post,
                    p=cfg.hidden_to_output_p,
                    weight_scale=cfg.hidden_to_output_weight_scale,
                    apply_distance_delay=True,
                )

    if cfg.relay_lateral and input_depth == 2:
        for idx, pop in enumerate(layers[1]):
            positions = pop.positions
            if positions is None:
                raise RuntimeError(f"Population {pop.name} is missing positions for relay_lateral")
            topo = build_bipartite_distance_topology(
                pre_positions=positions,
                post_positions=positions,
                p0=cfg.relay_lateral_p0,
                sigma=cfg.relay_lateral_sigma,
                dist_p_mode=cfg.relay_lateral_dist_p_mode,
                dist_p_sigma=cfg.relay_lateral_dist_p_sigma,
                device=device,
                dtype=dtype,
                seed=_seed_for(cfg.seed, 2000, idx),
                delay_from_distance=cfg.relay_lateral_delay_from_distance,
                delay_base_steps=0,
                delay_per_unit_steps=cfg.relay_lateral_delay_per_unit_steps,
            )
            topo = _remove_self_edges(topo)
            topo = replace(
                topo,
                target_compartment=default_target_compartment_for_post(pop.model),
            )
            topo = _apply_lateral_weights(
                topo,
                weight_init=cfg.weight_init,
                weight_scale=cfg.relay_lateral_weight_scale,
                ei_mode=cfg.relay_lateral_ei_mode,
                inhib_frac=cfg.relay_lateral_inhib_frac,
                dist_weight_mode=cfg.relay_lateral_dist_weight_mode,
                dist_weight_alpha=cfg.relay_lateral_dist_weight_alpha,
                dist_weight_floor=cfg.relay_lateral_dist_weight_floor,
                dist_weight_cap=cfg.relay_lateral_dist_weight_cap,
                dist_weight_sigma=cfg.relay_lateral_sigma,
                seed=_seed_for(cfg.seed, 3000, idx),
                device=device,
                dtype=dtype,
            )
            proj = ProjectionSpec(
                name=f"{pop.name}->{pop.name}",
                synapse=make_synapse(),
                topology=topo,
                pre=pop.name,
                post=pop.name,
            )
            projections.append(proj)
            topologies.append(topo)

    if cfg.hidden_lateral and hidden_layers:
        for layer_offset in range(hidden_layers):
            layer_pops = layers[hidden_layer_start + layer_offset]
            for pop_idx, pop in enumerate(layer_pops):
                positions = pop.positions
                if positions is None:
                    raise RuntimeError(f"Population {pop.name} is missing positions for hidden_lateral")
                topo = build_bipartite_distance_topology(
                    pre_positions=positions,
                    post_positions=positions,
                    p0=cfg.hidden_lateral_p0,
                    sigma=cfg.hidden_lateral_sigma,
                    dist_p_mode=cfg.hidden_lateral_dist_p_mode,
                    dist_p_sigma=cfg.hidden_lateral_dist_p_sigma,
                    device=device,
                    dtype=dtype,
                    seed=_seed_for(cfg.seed, 4000 + layer_offset, pop_idx),
                    delay_from_distance=cfg.hidden_lateral_delay_from_distance,
                    delay_base_steps=0,
                    delay_per_unit_steps=cfg.hidden_lateral_delay_per_unit_steps,
                )
                topo = _remove_self_edges(topo)
                topo = replace(
                    topo,
                    target_compartment=default_target_compartment_for_post(pop.model),
                )
                inhib_frac = 0.5 if cfg.hidden_lateral_ei_mode == "balanced" else 0.8
                topo = _apply_lateral_weights(
                    topo,
                    weight_init=cfg.weight_init,
                    weight_scale=cfg.hidden_lateral_weight_scale,
                    ei_mode=cfg.hidden_lateral_ei_mode,
                    inhib_frac=inhib_frac,
                    dist_weight_mode=cfg.hidden_lateral_dist_weight_mode,
                    dist_weight_alpha=cfg.hidden_lateral_dist_weight_alpha,
                    dist_weight_floor=cfg.hidden_lateral_dist_weight_floor,
                    dist_weight_cap=cfg.hidden_lateral_dist_weight_cap,
                    dist_weight_sigma=cfg.hidden_lateral_sigma,
                    seed=_seed_for(cfg.seed, 5000 + layer_offset, pop_idx),
                    device=device,
                    dtype=dtype,
                )
                proj = ProjectionSpec(
                    name=f"{pop.name}->{pop.name}",
                    synapse=make_synapse(),
                    topology=topo,
                    pre=pop.name,
                    post=pop.name,
                )
                projections.append(proj)
                topologies.append(topo)

    input_pops_list = list(layers[0])
    input_names = {pop.name for pop in input_pops_list}
    input_sizes_by_name = {pop.name: pop.n for pop in input_pops_list}
    input_scale_by_name = {pop.name: 1.0 + 0.1 * idx for idx, pop in enumerate(input_pops_list)}

    def external_drive_fn(t: float, step: int, pop_name: str, ctx) -> dict[Compartment, Any]:
        _ = (t, step)
        if pop_name not in input_names:
            return {}
        device_obj = torch.device(ctx.device) if ctx.device else None
        dtype_obj = getattr(torch, ctx.dtype) if isinstance(ctx.dtype, str) else ctx.dtype
        n_local = input_sizes_by_name.get(pop_name, 0)
        scale = 1.0
        if cfg.input_drive_mode.lower() == "perpop":
            scale = input_scale_by_name.get(pop_name, 1.0)
        drive = torch.full((n_local,), cfg.input_drive * scale, device=device_obj, dtype=dtype_obj)
        return {Compartment.SOMA: drive}

    engine = TorchNetworkEngine(
        populations=populations,
        projections=projections,
        external_drive_fn=external_drive_fn,
        fast_mode=fast_mode,
        compiled_mode=fast_mode,
        learning_use_scratch=fast_mode,
        parallel_compile=cfg.parallel_compile,
        parallel_compile_workers=cfg.parallel_compile_workers,
        parallel_compile_torch_threads=cfg.parallel_compile_torch_threads,
    )

    total_neurons = sum(pop.n for pop in populations)
    neuron_sample = _clamp_sample(cfg.neuron_sample, total_neurons, cap=64)
    edge_counts = [_edge_count(topo) for topo in topologies] or [0]
    synapse_sample = _clamp_sample(cfg.synapse_sample, min(edge_counts), cap=64)
    spike_stride = max(2, cfg.spike_stride)
    spike_cap = min(cfg.spike_cap, 5000)
    weights_stride = max(cfg.weights_stride, 10)
    weights_cap = min(cfg.weights_cap, 20000)
    safe_neuron_sample = cfg.monitor_neuron_sample if cfg.monitor_safe_defaults else None
    safe_edge_sample = cfg.monitor_edge_sample if cfg.monitor_safe_defaults else None

    weight_keys = [f"proj/{proj.name}/weights" for proj in projections]

    neuron_tensor_keys = (
        "v_soma",
        "v_dend",
        "w",
        "refrac_left",
        "spike_hold_left",
        "theta",
    )

    monitors: list[IMonitor]
    if fast_mode:
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
            NeuronCSVMonitor(
                out_dir / "neuron.csv",
                tensor_keys=neuron_tensor_keys,
                include_spikes=True,
                sample_indices=list(range(neuron_sample)) if neuron_sample > 0 else None,
                safe_sample=safe_neuron_sample,
                flush_every=25,
                async_gpu=monitor_async_gpu,
            ),
        ]
        if not cuda_device or allow_cuda_sync:
            monitors.append(
                _AggregatedSynapseCSVMonitor(
                    out_dir / "synapse.csv",
                    weight_keys=weight_keys,
                    sample_n=synapse_sample,
                    stats=("mean", "std"),
                    flush_every=25,
                )
            )
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
        if not cuda_device or allow_cuda_sync:
            monitors.append(
                ProjectionWeightsCSVMonitor(
                    str(out_dir / "weights.csv"),
                    projections=projections,
                    stride=weights_stride,
                    max_edges_sample=weights_cap,
                    safe_max_edges_sample=safe_edge_sample,
                    append=False,
                    flush_every=25,
                )
            )
        if cfg.drive_monitor and (not cuda_device or allow_cuda_sync):
            monitors.append(
                _PopDriveCSVMonitor(
                    out_dir / "drive.csv",
                    engine=engine,
                    pop_names=[pop.name for pop in populations],
                    every_n_steps=1,
                    flush_every=25,
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
    export_population_topology_json(
        populations,
        projections,
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


class _PopDriveCSVMonitor(IMonitor):
    name = "csv_drive"

    def __init__(
        self,
        path: Path,
        *,
        engine: TorchNetworkEngine,
        pop_names: Iterable[str],
        every_n_steps: int = 1,
        flush_every: int = 25,
        append: bool = False,
    ) -> None:
        self._engine = engine
        self._pop_names = list(pop_names)
        self._every_n_steps = max(1, every_n_steps)
        self._event_count = 0
        self._sink = CsvSink(path, flush_every=max(1, flush_every), append=append)

    def on_step(self, event: StepEvent) -> None:
        self._event_count += 1
        if self._event_count % self._every_n_steps != 0:
            return

        row: dict[str, Any] = {"t": event.t, "dt": event.dt}
        if event.scalars:
            step_val = event.scalars.get("step")
            if step_val is not None:
                row["step"] = scalar_to_float(step_val)

        drive_buffers = getattr(self._engine, "_drive_buffers", {})
        for pop in self._pop_names:
            drive = drive_buffers.get(pop)
            if not drive:
                continue
            pop_key = _sanitize_key(pop)
            soma = drive.get(Compartment.SOMA)
            dend = drive.get(Compartment.DENDRITE)
            if soma is not None:
                row[f"{pop_key}_soma_abs_mean"] = scalar_to_float(soma.abs().mean())
            if dend is not None:
                row[f"{pop_key}_dend_abs_mean"] = scalar_to_float(dend.abs().mean())

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


def _split_counts(total: int, parts: int) -> list[int]:
    parts = max(1, int(parts))
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _apply_lateral_weights(
    topology: Any,
    *,
    weight_init: float,
    weight_scale: float,
    ei_mode: Literal["mostly_inhib", "balanced"],
    inhib_frac: float,
    dist_weight_mode: Literal["off", "near_stronger", "near_weaker"] = "off",
    dist_weight_alpha: float = 1.0,
    dist_weight_floor: float = 0.25,
    dist_weight_cap: float = 2.0,
    dist_weight_sigma: float = 0.2,
    seed: int | None,
    device: str,
    dtype: str,
) -> Any:
    torch = require_torch()
    e = int(topology.pre_idx.numel()) if hasattr(topology.pre_idx, "numel") else len(topology.pre_idx)
    if e <= 0:
        return topology
    device_obj = torch.device(device) if device else None
    dtype_obj = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    weights = torch.full((e,), float(weight_init) * float(weight_scale), device=device_obj, dtype=dtype_obj)
    mode = ei_mode.lower()
    frac = 0.5 if mode == "balanced" else float(inhib_frac)
    frac = max(0.0, min(1.0, frac))
    if frac > 0.0:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device_obj) if device_obj is not None else torch.Generator()
            generator.manual_seed(seed)
        mask = torch.rand((e,), device=device_obj, generator=generator) < frac
        selected = mask.nonzero(as_tuple=False).flatten()
        if selected.numel() > 0:
            weights = weights.clone()
            neg_vals = weights.index_select(0, selected) * -1.0
            weights.index_copy_(0, selected, neg_vals)
    dist_mode = dist_weight_mode.lower()
    if dist_mode != "off" and getattr(topology, "edge_dist", None) is not None:
        dist = topology.edge_dist
        if hasattr(dist, "to"):
            dist = dist.to(device=weights.device, dtype=weights.dtype)
        sigma = max(1e-9, float(dist_weight_sigma))
        dn = torch.clamp(dist / sigma, 0.0, 3.0)
        if dist_mode == "near_weaker":
            mult = 1.0 + float(dist_weight_alpha) * dn
        else:
            mult = torch.exp(-float(dist_weight_alpha) * dn)
        mult = torch.clamp(mult, min=float(dist_weight_floor), max=float(dist_weight_cap))
        signs = torch.sign(weights)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        weights = weights.abs() * mult * signs
    return replace(topology, weights=weights)


def _apply_feedforward_delays(
    topology: Any,
    *,
    dt: float,
    base_velocity: float,
    myelin_scale: float,
    myelin_mean: float,
    myelin_std: float,
    distance_scale: float,
    min_delay: float,
    max_delay: float | None,
    use_ceil: bool,
    seed: int | None,
    device: str,
    dtype: str,
) -> Any:
    if dt <= 0:
        return topology
    pre_pos = getattr(topology, "pre_pos", None)
    post_pos = getattr(topology, "post_pos", None)
    if pre_pos is None or post_pos is None:
        return topology
    pre_idx = getattr(topology, "pre_idx", None)
    post_idx = getattr(topology, "post_idx", None)
    if pre_idx is None or post_idx is None:
        return topology
    e = int(pre_idx.numel()) if hasattr(pre_idx, "numel") else len(pre_idx)
    if e <= 0:
        return topology
    myelin = _sample_myelin(
        e,
        mean=myelin_mean,
        std=myelin_std,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    params = DelayParams(
        base_velocity=float(base_velocity),
        myelin_scale=float(myelin_scale),
        distance_scale=float(distance_scale),
        min_delay=float(min_delay),
        max_delay=max_delay,
        use_ceil=bool(use_ceil),
    )
    delay_steps = compute_delay_steps(
        pre_pos=pre_pos,
        post_pos=post_pos,
        pre_idx=pre_idx,
        post_idx=post_idx,
        dt=float(dt),
        myelin=myelin,
        params=params,
    )
    return replace(topology, delay_steps=delay_steps, myelin=myelin)


def _sample_myelin(
    e: int,
    *,
    mean: float,
    std: float,
    seed: int | None,
    device: str,
    dtype: str,
) -> Any:
    torch = require_torch()
    device_obj = torch.device(device) if device else None
    dtype_obj = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if e <= 0:
        return torch.empty((0,), device=device_obj, dtype=dtype_obj)
    if std <= 0:
        values = torch.full((e,), float(mean), device=device_obj, dtype=dtype_obj)
    else:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device_obj) if device_obj is not None else torch.Generator()
            generator.manual_seed(seed)
        values = torch.randn((e,), device=device_obj, dtype=dtype_obj, generator=generator) * float(std)
        values = values + float(mean)
    return torch.clamp(values, 0.0, 1.0)


def _remove_self_edges(topology: Any) -> Any:
    if not hasattr(topology, "pre_idx") or not hasattr(topology, "post_idx"):
        return topology
    pre_idx = topology.pre_idx
    post_idx = topology.post_idx
    if hasattr(pre_idx, "numel") and pre_idx.numel() == 0:
        return topology
    mask = pre_idx != post_idx
    if hasattr(mask, "numel"):
        invalid = (~mask).nonzero(as_tuple=False).flatten()
        if invalid.numel() == 0:
            return topology
    elif hasattr(mask, "all") and bool(mask.all()):
        return topology
    filtered = {
        "pre_idx": pre_idx[mask],
        "post_idx": post_idx[mask],
    }
    if getattr(topology, "delay_steps", None) is not None:
        filtered["delay_steps"] = topology.delay_steps[mask]
    if getattr(topology, "edge_dist", None) is not None:
        filtered["edge_dist"] = topology.edge_dist[mask]
    if getattr(topology, "weights", None) is not None:
        filtered["weights"] = topology.weights[mask]
    if getattr(topology, "target_compartments", None) is not None:
        filtered["target_compartments"] = topology.target_compartments[mask]
    if getattr(topology, "receptor", None) is not None:
        filtered["receptor"] = topology.receptor[mask]
    if getattr(topology, "myelin", None) is not None:
        filtered["myelin"] = topology.myelin[mask]
    return replace(topology, **filtered)


def _build_population_spec(
    *,
    name: str,
    model: Any,
    n: int,
    layer: int,
    role: str,
    group: int,
    device: str,
    dtype: str,
    seed: int | None,
    layout: Literal["grid", "random", "ring", "line"] = "line",
) -> PopulationSpec:
    frame = PopulationFrame(
        origin=(float(layer), 0.0, 0.0),
        extent=(1.0, 1.0, 0.0),
        layout=layout,
        seed=seed,
    )
    positions = _positions_tensor(frame, n, device=device, dtype=dtype)
    return PopulationSpec(
        name=name,
        model=model,
        n=n,
        frame=frame,
        positions=positions,
        meta={"layer": layer, "role": role, "group": group},
    )


def _positions_tensor(frame: PopulationFrame, n: int, *, device: str, dtype: str) -> Any:
    torch = require_torch()
    device_obj = torch.device(device) if device else None
    dtype_obj = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    positions = _generate_positions(frame, n)
    if not positions:
        return torch.empty((0, 3), device=device_obj, dtype=dtype_obj)
    data = [[pos.x, pos.y, pos.z] for pos in positions]
    return torch.tensor(data, device=device_obj, dtype=dtype_obj)


def _generate_positions(frame: PopulationFrame, n: int) -> list[NeuronPosition]:
    if n <= 0:
        return []
    origin_x, origin_y, origin_z = frame.origin
    extent_x, extent_y, extent_z = frame.extent
    layout = frame.layout
    if layout == "line":
        x_val = _center(origin_x, extent_x)
        z_val = _center(origin_z, extent_z)
        ys = _linspace(origin_y, origin_y + extent_y, n)
        return [NeuronPosition(x_val, y, z_val) for y in ys]
    if layout == "grid":
        cols, rows = _grid_dims(n, extent_x, extent_y)
        xs = _linspace(origin_x, origin_x + extent_x, cols)
        ys = _linspace(origin_y, origin_y + extent_y, rows)
        z_val = _center(origin_z, extent_z)
        grid_positions: list[NeuronPosition] = []
        for idx in range(n):
            row = idx // cols
            col = idx % cols
            if row >= rows:
                break
            grid_positions.append(NeuronPosition(xs[col], ys[row], z_val))
        return grid_positions
    if layout == "ring":
        center_x = _center(origin_x, extent_x)
        center_y = _center(origin_y, extent_y)
        z_val = _center(origin_z, extent_z)
        radius = 0.5 * min(abs(extent_x), abs(extent_y))
        if radius == 0.0:
            return [NeuronPosition(center_x, center_y, z_val) for _ in range(n)]
        return [
            NeuronPosition(
                center_x + radius * math.cos(2.0 * math.pi * idx / n),
                center_y + radius * math.sin(2.0 * math.pi * idx / n),
                z_val,
            )
            for idx in range(n)
        ]
    if layout == "random":
        rng = random.Random(frame.seed)
        rand_positions: list[NeuronPosition] = []
        for _ in range(n):
            rand_positions.append(
                NeuronPosition(
                    origin_x + rng.random() * extent_x,
                    origin_y + rng.random() * extent_y,
                    origin_z + rng.random() * extent_z,
                )
            )
        return rand_positions
    raise ValueError(f"Unknown population layout: {layout}")


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [_center(start, end - start)]
    step = (end - start) / (count - 1)
    return [start + step * idx for idx in range(count)]


def _grid_dims(n: int, extent_x: float, extent_y: float) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    aspect = abs(extent_x / extent_y) if extent_y not in (0.0, -0.0) else 1.0
    cols = max(1, int(math.ceil(math.sqrt(n * aspect))))
    rows = max(1, int(math.ceil(n / cols)))
    return cols, rows


def _center(origin: float, extent: float) -> float:
    return origin + extent * 0.5


def _seed_for(base_seed: int | None, *parts: int) -> int | None:
    if base_seed is None:
        return None
    value = int(base_seed)
    for part in parts:
        value = (value * 1_000_003 + int(part)) & 0x7FFFFFFF
    return value


def _stable_hash_text(value: str) -> int:
    h = 2166136261
    for ch in value:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def default_target_compartment_for_post(model: Any) -> Compartment:
    compartments = cast(frozenset[Compartment], getattr(model, "compartments", frozenset()))
    if compartments == frozenset({Compartment.SOMA}):
        return Compartment.SOMA
    if Compartment.DENDRITE in compartments:
        return Compartment.DENDRITE
    return Compartment.SOMA


def _sanitize_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)


__all__ = ["DemoNetworkConfig", "run_demo_network"]
