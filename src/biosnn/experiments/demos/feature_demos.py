"""Feature-focused demo builders used by the CLI demo registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseTopology
from biosnn.core.torch_utils import require_torch
from biosnn.experiments.demo_types import DemoModelSpec
from biosnn.io.sinks import CsvSink
from biosnn.learning.rules import ThreeFactorHebbianParams, ThreeFactorHebbianRule
from biosnn.monitors.csv import NeuronCSVMonitor
from biosnn.monitors.metrics.metrics_csv import MetricsCSVMonitor
from biosnn.monitors.metrics.scalar_utils import scalar_to_float
from biosnn.monitors.raster.spike_events_csv import SpikeEventsCSVMonitor
from biosnn.monitors.weights.projection_weights_csv import ProjectionWeightsCSVMonitor
from biosnn.neuromodulators import GlobalScalarField, GlobalScalarParams
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import ModulatorSpec, PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

FeatureDemoName = Literal[
    "propagation_impulse",
    "delay_impulse",
    "learning_gate",
    "dopamine_plasticity",
]
ParallelCompileMode = Literal["auto", "on", "off"]
DashboardMode = Literal["dashboard", "fast"]


@dataclass(slots=True)
class FeatureDemoConfig:
    out_dir: Path
    mode: DashboardMode = "dashboard"
    steps: int = 20
    dt: float = 1e-3
    seed: int | None = 7
    device: str = "cpu"
    max_ring_mib: float | None = 2048.0
    profile: bool = False
    profile_steps: int = 20
    allow_cuda_monitor_sync: bool | None = None
    parallel_compile: ParallelCompileMode = "auto"
    parallel_compile_workers: int | None = None
    parallel_compile_torch_threads: int = 1
    delay_steps: int = 3
    learning_lr: float = 0.1
    da_amount: float = 1.0
    da_step: int = 10
    fused_layout: Literal["auto", "coo", "csr"] = "auto"
    ring_dtype: str | None = None
    ring_strategy: Literal["dense", "event_bucketed"] = "dense"
    store_sparse_by_delay: bool | None = None

    def __post_init__(self) -> None:
        self.device = str(self.device).lower().strip()
        self.mode = str(self.mode).lower().strip()  # type: ignore[assignment]
        self.parallel_compile = str(self.parallel_compile).lower().strip()  # type: ignore[assignment]
        if self.mode not in {"dashboard", "fast"}:
            raise ValueError("mode must be one of: dashboard, fast")
        if self.parallel_compile not in {"auto", "on", "off"}:
            raise ValueError("parallel_compile must be one of: auto, on, off")
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        if self.learning_lr <= 0.0:
            raise ValueError("learning_lr must be > 0")
        if self.da_step < 0:
            raise ValueError("da_step must be >= 0")
        if self.ring_strategy not in {"dense", "event_bucketed"}:
            raise ValueError("ring_strategy must be one of: dense, event_bucketed")


FeatureDemoBuildResult = tuple[DemoModelSpec, FeatureDemoConfig, list[IMonitor]]
FeatureDemoBuilder = Callable[[FeatureDemoConfig], FeatureDemoBuildResult]


def build_propagation_impulse_demo(cfg: FeatureDemoConfig) -> FeatureDemoBuildResult:
    """Input->Relay->Hidden->Out impulse propagation demo."""

    torch = require_torch()
    runtime_cfg = _resolve_runtime_device(cfg)
    dtype = torch.float32
    n = 4

    positions = {
        "Input": _line_positions(n=n, x=0.0, device=runtime_cfg.device, dtype=dtype),
        "Relay": _line_positions(n=n, x=1.0, device=runtime_cfg.device, dtype=dtype),
        "Hidden": _line_positions(n=n, x=2.0, device=runtime_cfg.device, dtype=dtype),
        "Out": _line_positions(n=n, x=3.0, device=runtime_cfg.device, dtype=dtype),
    }
    populations = [
        PopulationSpec(name="Input", model=GLIFModel(), n=n, positions=positions["Input"]),
        PopulationSpec(name="Relay", model=GLIFModel(), n=n, positions=positions["Relay"]),
        PopulationSpec(name="Hidden", model=GLIFModel(), n=n, positions=positions["Hidden"]),
        PopulationSpec(name="Out", model=GLIFModel(), n=n, positions=positions["Out"]),
    ]

    projections = [
        ProjectionSpec(
            name="Input_to_Relay",
            synapse=DelayedCurrentSynapse(_delayed_current_params(cfg=runtime_cfg, init_weight=1e-7)),
            topology=_identity_topology(
                n=n,
                delay_steps=0,
                weight_value=1e-7,
                device=runtime_cfg.device,
                dtype=dtype,
                pre_pos=positions["Input"],
                post_pos=positions["Relay"],
            ),
            pre="Input",
            post="Relay",
        ),
        ProjectionSpec(
            name="Relay_to_Hidden",
            synapse=DelayedCurrentSynapse(_delayed_current_params(cfg=runtime_cfg, init_weight=1e-7)),
            topology=_identity_topology(
                n=n,
                delay_steps=0,
                weight_value=1e-7,
                device=runtime_cfg.device,
                dtype=dtype,
                pre_pos=positions["Relay"],
                post_pos=positions["Hidden"],
            ),
            pre="Relay",
            post="Hidden",
        ),
        ProjectionSpec(
            name="Hidden_to_Out",
            synapse=DelayedCurrentSynapse(_delayed_current_params(cfg=runtime_cfg, init_weight=1e-7)),
            topology=_identity_topology(
                n=n,
                delay_steps=0,
                weight_value=1e-7,
                device=runtime_cfg.device,
                dtype=dtype,
                pre_pos=positions["Hidden"],
                post_pos=positions["Out"],
            ),
            pre="Hidden",
            post="Out",
        ),
    ]

    def external_drive_fn(t: float, step: int, pop_name: str, ctx: StepContext) -> Mapping[Compartment, Any]:
        _ = t
        if pop_name != "Input":
            return {}
        drive = _pulse_range(step, n=n, ctx=ctx, start=5, end=7, value=2e-7)
        if drive is None:
            return {}
        return {Compartment.SOMA: drive}

    model_spec = DemoModelSpec(
        populations=populations,
        projections=projections,
        external_drive_fn=external_drive_fn,
        include_neuron_topology=True,
    )
    monitors = _build_demo_monitors(
        cfg=runtime_cfg,
        populations=populations,
        projections=projections,
        tap_tensor_keys=("v_soma", "theta"),
        expected_arrival_steps={"Hidden_to_Out": 8},
    )
    return model_spec, runtime_cfg, monitors


def build_delay_impulse_demo(cfg: FeatureDemoConfig) -> FeatureDemoBuildResult:
    """1->1 impulse delay alignment demo."""

    torch = require_torch()
    runtime_cfg = _resolve_runtime_device(cfg)
    dtype = torch.float32
    n = 1
    pre_pos = _line_positions(n=n, x=0.0, device=runtime_cfg.device, dtype=dtype)
    post_pos = _line_positions(n=n, x=1.0, device=runtime_cfg.device, dtype=dtype)

    populations = [
        PopulationSpec(name="Input", model=GLIFModel(), n=n, positions=pre_pos),
        PopulationSpec(name="Post", model=GLIFModel(), n=n, positions=post_pos),
    ]
    projection = ProjectionSpec(
        name="Input_to_Post",
        synapse=DelayedCurrentSynapse(_delayed_current_params(cfg=runtime_cfg, init_weight=1e-7)),
        topology=_identity_topology(
            n=n,
            delay_steps=int(runtime_cfg.delay_steps),
            weight_value=1e-7,
            device=runtime_cfg.device,
            dtype=dtype,
            pre_pos=pre_pos,
            post_pos=post_pos,
        ),
        pre="Input",
        post="Post",
    )
    projections = [projection]

    def external_drive_fn(t: float, step: int, pop_name: str, ctx: StepContext) -> Mapping[Compartment, Any]:
        _ = t
        if pop_name != "Input":
            return {}
        drive = _pulse_range(step, n=n, ctx=ctx, start=5, end=5, value=2e-7)
        if drive is None:
            return {}
        return {Compartment.SOMA: drive}

    expected_step = 5 + 1 + int(runtime_cfg.delay_steps)
    model_spec = DemoModelSpec(
        populations=populations,
        projections=projections,
        external_drive_fn=external_drive_fn,
        include_neuron_topology=True,
    )
    monitors = _build_demo_monitors(
        cfg=runtime_cfg,
        populations=populations,
        projections=projections,
        tap_tensor_keys=("v_soma",),
        expected_arrival_steps={"Input_to_Post": expected_step},
    )
    return model_spec, runtime_cfg, monitors


def build_learning_gate_demo(cfg: FeatureDemoConfig) -> FeatureDemoBuildResult:
    """One-edge learning gate demo with deterministic co-spiking schedule."""

    runtime_cfg, populations, projections, external_drive_fn = _build_learning_base(cfg)
    model_spec = DemoModelSpec(
        populations=populations,
        projections=projections,
        external_drive_fn=external_drive_fn,
        include_neuron_topology=True,
    )
    monitors = _build_demo_monitors(
        cfg=runtime_cfg,
        populations=populations,
        projections=projections,
        tap_tensor_keys=("v_soma", f"proj/{projections[0].name}/weights", f"learn/{projections[0].name}/last_mean_dw"),
        expected_arrival_steps={"Pre_to_Post": 6},
    )
    return model_spec, runtime_cfg, monitors


def build_dopamine_plasticity_demo(cfg: FeatureDemoConfig) -> FeatureDemoBuildResult:
    """Learning gate demo with DA off->on phases in a single run."""

    torch = require_torch()
    runtime_cfg, populations, projections, external_drive_fn = _build_learning_base(cfg)
    field = GlobalScalarField(
        kinds=(ModulatorKind.DOPAMINE,),
        params=GlobalScalarParams(decay_tau=8.0),
    )
    modulators = [ModulatorSpec(name="dopamine", field=field, kinds=(ModulatorKind.DOPAMINE,))]

    def releases_fn(t: float, step: int, ctx: StepContext) -> Sequence[ModulatorRelease]:
        _ = t
        if step < int(runtime_cfg.da_step):
            return []
        if not _is_learning_pulse(step):
            return []
        device_obj = torch.device(ctx.device) if ctx.device is not None else torch.device(runtime_cfg.device)
        dtype_obj = _resolve_dtype(torch, ctx.dtype)
        positions = torch.zeros((1, 3), device=device_obj, dtype=dtype_obj)
        amount = torch.tensor([float(runtime_cfg.da_amount)], device=device_obj, dtype=dtype_obj)
        return [
            ModulatorRelease(
                kind=ModulatorKind.DOPAMINE,
                positions=positions,
                amount=amount,
            )
        ]

    model_spec = DemoModelSpec(
        populations=populations,
        projections=projections,
        modulators=modulators,
        external_drive_fn=external_drive_fn,
        releases_fn=releases_fn,
        include_neuron_topology=True,
    )
    monitors = _build_demo_monitors(
        cfg=runtime_cfg,
        populations=populations,
        projections=projections,
        tap_tensor_keys=(
            "v_soma",
            f"proj/{projections[0].name}/weights",
            f"learn/{projections[0].name}/last_mean_dw",
            "mod/dopamine/levels",
        ),
        expected_arrival_steps={"Pre_to_Post": 6},
    )
    return model_spec, runtime_cfg, monitors


FEATURE_DEMO_BUILDERS: dict[FeatureDemoName, FeatureDemoBuilder] = {
    "propagation_impulse": build_propagation_impulse_demo,
    "delay_impulse": build_delay_impulse_demo,
    "learning_gate": build_learning_gate_demo,
    "dopamine_plasticity": build_dopamine_plasticity_demo,
}


class _ProjectionDriveCSVMonitor(IMonitor):
    """Writes projection post-drive summaries to CSV."""

    name = "projection_drive_csv"

    def __init__(
        self,
        path: Path,
        *,
        projection_names: Sequence[str],
        expected_arrival_steps: Mapping[str, int] | None = None,
        every_n_steps: int = 1,
        flush_every: int = 25,
        append: bool = False,
    ) -> None:
        self._engine: TorchNetworkEngine | None = None
        self._projection_names = tuple(projection_names)
        self._expected_arrival_steps = dict(expected_arrival_steps) if expected_arrival_steps else {}
        self._every_n_steps = max(1, int(every_n_steps))
        self._event_count = 0
        self._sink = CsvSink(path, flush_every=max(1, int(flush_every)), append=append)

    def bind_engine(self, engine: TorchNetworkEngine) -> None:
        self._engine = engine

    def compilation_requirements(self) -> Mapping[str, bool]:
        return {
            "wants_projection_drive_tensor": True,
            "wants_weights_snapshot_each_step": True,
        }

    def requirements(self) -> MonitorRequirements:
        return MonitorRequirements(
            needs_projection_drive=True,
            needs_projection_weights=True,
            needs_scalars=True,
        )

    def on_step(self, event: StepEvent) -> None:
        self._event_count += 1
        if self._event_count % self._every_n_steps != 0:
            return
        if self._engine is None:
            return

        step_val = -1
        if event.scalars and "step" in event.scalars:
            step_val = int(scalar_to_float(event.scalars["step"]))
        for proj_name in self._projection_names:
            drive_map = self._engine.last_projection_drive.get(proj_name)
            if drive_map is None:
                continue
            weights = None
            if event.tensors is not None:
                weights = event.tensors.get(f"proj/{proj_name}/weights")
            for comp, drive in drive_map.items():
                expected = self._expected_arrival_steps.get(proj_name)
                row: dict[str, Any] = {
                    "step": step_val,
                    "t": event.t,
                    "proj": proj_name,
                    "compartment": comp.value,
                    "drive_sum": float(drive.sum().item()),
                    "drive_abs_mean": float(drive.abs().mean().item()),
                    "drive_max": float(drive.max().item()),
                    "expected_arrival_step": int(expected) if expected is not None else "",
                    "weight_mean": "",
                    "weight_min": "",
                    "weight_max": "",
                }
                if weights is not None and int(weights.numel()) > 0:
                    row["weight_mean"] = float(weights.mean().item())
                    row["weight_min"] = float(weights.min().item())
                    row["weight_max"] = float(weights.max().item())
                self._sink.write_row(row)

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


def _build_learning_base(
    cfg: FeatureDemoConfig,
) -> tuple[FeatureDemoConfig, list[PopulationSpec], list[ProjectionSpec], Any]:
    torch = require_torch()
    runtime_cfg = _resolve_runtime_device(cfg)
    dtype = torch.float32
    n = 1
    pre_pos = _line_positions(n=n, x=0.0, device=runtime_cfg.device, dtype=dtype)
    post_pos = _line_positions(n=n, x=1.0, device=runtime_cfg.device, dtype=dtype)

    populations = [
        PopulationSpec(name="Pre", model=GLIFModel(), n=n, positions=pre_pos),
        PopulationSpec(name="Post", model=GLIFModel(), n=n, positions=post_pos),
    ]
    projection = ProjectionSpec(
        name="Pre_to_Post",
        synapse=DelayedCurrentSynapse(_delayed_current_params(cfg=runtime_cfg, init_weight=0.0)),
        topology=_identity_topology(
            n=n,
            delay_steps=0,
            weight_value=0.0,
            device=runtime_cfg.device,
            dtype=dtype,
            pre_pos=pre_pos,
            post_pos=post_pos,
        ),
        pre="Pre",
        post="Post",
        learning=ThreeFactorHebbianRule(
            ThreeFactorHebbianParams(
                lr=float(runtime_cfg.learning_lr),
                weight_decay=0.0,
            )
        ),
        sparse_learning=True,
    )
    projections = [projection]

    def external_drive_fn(t: float, step: int, pop_name: str, ctx: StepContext) -> Mapping[Compartment, Any]:
        _ = t
        if pop_name not in {"Pre", "Post"}:
            return {}
        if not _is_learning_pulse(step):
            return {}
        drive = _pulse_range(step, n=n, ctx=ctx, start=step, end=step, value=2e-7)
        if drive is None:
            return {}
        return {Compartment.SOMA: drive}

    return runtime_cfg, populations, projections, external_drive_fn


def _build_demo_monitors(
    *,
    cfg: FeatureDemoConfig,
    populations: Sequence[PopulationSpec],
    projections: Sequence[ProjectionSpec],
    tap_tensor_keys: Sequence[str],
    expected_arrival_steps: Mapping[str, int] | None = None,
) -> list[IMonitor]:
    run_mode = cfg.mode.lower().strip()
    fast_mode = run_mode == "fast"
    cuda_device = cfg.device == "cuda"
    allow_cuda_sync = cfg.allow_cuda_monitor_sync
    if allow_cuda_sync is None:
        allow_cuda_sync = run_mode == "dashboard"
    allow_cuda_sync = bool(allow_cuda_sync)
    monitor_async_gpu = cuda_device and not allow_cuda_sync

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    projection_names = [proj.name for proj in projections]
    sample_n = max(1, min(8, min(pop.n for pop in populations)))
    sample_indices = list(range(sample_n))

    monitors: list[IMonitor] = [
        MetricsCSVMonitor(
            str(out_dir / "metrics.csv"),
            stride=1,
            append=False,
            flush_every=25,
            async_gpu=monitor_async_gpu,
        ),
        ProjectionWeightsCSVMonitor(
            str(out_dir / "weights.csv"),
            projections=projections,
            stride=1,
            max_edges_sample=10_000,
            append=False,
            flush_every=25,
        ),
        _ProjectionDriveCSVMonitor(
            out_dir / "synapse.csv",
            projection_names=projection_names,
            expected_arrival_steps=expected_arrival_steps,
            every_n_steps=1,
            flush_every=25,
        ),
    ]

    if not fast_mode:
        monitors.insert(
            0,
            NeuronCSVMonitor(
                out_dir / "neuron.csv",
                tensor_keys=("v_soma", "theta"),
                include_spikes=True,
                include_scalars=True,
                sample_indices=sample_indices,
                flush_every=25,
                async_gpu=monitor_async_gpu,
            ),
        )
        monitors.insert(
            1,
            NeuronCSVMonitor(
                out_dir / "tap.csv",
                tensor_keys=tap_tensor_keys,
                include_spikes=True,
                include_scalars=True,
                sample_indices=sample_indices,
                flush_every=25,
                async_gpu=monitor_async_gpu,
            ),
        )
        if not cuda_device or allow_cuda_sync:
            monitors.insert(
                2,
                SpikeEventsCSVMonitor(
                    str(out_dir / "spikes.csv"),
                    stride=1,
                    max_spikes_per_step=10_000,
                    safe_neuron_sample=512,
                    allow_cuda_sync=allow_cuda_sync,
                    append=False,
                    flush_every=25,
                ),
            )

    return monitors


def _resolve_runtime_device(cfg: FeatureDemoConfig) -> FeatureDemoConfig:
    torch = require_torch()
    requested = cfg.device
    resolved = requested
    if requested == "cuda" and not torch.cuda.is_available():
        resolved = "cpu"
    if resolved == requested:
        return cfg
    return FeatureDemoConfig(
        out_dir=cfg.out_dir,
        mode=cfg.mode,
        steps=cfg.steps,
        dt=cfg.dt,
        seed=cfg.seed,
        device=resolved,
        max_ring_mib=cfg.max_ring_mib,
        profile=cfg.profile,
        profile_steps=cfg.profile_steps,
        allow_cuda_monitor_sync=cfg.allow_cuda_monitor_sync,
        parallel_compile=cfg.parallel_compile,
        parallel_compile_workers=cfg.parallel_compile_workers,
        parallel_compile_torch_threads=cfg.parallel_compile_torch_threads,
        delay_steps=cfg.delay_steps,
        learning_lr=cfg.learning_lr,
        da_amount=cfg.da_amount,
        da_step=cfg.da_step,
        fused_layout=cfg.fused_layout,
        ring_dtype=cfg.ring_dtype,
        ring_strategy=cfg.ring_strategy,
        store_sparse_by_delay=cfg.store_sparse_by_delay,
    )


def _identity_topology(
    *,
    n: int,
    delay_steps: int,
    weight_value: float,
    device: str,
    dtype: Any,
    pre_pos: Any,
    post_pos: Any,
) -> SynapseTopology:
    torch = require_torch()
    device_obj = torch.device(device)
    pre_idx = torch.arange(n, device=device_obj, dtype=torch.long)
    post_idx = torch.arange(n, device=device_obj, dtype=torch.long)
    delays = torch.full((n,), int(delay_steps), device=device_obj, dtype=torch.int32)
    weights = torch.full((n,), float(weight_value), device=device_obj, dtype=dtype)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delays,
        weights=weights,
        pre_pos=pre_pos,
        post_pos=post_pos,
        target_compartment=Compartment.DENDRITE,
    )


def _line_positions(*, n: int, x: float, device: str, dtype: Any) -> Any:
    torch = require_torch()
    device_obj = torch.device(device)
    pos = torch.zeros((n, 3), device=device_obj, dtype=dtype)
    pos[:, 0] = float(x)
    if n > 1:
        pos[:, 1] = torch.linspace(0.0, 1.0, n, device=device_obj, dtype=dtype)
    return pos


def _pulse_range(
    step: int,
    *,
    n: int,
    ctx: StepContext,
    start: int,
    end: int,
    value: float,
) -> Any | None:
    if step < start or step > end:
        return None
    torch = require_torch()
    device_obj = torch.device(ctx.device) if ctx.device is not None else None
    dtype_obj = _resolve_dtype(torch, ctx.dtype)
    return torch.full((n,), float(value), device=device_obj, dtype=dtype_obj)


def _is_learning_pulse(step: int, *, start: int = 5, period: int = 2) -> bool:
    if step < start:
        return False
    return (step - start) % max(1, int(period)) == 0


def _resolve_dtype(torch: Any, dtype: Any) -> Any:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, str):
        attr = dtype.split(".", 1)[-1]
        return getattr(torch, attr, torch.float32)
    return dtype


def _delayed_current_params(*, cfg: FeatureDemoConfig, init_weight: float) -> DelayedCurrentParams:
    use_event_bucketed = cfg.ring_strategy == "event_bucketed"
    return DelayedCurrentParams(
        init_weight=init_weight,
        ring_dtype=cfg.ring_dtype,
        ring_strategy=_ring_strategy_for_delayed_current(cfg.ring_strategy),
        event_driven=use_event_bucketed,
    )


def _ring_strategy_for_delayed_current(
    strategy: Literal["dense", "event_bucketed"],
) -> Literal["dense", "event_list_proto"]:
    return "event_list_proto" if strategy == "event_bucketed" else "dense"


__all__ = [
    "FEATURE_DEMO_BUILDERS",
    "FeatureDemoBuildResult",
    "FeatureDemoBuilder",
    "FeatureDemoConfig",
    "FeatureDemoName",
    "build_delay_impulse_demo",
    "build_dopamine_plasticity_demo",
    "build_learning_gate_demo",
    "build_propagation_impulse_demo",
]
