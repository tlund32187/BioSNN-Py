"""Torch-backed multi-population simulation engine."""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, SupportsInt, cast

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.homeostasis import HomeostasisPopulation, IHomeostasisRule
from biosnn.contracts.learning import LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.monitors import IMonitor, MonitorRequirements, Scalar, StepEvent
from biosnn.contracts.neurons import Compartment, INeuronModel, NeuronInputs, StepContext
from biosnn.contracts.simulation import ISimulationEngine, SimulationConfig
from biosnn.contracts.synapses import (
    ISynapseModel,
    ISynapseModelInplace,
    SynapseInputs,
    SynapseTopology,
)
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype
from biosnn.simulation.engine.subsystems import (
    BufferSubsystem,
    CompiledNetworkPlan,
    EngineContext,
    LearningScratch,
    LearningSubsystem,
    ModulatorSubsystem,
    MonitorSubsystem,
    NetworkRequirements,
    StepEventPayloadPlan,
    StepEventSubsystem,
    TopologySubsystem,
)
from biosnn.simulation.network.specs import ModulatorSpec, PopulationSpec, ProjectionSpec

ExternalDriveFn = Callable[[float, int, str, StepContext], Mapping[Compartment, Tensor]]
ReleasesFn = Callable[[float, int, StepContext], Sequence[ModulatorRelease]]

_LearningScratch = LearningScratch


@dataclass(slots=True)
class _ProjectionState:
    state: Any
    learning_state: Any | None


@dataclass(slots=True)
class _PopulationState:
    state: Any
    spikes: Tensor


@dataclass(slots=True)
class _CompiledProjectionPlan:
    name: str
    pre_name: str
    post_name: str
    synapse: ISynapseModel
    topology: SynapseTopology
    learning_enabled: bool
    needs_bucket_mapping: bool
    use_fused_sparse: bool
    fast_mode: bool
    compiled_mode: bool


@dataclass(slots=True)
class _ProjectionRuntime:
    plan: _CompiledProjectionPlan
    state: _ProjectionState
    learning: Any | None
    learn_every: int
    use_sparse_learning: bool
    has_step_into: bool
    spec: ProjectionSpec
    pre_state: _PopulationState
    post_state: _PopulationState
    drive_target: dict[Compartment, Tensor]

    @property
    def name(self) -> str:
        return self.plan.name


@dataclass(frozen=True, slots=True)
class _ProjectionCompileFlags:
    build_edges_by_delay: bool
    build_pre_adjacency: bool
    build_sparse_delay_mats: bool
    build_bucket_edge_mapping: bool
    fuse_delay_buckets: bool
    store_sparse_by_delay: bool
    build_fused_csr: bool


class TorchNetworkEngine(ISimulationEngine):
    """Multi-population engine with projections, modulators, and learning."""

    name = "torch_network"

    def __init__(
        self,
        *,
        populations: Sequence[PopulationSpec],
        projections: Sequence[ProjectionSpec],
        modulators: Sequence[ModulatorSpec] | None = None,
        homeostasis: IHomeostasisRule | None = None,
        external_drive_fn: ExternalDriveFn | None = None,
        releases_fn: ReleasesFn | None = None,
        fast_mode: bool = False,
        compiled_mode: bool = False,
        learning_use_scratch: bool = False,
        parallel_compile: str = "auto",
        parallel_compile_workers: int | None = None,
        parallel_compile_torch_threads: int = 1,
    ) -> None:
        self._pop_specs = list(populations)
        self._proj_specs = list(projections)
        self._mod_specs = list(modulators) if modulators is not None else []
        self._homeostasis = homeostasis
        self._external_drive_fn = external_drive_fn
        self._releases_fn = releases_fn
        self._fast_mode = bool(fast_mode)
        self._compiled_mode = bool(compiled_mode)
        self._learning_use_scratch = bool(learning_use_scratch)
        self._parallel_compile = parallel_compile.lower().strip()
        if self._parallel_compile not in {"auto", "on", "off"}:
            raise ValueError(f"Invalid parallel_compile mode: {parallel_compile}")
        self._parallel_compile_workers = parallel_compile_workers
        self._parallel_compile_torch_threads = int(parallel_compile_torch_threads)

        self._ctx = StepContext()
        self._engine_context = EngineContext(device=None, dtype=None, dt=0.0, seed=None, rng=None)
        self._dt = 0.0
        self._t = 0.0
        self._step = 0
        self._device = None
        self._dtype = None

        self._pop_states: dict[str, _PopulationState] = {}
        self._proj_states: dict[str, _ProjectionState] = {}
        self._mod_states: dict[str, Any] = {}
        self._monitors: list[IMonitor] = []
        self._monitor_requirements = MonitorRequirements.none()
        self._network_requirements = NetworkRequirements.none()
        self._event_payload_plan: StepEventPayloadPlan | None = None
        self._drive_buffers: dict[str, dict[Compartment, Tensor]] = {}
        self._drive_global: dict[Compartment, Tensor] = {}
        self._drive_global_buffers: list[Tensor] = []
        self._pop_slices: dict[str, slice] = {}
        self._pop_slice_tuples: dict[str, tuple[int, int]] = {}
        self._spikes_global: Tensor | None = None
        self._pop_spike_views: dict[str, Tensor] = {}
        self._compiled_event_tensors: dict[str, Tensor] | None = None
        self._compiled_scalars: dict[str, Any] | None = None
        self._compiled_meta: dict[str, Any] | None = None
        self._compiled_pop_tensor_views: dict[str, dict[str, Tensor]] = {}
        self._compiled_pop_tensor_keys: dict[str, tuple[str, ...]] = {}
        self._compiled_mod_by_pop: dict[str, dict[ModulatorKind, Tensor]] = {}
        self._compiled_edge_mods_by_proj: dict[str, dict[ModulatorKind, Tensor]] = {}
        self._homeostasis_scalars: dict[str, Tensor] = {}
        self._last_event: StepEvent | None = None

        self.last_projection_drive: dict[str, Mapping[Compartment, Tensor]] = {}
        self.last_d_weights: dict[str, Tensor] = {}
        self._learning_scratch: dict[str, _LearningScratch] = {}
        self._proj_runtime_list: list[_ProjectionRuntime] = []
        self._proj_plan_list: list[_CompiledProjectionPlan] = []
        self._proj_plans: dict[str, _CompiledProjectionPlan] = {}
        self._compiled_network_plan: CompiledNetworkPlan | None = None

        self._topology_subsystem = TopologySubsystem()
        self._buffer_subsystem = BufferSubsystem()
        self._modulator_subsystem = ModulatorSubsystem()
        self._learning_subsystem = LearningSubsystem()
        self._monitor_subsystem = MonitorSubsystem()
        self._event_subsystem = StepEventSubsystem()
        self._event_payload_plan = self._event_subsystem.payload_plan(
            self._network_requirements,
            fast_mode=self._fast_mode,
        )

        self._pop_order = [spec.name for spec in self._pop_specs]
        self._pop_map = {spec.name: spec for spec in self._pop_specs}
        self._modulator_kinds = self._modulator_subsystem.collect_modulator_kinds(self._mod_specs)

        self._topology_subsystem.validate_specs(self._pop_specs, self._proj_specs)

    def reset(self, *, config: SimulationConfig) -> None:
        torch = require_torch()

        self._dt = float(config.dt)
        self._t = 0.0
        self._step = 0
        self._ctx = StepContext(
            device=config.device,
            dtype=config.dtype,
            seed=config.seed,
            is_training=True,
            extras=config.meta or None,
        )
        self._device, self._dtype = resolve_device_dtype(self._ctx)
        self._engine_context = EngineContext(
            device=self._device,
            dtype=self._dtype,
            dt=float(config.dt),
            seed=config.seed,
            rng=None,
        )

        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        self._pop_states.clear()
        for spec in self._pop_specs:
            state = spec.model.init_state(spec.n, ctx=self._ctx)
            spikes = torch.zeros((spec.n,), device=self._device, dtype=torch.bool)
            self._pop_states[spec.name] = _PopulationState(state=state, spikes=spikes)

        self._homeostasis_scalars = {}
        if self._homeostasis is not None:
            homeo_pops = [
                HomeostasisPopulation(
                    name=spec.name,
                    model=spec.model,
                    state=self._pop_states[spec.name].state,
                    n=spec.n,
                )
                for spec in self._pop_specs
            ]
            self._homeostasis.init(
                homeo_pops,
                device=self._device,
                dtype=self._dtype,
                ctx=self._ctx,
            )

        self._proj_states.clear()
        self._monitor_requirements = self._monitor_subsystem.collect_requirements(self._monitors)
        monitor_compile_requirements = self._monitor_subsystem.collect_compilation_requirements(
            self._monitors
        )
        self._network_requirements = self._monitor_subsystem.build_network_requirements(
            self._monitors,
            monitor_requirements=self._monitor_requirements,
            compilation_requirements=monitor_compile_requirements,
        )
        self._event_payload_plan = self._event_subsystem.payload_plan(
            self._network_requirements,
            fast_mode=self._fast_mode,
        )
        pop_map = {spec.name: spec for spec in self._pop_specs}
        compile_jobs: list[
            tuple[int, ProjectionSpec, SynapseTopology, dict[str, Any], bool]
        ] = []

        for idx, proj in enumerate(self._proj_specs):
            edge_count = self._topology_subsystem.edge_count(proj.topology)
            syn_state = proj.synapse.init_state(edge_count, ctx=self._ctx)
            learn_state = None
            if proj.learning is not None:
                learn_state = proj.learning.init_state(edge_count, ctx=self._ctx)
            self._proj_states[proj.name] = _ProjectionState(state=syn_state, learning_state=learn_state)

            topology = proj.topology
            if topology.weights is None and getattr(syn_state, "bind_weights_to_topology", False):
                topology = replace(topology, weights=syn_state.weights)
            self._topology_subsystem.copy_topology_weights(syn_state, topology.weights)

            topology = self._topology_subsystem.ensure_topology_meta(
                topology,
                n_pre=pop_map[proj.pre].n,
                n_post=pop_map[proj.post].n,
            )

            projection_requirements = self._topology_subsystem.projection_network_requirements(
                proj,
                base_requirements=self._network_requirements,
            )
            projection_requirements = self._learning_subsystem.projection_network_requirements(
                proj,
                base_requirements=projection_requirements,
            )
            compile_flags = self._topology_subsystem.compile_flags_for_projection(
                proj,
                requirements=projection_requirements,
                device=self._device,
            )
            requires_ring = compile_flags.build_edges_by_delay or compile_flags.build_sparse_delay_mats
            if compile_flags.build_sparse_delay_mats and hasattr(proj.synapse, "params"):
                params = getattr(proj.synapse, "params", None)
                receptor_scale = getattr(params, "receptor_scale", None) if params is not None else None
                if receptor_scale is not None:
                    meta = dict(topology.meta) if topology.meta else {}
                    meta.setdefault("receptor_scale", receptor_scale)
                    topology = replace(topology, meta=meta)

            compile_kwargs: dict[str, Any] = {
                "device": self._device,
                "dtype": self._dtype,
                "build_edges_by_delay": compile_flags.build_edges_by_delay,
                "build_pre_adjacency": compile_flags.build_pre_adjacency,
                "build_sparse_delay_mats": compile_flags.build_sparse_delay_mats,
                "build_bucket_edge_mapping": compile_flags.build_bucket_edge_mapping,
                "fuse_delay_buckets": compile_flags.fuse_delay_buckets,
                "store_sparse_by_delay": compile_flags.store_sparse_by_delay,
                "build_fused_csr": compile_flags.build_fused_csr,
            }
            compile_jobs.append((idx, proj, topology, compile_kwargs, requires_ring))

        self._proj_specs = self._topology_subsystem.compile_projection_jobs(
            projection_specs=self._proj_specs,
            jobs=compile_jobs,
            parallel_mode=self._parallel_compile,
            parallel_workers=self._parallel_compile_workers,
            parallel_torch_threads=self._parallel_compile_torch_threads,
            device=self._device,
            max_ring_mib=config.max_ring_mib,
            executor_factory=ThreadPoolExecutor,
        )

        self._mod_states.clear()
        for mod in self._mod_specs:
            self._mod_states[mod.name] = mod.field.init_state(ctx=self._ctx)

        pop_compartments = self._buffer_subsystem.collect_drive_compartments(
            self._pop_specs,
            self._proj_specs,
        )
        if self._compiled_mode:
            self._init_compiled_buffers(pop_compartments)
        else:
            self._drive_buffers = {
                spec.name: {
                    comp: torch.zeros((spec.n,), device=self._device, dtype=self._dtype)
                    for comp in pop_compartments.get(spec.name, spec.model.compartments)
                }
                for spec in self._pop_specs
            }
            self._drive_global = {}
            self._drive_global_buffers = []
            self._spikes_global = None
            self._pop_spike_views = {}
            self._compiled_event_tensors = None
            self._compiled_scalars = None
            self._compiled_meta = None
            self._compiled_pop_tensor_views = {}
            self._compiled_pop_tensor_keys = {}
            self._compiled_mod_by_pop = {}
            self._compiled_edge_mods_by_proj = {}

        _apply_initial_spikes(self._pop_states, self._pop_order, config.meta)
        self._learning_scratch.clear()
        self._proj_plan_list = self._build_projection_plans()
        self._proj_runtime_list = self._build_proj_runtime()
        self._compiled_network_plan = self._topology_subsystem.build_compiled_network_plan(
            compiled_projections=list(self._proj_specs),
            fast_mode=self._fast_mode,
            compiled_mode=self._compiled_mode,
        )

    def _build_projection_plans(self) -> list[_CompiledProjectionPlan]:
        plans: list[_CompiledProjectionPlan] = []
        for proj in self._proj_specs:
            external_plan = self._topology_subsystem.build_projection_plan(
                proj,
                fast_mode=self._fast_mode,
                compiled_mode=self._compiled_mode,
            )
            plans.append(
                _CompiledProjectionPlan(
                    name=external_plan.name,
                    pre_name=external_plan.pre_name,
                    post_name=external_plan.post_name,
                    synapse=external_plan.synapse,
                    topology=external_plan.topology,
                    learning_enabled=external_plan.learning_enabled,
                    needs_bucket_mapping=external_plan.needs_bucket_mapping,
                    use_fused_sparse=external_plan.use_fused_sparse,
                    fast_mode=external_plan.fast_mode,
                    compiled_mode=external_plan.compiled_mode,
                )
            )
        self._proj_plans = {plan.name: plan for plan in plans}
        return plans

    def _build_proj_runtime(self) -> list[_ProjectionRuntime]:
        runtimes: list[_ProjectionRuntime] = []
        spec_by_name = {spec.name: spec for spec in self._proj_specs}
        for plan in self._proj_plan_list:
            proj = spec_by_name[plan.name]
            proj_state = self._proj_states[plan.name]
            learning = proj.learning
            supports_sparse = bool(getattr(learning, "supports_sparse", False)) if learning is not None else False
            use_sparse = bool(proj.sparse_learning and supports_sparse)
            runtimes.append(
                _ProjectionRuntime(
                    plan=plan,
                    state=proj_state,
                    learning=learning,
                    learn_every=proj.learn_every,
                    use_sparse_learning=use_sparse,
                    has_step_into=isinstance(plan.synapse, ISynapseModelInplace),
                    spec=proj,
                    pre_state=self._pop_states[plan.pre_name],
                    post_state=self._pop_states[plan.post_name],
                    drive_target=self._drive_buffers[plan.post_name],
                )
            )
        return runtimes

    def attach_monitors(self, monitors: Sequence[IMonitor]) -> None:
        monitor_list = list(monitors)
        if self._fast_mode:
            self._monitor_subsystem.validate_fast_mode_monitors(monitor_list)
        self._monitors = monitor_list
        self._monitor_requirements = self._monitor_subsystem.collect_requirements(monitor_list)
        compile_requirements = self._monitor_subsystem.collect_compilation_requirements(monitor_list)
        self._network_requirements = self._monitor_subsystem.build_network_requirements(
            monitor_list,
            monitor_requirements=self._monitor_requirements,
            compilation_requirements=compile_requirements,
        )
        self._event_payload_plan = self._event_subsystem.payload_plan(
            self._network_requirements,
            fast_mode=self._fast_mode,
        )
        if self._compiled_mode and self._pop_states and self._device is not None:
            pop_compartments = self._buffer_subsystem.collect_drive_compartments(
                self._pop_specs,
                self._proj_specs,
            )
            prior_spikes = {
                name: self._pop_states[name].spikes.clone() for name in self._pop_order
            }
            self._init_compiled_buffers(pop_compartments)
            for name, previous in prior_spikes.items():
                current = self._pop_states[name].spikes
                if current.shape == previous.shape:
                    current.copy_(previous)
            self._proj_runtime_list = self._build_proj_runtime()

    def step(self) -> Mapping[str, Any]:
        torch = require_torch()
        if self._compiled_mode:
            return self._step_compiled()

        no_monitors = not self._monitor_subsystem.has_active(self._monitors)
        event_plan = self._event_payload_plan or self._event_subsystem.payload_plan(
            self._network_requirements,
            fast_mode=self._fast_mode,
        )
        needs_population_tensors = event_plan.needs_population_tensors
        needs_population_slices = event_plan.needs_population_slices
        needs_event_spikes = event_plan.needs_event_spikes
        needs_event_scalars = event_plan.needs_scalars
        needs_projection_weights = event_plan.needs_projection_weights
        needs_synapse_state = event_plan.needs_synapse_state
        needs_learning_state = event_plan.needs_learning_state
        needs_homeostasis_state = event_plan.needs_homeostasis_state
        needs_modulator_state = event_plan.needs_modulator_state
        mod_by_pop, edge_mods_by_proj = self._modulator_subsystem.step(
            mod_specs=self._mod_specs,
            mod_states=self._mod_states,
            pop_specs=self._pop_specs,
            proj_specs=self._proj_specs,
            t=self._t,
            step=self._step,
            dt=self._dt,
            ctx=self._ctx,
            device=self._device,
            dtype=self._dtype,
            kinds=self._modulator_kinds,
            releases=self._resolve_releases(),
        )

        mod_by_pop = mod_by_pop or {}
        edge_mods_by_proj = edge_mods_by_proj or {}

        drive_acc = self._drive_buffers
        for drive_by_comp in drive_acc.values():
            for buffer in drive_by_comp.values():
                buffer.zero_()

        self.last_projection_drive = {}
        for runtime in self._proj_runtime_list:
            pre_spikes = runtime.pre_state.spikes
            if runtime.has_step_into:
                synapse_inplace = cast(ISynapseModelInplace, runtime.plan.synapse)
                synapse_inplace.step_into(
                    runtime.state.state,
                    pre_spikes,
                    runtime.drive_target,
                    t=self._step,
                    topology=runtime.plan.topology,
                    dt=self._dt,
                    ctx=self._ctx,
                )
                self.last_projection_drive[runtime.plan.name] = runtime.drive_target
            else:
                runtime.state.state, syn_result = runtime.plan.synapse.step(
                    runtime.state.state,
                    runtime.plan.topology,
                    SynapseInputs(pre_spikes=pre_spikes),
                    dt=self._dt,
                    t=self._t,
                    ctx=self._ctx,
                )
                self.last_projection_drive[runtime.plan.name] = syn_result.post_drive
                self._event_subsystem.accumulate_drive(
                    runtime.drive_target,
                    syn_result.post_drive,
                )

        if self._external_drive_fn is not None:
            for spec in self._pop_specs:
                extra = self._external_drive_fn(self._t, self._step, spec.name, self._ctx)
                self._event_subsystem.accumulate_drive(drive_acc[spec.name], extra)

        spikes_concat: list[Tensor] = []
        neuron_tensors_by_pop: dict[str, Mapping[str, Tensor]] = {}
        pop_slices: dict[str, tuple[int, int]] = {}
        offset = 0

        for spec in self._pop_specs:
            pop_state = self._pop_states[spec.name]
            drive = drive_acc[spec.name]
            neuron_inputs = NeuronInputs(
                drive=drive,
                modulators=mod_by_pop.get(spec.name) if mod_by_pop else None,
            )
            pop_state.state, result = spec.model.step(
                pop_state.state,
                neuron_inputs,
                dt=self._dt,
                t=self._t,
                ctx=self._ctx,
            )
            spikes = result.spikes if result.spikes.dtype == torch.bool else (result.spikes > 0)
            if spikes.device != pop_state.spikes.device:
                raise RuntimeError(
                    "Neuron model returned spikes on a different device. "
                    "Ensure model state is initialized/reset on the engine device."
                )
            if pop_state.spikes.shape == spikes.shape:
                pop_state.spikes.copy_(spikes)
                spikes = pop_state.spikes
            else:
                pop_state.spikes = spikes
            self._pop_states[spec.name] = pop_state

            if not no_monitors:
                if needs_population_slices:
                    start = offset
                    end = offset + spec.n
                    pop_slices[spec.name] = (start, end)
                    offset = end
                if needs_event_spikes:
                    spikes_concat.append(spikes)
                if needs_population_tensors:
                    tensors = spec.model.state_tensors(pop_state.state)
                    if event_plan.needs_population_state:
                        neuron_tensors_by_pop[spec.name] = tensors
                    elif event_plan.needs_v_soma:
                        v_soma = tensors.get("v_soma")
                        if v_soma is not None:
                            neuron_tensors_by_pop[spec.name] = {"v_soma": v_soma}

        if self._homeostasis is not None:
            self._homeostasis_scalars = dict(
                self._homeostasis.step(
                    {spec.name: self._pop_states[spec.name].spikes for spec in self._pop_specs},
                    dt=self._dt,
                    ctx=self._ctx,
                )
            )
        else:
            self._homeostasis_scalars = {}

        self.last_d_weights = {}
        for runtime in self._proj_runtime_list:
            proj = runtime.spec
            if runtime.learning is None:
                continue
            if runtime.learn_every <= 0 or (self._step % runtime.learn_every) != 0:
                continue
            learn_state = runtime.state.learning_state
            if learn_state is None:
                continue

            weights = self._learning_subsystem.require_weights(runtime.state.state, proj.name)
            batch, active_edges = self._learning_subsystem.build_learning_batch(
                proj=proj,
                pre_spikes=runtime.pre_state.spikes,
                post_spikes=runtime.post_state.spikes,
                weights=weights,
                edge_mods=edge_mods_by_proj.get(runtime.plan.name),
                device=self._device,
                use_sparse=runtime.use_sparse_learning,
                scratch=self._get_learning_scratch(proj.name) if self._learning_use_scratch else None,
            )
            new_state, res = runtime.learning.step(
                learn_state,
                batch,
                dt=self._dt,
                t=self._t,
                ctx=self._ctx,
            )
            runtime.state.learning_state = new_state
            self._learning_subsystem.apply_learning_update(
                weights=weights,
                result=res,
                active_edges=active_edges,
                proj_name=proj.name,
                synapse=runtime.plan.synapse,
                topology=runtime.plan.topology,
            )
            self._learning_subsystem.apply_weight_clamp(weights, proj)
            self.last_d_weights[proj.name] = res.d_weights

        if no_monitors:
            event = StepEvent(t=self._t, dt=self._dt)
            self._last_event = event
            t_now = self._t
            step_now = self._step
            self._t += self._dt
            self._step += 1
            return {"step": float(step_now), "t": float(t_now)}

        event_tensors: dict[str, Tensor] = {}
        if self._fast_mode:
            spikes_global = None
            spike_count, spike_fraction = self._event_subsystem.summarize_spikes(
                self._pop_specs,
                self._pop_states,
            )
            if needs_population_tensors:
                event_tensors.update(
                    self._event_subsystem.build_population_tensors(
                        neuron_tensors_by_pop,
                        self._pop_specs,
                        self._pop_states,
                    )
                )
        else:
            if needs_event_spikes:
                spikes_global = (
                    torch.cat(spikes_concat)
                    if spikes_concat
                    else torch.zeros((0,), device=self._device, dtype=torch.bool)
                )
                if spikes_global.numel():
                    spike_count = spikes_global.sum()
                    spike_fraction = spike_count / float(spikes_global.numel())
                else:
                    spike_count = spikes_global.new_zeros(())
                    spike_fraction = spikes_global.new_zeros(())
            else:
                spikes_global = None
                spike_count, spike_fraction = self._event_subsystem.summarize_spikes(
                    self._pop_specs,
                    self._pop_states,
                )
            if needs_population_tensors:
                event_tensors.update(
                    _merge_population_tensors(
                        neuron_tensors_by_pop,
                        self._pop_specs,
                        self._device,
                    )
                )
        if needs_synapse_state:
            event_tensors.update(self._event_subsystem.projection_tensors(self._proj_specs, self._proj_states))
        elif needs_projection_weights:
            event_tensors.update(
                self._event_subsystem.projection_weight_tensors(self._proj_specs, self._proj_states)
            )
        if needs_learning_state:
            event_tensors.update(self._event_subsystem.learning_tensors(self._proj_specs, self._proj_states))
        if needs_homeostasis_state:
            event_tensors.update(self._event_subsystem.homeostasis_tensors(self._homeostasis))
        if needs_modulator_state:
            event_tensors.update(self._event_subsystem.modulator_tensors(self._mod_specs, self._mod_states))

        scalars: dict[str, Scalar] = {
            "step": float(self._step),
            "t": float(self._t),
            "spike_count_total": spike_count,
            "spike_fraction_total": spike_fraction,
        }
        for spec in self._pop_specs:
            scalars[f"spike_count/{spec.name}"] = self._pop_states[spec.name].spikes.sum()
        if self._homeostasis_scalars:
            scalars.update(self._homeostasis_scalars)

        event = StepEvent(
            t=self._t,
            dt=self._dt,
            spikes=spikes_global if needs_event_spikes else None,
            tensors=event_tensors or None,
            scalars=cast(Mapping[str, Scalar], scalars) if needs_event_scalars else None,
            meta={"population_slices": pop_slices} if needs_population_slices else None,
        )
        self._last_event = event

        self._monitor_subsystem.on_step(self._monitors, event)

        self._t += self._dt
        self._step += 1

        return scalars

    def _step_compiled(self) -> Mapping[str, Any]:
        if self._spikes_global is None:
            raise RuntimeError("Engine must be reset before stepping.")

        torch = require_torch()
        no_monitors = not self._monitor_subsystem.has_active(self._monitors)
        event_plan = self._event_payload_plan or self._event_subsystem.payload_plan(
            self._network_requirements,
            fast_mode=self._fast_mode,
        )
        needs_population_tensors = event_plan.needs_population_tensors
        needs_event_spikes = event_plan.needs_event_spikes
        needs_event_scalars = event_plan.needs_scalars
        needs_population_slices = event_plan.needs_population_slices
        build_event_payload = not no_monitors
        if self._mod_specs:
            mod_by_pop, edge_mods_by_proj = self._modulator_subsystem.step_compiled(
                mod_specs=self._mod_specs,
                mod_states=self._mod_states,
                pop_specs=self._pop_specs,
                proj_specs=self._proj_specs,
                t=self._t,
                step=self._step,
                dt=self._dt,
                ctx=self._ctx,
                device=self._device,
                dtype=self._dtype,
                kinds=self._modulator_kinds,
                pop_map=self._pop_map,
                mod_by_pop=self._compiled_mod_by_pop,
                edge_mods_by_proj=self._compiled_edge_mods_by_proj,
                releases=self._resolve_releases(),
            )
        else:
            mod_by_pop = self._compiled_mod_by_pop
            edge_mods_by_proj = self._compiled_edge_mods_by_proj

        if self._drive_global_buffers:
            for buffer in self._drive_global_buffers:
                buffer.zero_()
        else:
            for drive_by_comp in self._drive_buffers.values():
                for buffer in drive_by_comp.values():
                    buffer.zero_()

        self.last_projection_drive = {}
        for runtime in self._proj_runtime_list:
            pre_spikes = runtime.pre_state.spikes
            if runtime.has_step_into:
                synapse_inplace = cast(ISynapseModelInplace, runtime.plan.synapse)
                synapse_inplace.step_into(
                    runtime.state.state,
                    pre_spikes,
                    runtime.drive_target,
                    t=self._step,
                    topology=runtime.plan.topology,
                    dt=self._dt,
                    ctx=self._ctx,
                )
                self.last_projection_drive[runtime.plan.name] = runtime.drive_target
            else:
                runtime.state.state, syn_result = runtime.plan.synapse.step(
                    runtime.state.state,
                    runtime.plan.topology,
                    SynapseInputs(pre_spikes=pre_spikes),
                    dt=self._dt,
                    t=self._t,
                    ctx=self._ctx,
                )
                self.last_projection_drive[runtime.plan.name] = syn_result.post_drive
                self._event_subsystem.accumulate_drive(runtime.drive_target, syn_result.post_drive)

        if self._external_drive_fn is not None:
            for spec in self._pop_specs:
                extra = self._external_drive_fn(self._t, self._step, spec.name, self._ctx)
                self._event_subsystem.accumulate_drive(self._drive_buffers[spec.name], extra)

        for spec in self._pop_specs:
            pop_state = self._pop_states[spec.name]
            drive = self._drive_buffers[spec.name]
            neuron_inputs = NeuronInputs(
                drive=drive,
                modulators=mod_by_pop.get(spec.name) if mod_by_pop else None,
            )
            pop_state.state, result = spec.model.step(
                pop_state.state,
                neuron_inputs,
                dt=self._dt,
                t=self._t,
                ctx=self._ctx,
            )
            spikes_view = pop_state.spikes
            spikes = result.spikes if result.spikes.dtype == torch.bool else (result.spikes > 0)
            if spikes.device != spikes_view.device:
                raise RuntimeError(
                    "Neuron model returned spikes on a different device. "
                    "Ensure model state is initialized/reset on the engine device."
                )
            spikes_view.copy_(spikes)
            self._pop_states[spec.name] = pop_state

            if build_event_payload and needs_population_tensors:
                if self._fast_mode:
                    if self._compiled_event_tensors is not None:
                        if event_plan.needs_spike_tensors:
                            self._compiled_event_tensors[f"pop/{spec.name}/spikes"] = spikes_view
                        tensors = spec.model.state_tensors(pop_state.state)
                        for key in self._compiled_pop_tensor_keys.get(spec.name, ()):
                            if key == "spikes":
                                continue
                            value = tensors.get(key)
                            if value is None:
                                continue
                            self._compiled_event_tensors[f"pop/{spec.name}/{key}"] = value
                else:
                    views = self._compiled_pop_tensor_views.get(spec.name)
                    if views:
                        tensors = spec.model.state_tensors(pop_state.state)
                        for key, view in views.items():
                            value = tensors.get(key)
                            if value is None:
                                continue
                            if value.device != view.device or value.dtype != view.dtype:
                                value = value.to(device=view.device, dtype=view.dtype)
                            view.copy_(value)

        if self._homeostasis is not None:
            self._homeostasis_scalars = dict(
                self._homeostasis.step(
                    {spec.name: self._pop_states[spec.name].spikes for spec in self._pop_specs},
                    dt=self._dt,
                    ctx=self._ctx,
                )
            )
        else:
            self._homeostasis_scalars = {}

        self.last_d_weights = {}
        for runtime in self._proj_runtime_list:
            proj = runtime.spec
            if runtime.learning is None:
                continue
            if runtime.learn_every <= 0 or (self._step % runtime.learn_every) != 0:
                continue
            learn_state = runtime.state.learning_state
            if learn_state is None:
                continue

            weights = self._learning_subsystem.require_weights(runtime.state.state, proj.name)
            batch, active_edges = self._learning_subsystem.build_learning_batch(
                proj=proj,
                pre_spikes=runtime.pre_state.spikes,
                post_spikes=runtime.post_state.spikes,
                weights=weights,
                edge_mods=edge_mods_by_proj.get(runtime.plan.name),
                device=self._device,
                use_sparse=runtime.use_sparse_learning,
                scratch=self._get_learning_scratch(proj.name) if self._learning_use_scratch else None,
            )
            new_state, res = runtime.learning.step(
                learn_state,
                batch,
                dt=self._dt,
                t=self._t,
                ctx=self._ctx,
            )
            runtime.state.learning_state = new_state
            self._learning_subsystem.apply_learning_update(
                weights=weights,
                result=res,
                active_edges=active_edges,
                proj_name=proj.name,
                synapse=runtime.plan.synapse,
                topology=runtime.plan.topology,
            )
            self._learning_subsystem.apply_weight_clamp(weights, proj)
            self.last_d_weights[proj.name] = res.d_weights

        if no_monitors:
            event = StepEvent(t=self._t, dt=self._dt)
            self._last_event = event
            t_now = self._t
            step_now = self._step
            self._t += self._dt
            self._step += 1
            return {"step": float(step_now), "t": float(t_now)}

        spikes_global = self._spikes_global
        spike_count = spikes_global.sum() if spikes_global.numel() else spikes_global.new_zeros(())
        spike_fraction = (
            spike_count / float(spikes_global.numel()) if spikes_global.numel() else spikes_global.new_zeros(())
        )

        scalars = self._compiled_scalars or {}
        scalars["step"] = float(self._step)
        scalars["t"] = float(self._t)
        scalars["spike_count_total"] = spike_count
        scalars["spike_fraction_total"] = spike_fraction
        for spec in self._pop_specs:
            scalars[f"spike_count/{spec.name}"] = self._pop_states[spec.name].spikes.sum()
        if self._homeostasis_scalars:
            scalars.update(self._homeostasis_scalars)

        event = StepEvent(
            t=self._t,
            dt=self._dt,
            spikes=spikes_global if needs_event_spikes else None,
            tensors=self._compiled_event_tensors or None,
            scalars=cast(Mapping[str, Scalar], scalars) if needs_event_scalars else None,
            meta=self._compiled_meta if needs_population_slices else None,
        )
        self._last_event = event

        self._monitor_subsystem.on_step(self._monitors, event)

        self._t += self._dt
        self._step += 1

        return scalars

    def run(self, steps: int) -> None:
        try:
            for _ in range(steps):
                self.step()
        finally:
            self._monitor_subsystem.flush_and_close(self._monitors)

    def _resolve_releases(self) -> Sequence[ModulatorRelease]:
        if self._releases_fn is not None:
            return self._releases_fn(self._t, self._step, self._ctx)
        if self._ctx.extras and "releases" in self._ctx.extras:
            releases = self._ctx.extras.get("releases")
            if isinstance(releases, list):
                return releases
        return []

    def _init_compiled_buffers(self, pop_compartments: Mapping[str, Sequence[Compartment]]) -> None:
        torch = require_torch()
        total = 0
        self._pop_slices = {}
        self._pop_slice_tuples = {}
        for spec in self._pop_specs:
            start = total
            end = total + spec.n
            self._pop_slices[spec.name] = slice(start, end)
            self._pop_slice_tuples[spec.name] = (start, end)
            total = end

        self._spikes_global = torch.zeros((total,), device=self._device, dtype=torch.bool)
        self._pop_spike_views = {
            name: self._spikes_global[self._pop_slices[name]] for name in self._pop_order
        }
        for name in self._pop_order:
            pop_state = self._pop_states[name]
            pop_state.spikes = self._pop_spike_views[name]
            self._pop_states[name] = pop_state

        comp_union: set[Compartment] = set()
        for spec in self._pop_specs:
            comp_union.update(pop_compartments.get(spec.name, spec.model.compartments))
        self._drive_global = {
            comp: torch.zeros((total,), device=self._device, dtype=self._dtype) for comp in comp_union
        }
        self._drive_global_buffers = list(self._drive_global.values())
        self._drive_buffers = {
            spec.name: {
                comp: self._drive_global[comp][self._pop_slices[spec.name]]
                for comp in pop_compartments.get(spec.name, spec.model.compartments)
            }
            for spec in self._pop_specs
        }

        self._compiled_pop_tensor_views = {}
        self._compiled_pop_tensor_keys = {}
        event_plan = self._event_payload_plan or self._event_subsystem.payload_plan(
            self._network_requirements,
            fast_mode=self._fast_mode,
        )
        needs_population_tensors = event_plan.needs_population_tensors
        needs_projection_weights = event_plan.needs_projection_weights
        needs_synapse_state = event_plan.needs_synapse_state
        needs_learning_state = event_plan.needs_learning_state
        needs_homeostasis_state = event_plan.needs_homeostasis_state
        needs_modulator_state = event_plan.needs_modulator_state
        compiled_event_tensors: dict[str, Tensor] = {}
        if self._fast_mode:
            if needs_population_tensors:
                for spec in self._pop_specs:
                    pop_state = self._pop_states[spec.name]
                    if event_plan.needs_spike_tensors:
                        compiled_event_tensors[f"pop/{spec.name}/spikes"] = pop_state.spikes
                    tensors = spec.model.state_tensors(pop_state.state)
                    if event_plan.needs_population_state:
                        keys_for_pop = tuple(tensors.keys())
                    else:
                        keys_for_pop = tuple(key for key in tensors if key == "v_soma")
                    self._compiled_pop_tensor_keys[spec.name] = keys_for_pop
                    for key in keys_for_pop:
                        if key == "spikes":
                            continue
                        value = tensors.get(key)
                        if value is None:
                            continue
                        compiled_event_tensors[f"pop/{spec.name}/{key}"] = value
        else:
            if needs_population_tensors:
                keys: set[str] = set()
                pop_keys: dict[str, set[str]] = {}
                pop_tensors: dict[str, Mapping[str, Tensor]] = {}
                for spec in self._pop_specs:
                    tensors = spec.model.state_tensors(self._pop_states[spec.name].state)
                    if event_plan.needs_population_state:
                        selected = tensors
                    else:
                        v_soma = tensors.get("v_soma")
                        selected = {"v_soma": v_soma} if v_soma is not None else {}
                    pop_tensors[spec.name] = selected
                    pop_key_set = set(selected.keys())
                    pop_keys[spec.name] = pop_key_set
                    keys.update(pop_key_set)

                for key in keys:
                    dtypes = []
                    missing = False
                    for spec in self._pop_specs:
                        tensors = pop_tensors[spec.name]
                        maybe_value = tensors.get(key)
                        if maybe_value is None:
                            missing = True
                            continue
                        dtypes.append(maybe_value.dtype)
                    dtype = dtypes[0] if dtypes else torch.float32
                    for other in dtypes[1:]:
                        dtype = torch.promote_types(dtype, other)
                    if missing:
                        dtype = torch.promote_types(dtype, torch.float32)
                    if getattr(dtype, "is_floating_point", False):
                        buffer = torch.full((total,), float("nan"), device=self._device, dtype=dtype)
                    else:
                        buffer = torch.zeros((total,), device=self._device, dtype=dtype)
                    compiled_event_tensors[key] = buffer

                for spec in self._pop_specs:
                    views: dict[str, Tensor] = {}
                    for key in pop_keys[spec.name]:
                        views[key] = compiled_event_tensors[key][self._pop_slices[spec.name]]
                    self._compiled_pop_tensor_views[spec.name] = views
                    self._compiled_pop_tensor_keys[spec.name] = tuple(pop_keys[spec.name])

        if needs_synapse_state:
            compiled_event_tensors.update(
                self._event_subsystem.projection_tensors(self._proj_specs, self._proj_states)
            )
        elif needs_projection_weights:
            compiled_event_tensors.update(
                self._event_subsystem.projection_weight_tensors(self._proj_specs, self._proj_states)
            )
        if needs_learning_state:
            compiled_event_tensors.update(
                self._event_subsystem.learning_tensors(self._proj_specs, self._proj_states)
            )
        if needs_homeostasis_state:
            compiled_event_tensors.update(self._event_subsystem.homeostasis_tensors(self._homeostasis))
        if needs_modulator_state:
            compiled_event_tensors.update(
                self._event_subsystem.modulator_tensors(self._mod_specs, self._mod_states)
            )
        self._compiled_event_tensors = compiled_event_tensors

        self._compiled_scalars = {
            "step": 0.0,
            "t": 0.0,
            "spike_count_total": torch.zeros((), device=self._device, dtype=self._dtype),
            "spike_fraction_total": torch.zeros((), device=self._device, dtype=self._dtype),
        }
        for spec in self._pop_specs:
            self._compiled_scalars[f"spike_count/{spec.name}"] = torch.zeros(
                (), device=self._device, dtype=self._dtype
            )
        self._compiled_meta = (
            {"population_slices": self._pop_slice_tuples}
            if event_plan.needs_population_slices
            else None
        )

        self._compiled_mod_by_pop = {spec.name: {} for spec in self._pop_specs}
        self._compiled_edge_mods_by_proj = {proj.name: {} for proj in self._proj_specs}

    def _get_learning_scratch(self, proj_name: str) -> _LearningScratch:
        return cast(
            _LearningScratch,
            self._buffer_subsystem.get_learning_scratch(
                scratch_by_proj=cast(dict[str, LearningScratch], self._learning_scratch),
                proj_name=proj_name,
                device=self._device,
            ),
        )


def _validate_specs(populations: Sequence[PopulationSpec], projections: Sequence[ProjectionSpec]) -> None:
    names = {spec.name for spec in populations}
    if len(names) != len(populations):
        raise ValueError("Population names must be unique")
    for proj in projections:
        if proj.pre not in names or proj.post not in names:
            raise ValueError(f"Projection {proj.name} references unknown population")


def _edge_count(topology: SynapseTopology) -> int:
    if hasattr(topology.pre_idx, "numel"):
        return int(topology.pre_idx.numel())
    try:
        return len(topology.pre_idx)
    except TypeError:
        return 0


def _collect_modulator_kinds(mod_specs: Sequence[ModulatorSpec]) -> tuple[ModulatorKind, ...]:
    kinds: list[ModulatorKind] = []
    for spec in mod_specs:
        for kind in spec.kinds:
            if kind not in kinds:
                kinds.append(kind)
    return tuple(kinds)


def _monitors_active(monitors: Sequence[IMonitor]) -> bool:
    if not monitors:
        return False
    return any(getattr(monitor, "enabled", True) for monitor in monitors)


def _collect_monitor_requirements(monitors: Sequence[IMonitor]) -> MonitorRequirements:
    requirements = MonitorRequirements.none()
    for monitor in monitors:
        if not getattr(monitor, "enabled", True):
            continue
        req_fn = getattr(monitor, "requirements", None)
        if callable(req_fn):
            try:
                monitor_requirements = req_fn()
            except Exception:
                monitor_requirements = None
            if isinstance(monitor_requirements, MonitorRequirements):
                requirements = requirements.merge(monitor_requirements)
                continue
        # Backward compatibility: monitors without requirements() get full payloads.
        requirements = requirements.merge(MonitorRequirements.all())
    return requirements


def _collect_monitor_compilation_requirements(monitors: Sequence[IMonitor]) -> dict[str, bool]:
    requirements: dict[str, bool] = {}
    monitor_payload_requirements = _collect_monitor_requirements(monitors)
    if monitor_payload_requirements.needs_projection_weights:
        requirements["wants_weights_snapshot_each_step"] = True
    if monitor_payload_requirements.needs_projection_drive:
        requirements["wants_projection_drive_tensor"] = True

    for monitor in monitors:
        if not getattr(monitor, "enabled", True):
            continue
        req_fn = getattr(monitor, "compilation_requirements", None)
        if not callable(req_fn):
            continue
        try:
            monitor_requirements = req_fn()
        except Exception:
            continue
        if not isinstance(monitor_requirements, Mapping):
            continue
        for key, value in monitor_requirements.items():
            requirements[str(key)] = bool(requirements.get(str(key), False) or bool(value))
    return requirements


def _compile_flags_for_projection(
    proj: ProjectionSpec,
    *,
    monitor_requirements: Mapping[str, bool],
    device: Any,
) -> _ProjectionCompileFlags:
    build_edges_by_delay = False
    build_pre_adjacency = False
    build_sparse_delay_mats = False
    build_bucket_edge_mapping = False
    wants_fused_sparse = False
    wants_by_delay_sparse = False
    wants_bucket_edge_mapping = False
    wants_fused_csr: bool | None = None
    store_sparse_by_delay_override: bool | None = None
    reqs = None
    if hasattr(proj.synapse, "compilation_requirements"):
        try:
            reqs = proj.synapse.compilation_requirements()
        except Exception:
            reqs = None
    if isinstance(reqs, Mapping):
        build_edges_by_delay = bool(reqs.get("needs_edges_by_delay", False))
        build_pre_adjacency = bool(reqs.get("needs_pre_adjacency", False))
        build_sparse_delay_mats = bool(reqs.get("needs_sparse_delay_mats", False))
        build_bucket_edge_mapping = bool(reqs.get("needs_bucket_edge_mapping", False))
        wants_fused_sparse = bool(reqs.get("wants_fused_sparse", False))
        wants_by_delay_sparse = bool(reqs.get("wants_by_delay_sparse", False))
        wants_bucket_edge_mapping = bool(reqs.get("wants_bucket_edge_mapping", False))
        if "wants_fused_csr" in reqs:
            wants_fused_csr = bool(reqs.get("wants_fused_csr", False))
        if "store_sparse_by_delay" in reqs:
            store_sparse_by_delay_override = bool(reqs.get("store_sparse_by_delay"))

    if build_sparse_delay_mats and not wants_fused_sparse and not wants_by_delay_sparse:
        # Backward compatibility for legacy requirement dicts.
        wants_fused_sparse = True

    wants_fused_sparse = bool(wants_fused_sparse or monitor_requirements.get("wants_fused_sparse", False))
    wants_by_delay_sparse = bool(
        wants_by_delay_sparse or monitor_requirements.get("wants_by_delay_sparse", False)
    )
    wants_bucket_edge_mapping = bool(
        wants_bucket_edge_mapping or monitor_requirements.get("wants_bucket_edge_mapping", False)
    )

    if wants_fused_csr is None:
        wants_fused_csr = bool(wants_fused_sparse and _is_cpu_device(device))
    build_fused_csr = bool(wants_fused_sparse and wants_fused_csr)

    build_sparse_delay_mats = bool(
        build_sparse_delay_mats
        or wants_fused_sparse
        or wants_by_delay_sparse
        or wants_bucket_edge_mapping
    )
    build_bucket_edge_mapping = bool(build_bucket_edge_mapping or wants_bucket_edge_mapping)

    if proj.learning is not None:
        build_bucket_edge_mapping = True
    else:
        params = getattr(proj.synapse, "params", None)
        if getattr(params, "enable_sparse_updates", False):
            build_bucket_edge_mapping = True

    supports_sparse = bool(getattr(proj.learning, "supports_sparse", False))
    if proj.sparse_learning and supports_sparse:
        build_pre_adjacency = True

    if build_bucket_edge_mapping:
        build_sparse_delay_mats = True

    if store_sparse_by_delay_override:
        build_sparse_delay_mats = True
    store_sparse_by_delay = bool(build_sparse_delay_mats and wants_by_delay_sparse)
    if store_sparse_by_delay_override is not None:
        store_sparse_by_delay = bool(store_sparse_by_delay_override)
    fuse_delay_buckets = bool(build_sparse_delay_mats and (wants_fused_sparse or not store_sparse_by_delay))

    return _ProjectionCompileFlags(
        build_edges_by_delay=build_edges_by_delay,
        build_pre_adjacency=build_pre_adjacency,
        build_sparse_delay_mats=build_sparse_delay_mats,
        build_bucket_edge_mapping=build_bucket_edge_mapping,
        fuse_delay_buckets=fuse_delay_buckets,
        store_sparse_by_delay=store_sparse_by_delay,
        build_fused_csr=build_fused_csr,
    )


def _is_cpu_device(device: Any) -> bool:
    if device is None:
        return True
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type == "cpu"
    if isinstance(device, str):
        return device.lower().strip().startswith("cpu")
    return False


def _ensure_learning_bucket_mapping(proj: ProjectionSpec, topology: SynapseTopology) -> None:
    if proj.learning is None:
        return
    meta = topology.meta or {}
    if (
        meta.get("edge_bucket_comp") is None
        or meta.get("edge_bucket_delay") is None
        or meta.get("edge_bucket_pos") is None
    ):
        raise RuntimeError(
            f"Projection {proj.name} requires edge bucket mappings for learning, "
            "but they were not built. Enable build_bucket_edge_mapping."
        )


_TORCH_THREAD_LIMIT_LOCK = threading.Lock()
_TORCH_THREAD_LIMIT_ACTIVE = 0
_TORCH_THREAD_LIMIT_STATE: tuple[int | None, int | None, bool] | None = None


def _compile_topology_with_thread_limits(
    topology: SynapseTopology,
    compile_kwargs: Mapping[str, Any],
    torch_threads: int,
) -> SynapseTopology:
    torch = require_torch()
    if torch_threads <= 0 or not hasattr(torch, "set_num_threads"):
        return compile_topology(topology, **compile_kwargs)
    _enter_torch_thread_limits(torch, torch_threads)
    try:
        return compile_topology(topology, **compile_kwargs)
    finally:
        _exit_torch_thread_limits(torch)


def _enter_torch_thread_limits(torch: Any, torch_threads: int) -> None:
    global _TORCH_THREAD_LIMIT_ACTIVE, _TORCH_THREAD_LIMIT_STATE
    with _TORCH_THREAD_LIMIT_LOCK:
        if _TORCH_THREAD_LIMIT_ACTIVE == 0:
            prev_threads = torch.get_num_threads() if hasattr(torch, "get_num_threads") else None
            prev_interop = (
                torch.get_num_interop_threads()
                if hasattr(torch, "get_num_interop_threads")
                else None
            )
            interop_set = False
            torch.set_num_threads(int(torch_threads))
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                    interop_set = True
                except RuntimeError:
                    prev_interop = None
            _TORCH_THREAD_LIMIT_STATE = (prev_threads, prev_interop, interop_set)
        _TORCH_THREAD_LIMIT_ACTIVE += 1


def _exit_torch_thread_limits(torch: Any) -> None:
    global _TORCH_THREAD_LIMIT_ACTIVE, _TORCH_THREAD_LIMIT_STATE
    with _TORCH_THREAD_LIMIT_LOCK:
        _TORCH_THREAD_LIMIT_ACTIVE = max(0, _TORCH_THREAD_LIMIT_ACTIVE - 1)
        if _TORCH_THREAD_LIMIT_ACTIVE == 0 and _TORCH_THREAD_LIMIT_STATE is not None:
            prev_threads, prev_interop, interop_set = _TORCH_THREAD_LIMIT_STATE
            if prev_threads is not None:
                torch.set_num_threads(prev_threads)
            if (
                interop_set
                and prev_interop is not None
                and hasattr(torch, "set_num_interop_threads")
            ):
                with contextlib.suppress(RuntimeError):
                    torch.set_num_interop_threads(prev_interop)
            _TORCH_THREAD_LIMIT_STATE = None


def _resolve_parallel_compile(
    mode: str,
    job_count: int,
    device: Any,
    workers: int | None,
) -> tuple[bool, int]:
    if device is not None and getattr(device, "type", None) == "cuda":
        return False, 1
    if job_count < 2:
        return False, 1
    mode_norm = mode.lower().strip()
    cpu_count = os.cpu_count() or 1
    if mode_norm == "off":
        return False, 1
    if mode_norm == "auto" and cpu_count < 4:
        return False, 1
    max_workers = workers if workers is not None else min(cpu_count, job_count)
    max_workers = max(1, min(int(max_workers), job_count))
    return True, max_workers


def _apply_initial_spikes(
    pop_states: Mapping[str, _PopulationState],
    pop_order: Sequence[str],
    meta: Mapping[str, Any] | None,
) -> None:
    if meta is None:
        return
    by_pop = meta.get("initial_spikes_by_pop")
    if isinstance(by_pop, Mapping):
        for name, indices in by_pop.items():
            if name not in pop_states:
                continue
            _apply_indices(pop_states[name].spikes, indices)
        return
    indices = meta.get("initial_spike_indices")
    if indices is None:
        return
    if pop_order:
        _apply_indices(pop_states[pop_order[0]].spikes, indices)


def _apply_indices(spikes: Tensor, indices: Any) -> None:
    if hasattr(indices, "numel"):
        if indices.numel() == spikes.numel():
            spikes.copy_(indices.to(device=spikes.device, dtype=spikes.dtype))
            return
        idx_tensor = indices.to(device=spikes.device, dtype=require_torch().long)
        spikes[idx_tensor] = True
        return

    if isinstance(indices, (list, tuple)):
        for idx in indices:
            spikes[int(idx)] = True
        return

    spikes[int(indices)] = True


def _copy_topology_weights(state: Any, weights: Tensor | None) -> None:
    if weights is None or not hasattr(state, "weights"):
        if weights is None and getattr(state, "bind_weights_to_topology", False):
            # Topology will be set to state weights later if needed.
            return
        return
    state_weights = state.weights
    if not hasattr(state_weights, "shape"):
        return
    if getattr(state, "bind_weights_to_topology", False):
        state.weights = weights
        return
    if state_weights.shape != weights.shape:
        return
    if hasattr(weights, "to"):
        weights = weights.to(device=state_weights.device, dtype=state_weights.dtype)
    state_weights.copy_(weights)


def _accumulate_drive(
    target: dict[Compartment, Tensor],
    update: Mapping[Compartment, Tensor],
) -> None:
    for comp, tensor in update.items():
        if comp not in target:
            raise KeyError(f"Drive accumulator missing compartment {comp}")
        existing = target[comp]
        if hasattr(existing, "add_"):
            existing.add_(tensor)
        else:
            target[comp] = existing + tensor


def _zero_drive(model: INeuronModel, n: int, device: Any, dtype: Any) -> dict[Compartment, Tensor]:
    torch = require_torch()
    return {comp: torch.zeros((n,), device=device, dtype=dtype) for comp in model.compartments}


def _collect_drive_compartments(
    pop_specs: Sequence[PopulationSpec],
    proj_specs: Sequence[ProjectionSpec],
) -> dict[str, tuple[Compartment, ...]]:
    comp_order = tuple(Compartment)
    comps_by_pop: dict[str, set[Compartment]] = {
        spec.name: set(spec.model.compartments) for spec in pop_specs
    }
    for proj in proj_specs:
        comps = comps_by_pop.get(proj.post)
        if comps is None:
            continue
        topology = proj.topology
        if topology.target_compartments is None:
            comps.add(topology.target_compartment)
            continue
        meta = topology.meta or {}
        comp_ids = meta.get("target_comp_ids")
        if isinstance(comp_ids, list) and comp_ids:
            for comp_id in comp_ids:
                try:
                    idx = int(comp_id)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(comp_order):
                    comps.add(comp_order[idx])
            continue
        comps.update(comp_order)
    return {name: tuple(comps) for name, comps in comps_by_pop.items()}


def _step_modulators(
    mod_specs: Sequence[ModulatorSpec],
    mod_states: dict[str, Any],
    pop_specs: Sequence[PopulationSpec],
    proj_specs: Sequence[ProjectionSpec],
    t: float,
    step: int,
    dt: float,
    ctx: StepContext,
    device: Any,
    dtype: Any,
    kinds: tuple[ModulatorKind, ...],
    releases: Sequence[ModulatorRelease],
) -> tuple[
    dict[str, dict[ModulatorKind, Tensor]] | None,
    dict[str, dict[ModulatorKind, Tensor]] | None,
]:
    if not mod_specs:
        return None, None

    torch = require_torch()
    mod_by_pop: dict[str, dict[ModulatorKind, Tensor]] = {}
    edge_mods_by_proj: dict[str, dict[ModulatorKind, Tensor]] = {}
    pop_map = {spec.name: spec for spec in pop_specs}

    for spec in mod_specs:
        state = mod_states[spec.name]
        rel = [r for r in releases if r.kind in spec.kinds]
        mod_states[spec.name] = spec.field.step(state, releases=rel, dt=dt, t=t, ctx=ctx)

    for pop in pop_specs:
        if pop.positions is None:
            continue
        for kind in kinds:
            total = torch.zeros((pop.n,), device=device, dtype=dtype)
            for spec in mod_specs:
                if kind not in spec.kinds:
                    continue
                state = mod_states[spec.name]
                sampled = spec.field.sample_at(state, positions=pop.positions, kind=kind, ctx=ctx)
                total = total + sampled
            mod_by_pop.setdefault(pop.name, {})[kind] = total

    for proj in proj_specs:
        post_pop = pop_map[proj.post]
        mods = mod_by_pop.get(post_pop.name)
        edge_len = _edge_count(proj.topology)
        if mods is None:
            edge_mods_by_proj[proj.name] = {
                kind: torch.zeros((edge_len,), device=device, dtype=dtype) for kind in kinds
            }
            continue
        edge_mods: dict[ModulatorKind, Tensor] = {}
        for kind in kinds:
            mod_post = mods.get(kind)
            if mod_post is None:
                edge_mods[kind] = torch.zeros((edge_len,), device=device, dtype=dtype)
            else:
                edge_mods[kind] = mod_post[proj.topology.post_idx]
        edge_mods_by_proj[proj.name] = edge_mods

    return mod_by_pop, edge_mods_by_proj


def _step_modulators_compiled(
    mod_specs: Sequence[ModulatorSpec],
    mod_states: dict[str, Any],
    pop_specs: Sequence[PopulationSpec],
    proj_specs: Sequence[ProjectionSpec],
    t: float,
    step: int,
    dt: float,
    ctx: StepContext,
    device: Any,
    dtype: Any,
    kinds: tuple[ModulatorKind, ...],
    pop_map: Mapping[str, PopulationSpec],
    mod_by_pop: dict[str, dict[ModulatorKind, Tensor]],
    edge_mods_by_proj: dict[str, dict[ModulatorKind, Tensor]],
    releases: Sequence[ModulatorRelease],
) -> tuple[dict[str, dict[ModulatorKind, Tensor]], dict[str, dict[ModulatorKind, Tensor]]]:
    if not mod_specs:
        return mod_by_pop, edge_mods_by_proj

    torch = require_torch()
    for spec in mod_specs:
        state = mod_states[spec.name]
        rel = [r for r in releases if r.kind in spec.kinds]
        mod_states[spec.name] = spec.field.step(state, releases=rel, dt=dt, t=t, ctx=ctx)

    for pop in pop_specs:
        pop_dict = mod_by_pop.get(pop.name)
        if pop_dict is None:
            pop_dict = {}
            mod_by_pop[pop.name] = pop_dict
        else:
            pop_dict.clear()
        if pop.positions is None:
            continue
        for kind in kinds:
            total = torch.zeros((pop.n,), device=device, dtype=dtype)
            for spec in mod_specs:
                if kind not in spec.kinds:
                    continue
                state = mod_states[spec.name]
                sampled = spec.field.sample_at(state, positions=pop.positions, kind=kind, ctx=ctx)
                total = total + sampled
            pop_dict[kind] = total

    for proj in proj_specs:
        proj_dict = edge_mods_by_proj.get(proj.name)
        if proj_dict is None:
            proj_dict = {}
            edge_mods_by_proj[proj.name] = proj_dict
        else:
            proj_dict.clear()
        post_pop = pop_map[proj.post]
        mods = mod_by_pop.get(post_pop.name)
        edge_len = _edge_count(proj.topology)
        if mods is None:
            for kind in kinds:
                proj_dict[kind] = torch.zeros((edge_len,), device=device, dtype=dtype)
            continue
        for kind in kinds:
            mod_post = mods.get(kind)
            if mod_post is None:
                proj_dict[kind] = torch.zeros((edge_len,), device=device, dtype=dtype)
            else:
                proj_dict[kind] = mod_post[proj.topology.post_idx]

    return mod_by_pop, edge_mods_by_proj


def _require_weights(state: Any, proj_name: str) -> Tensor:
    if not hasattr(state, "weights"):
        raise RuntimeError(f"Projection {proj_name} learning requires synapse weights")
    return cast(Tensor, state.weights)


def _apply_weight_clamp(weights: Tensor, proj: ProjectionSpec) -> None:
    clamp_min = None
    clamp_max = None
    if proj.meta:
        clamp_min = proj.meta.get("clamp_min")
        clamp_max = proj.meta.get("clamp_max")
    if hasattr(proj.synapse, "params"):
        params = proj.synapse.params
        clamp_min = getattr(params, "clamp_min", clamp_min)
        clamp_max = getattr(params, "clamp_max", clamp_max)
    if clamp_min is None and clamp_max is None:
        return
    weights.clamp_(min=clamp_min, max=clamp_max)
    sync_fn = getattr(proj.synapse, "sync_sparse_values", None)
    if callable(sync_fn):
        sync_fn(proj.topology)


def _merge_population_tensors(
    tensors_by_pop: Mapping[str, Mapping[str, Tensor]],
    pop_specs: Sequence[PopulationSpec],
    device: Any,
) -> dict[str, Tensor]:
    torch = require_torch()
    keys: set[str] = set()
    for tensors in tensors_by_pop.values():
        keys.update(tensors.keys())

    merged: dict[str, Tensor] = {}
    for key in keys:
        parts: list[Tensor] = []
        for spec in pop_specs:
            value = tensors_by_pop.get(spec.name, {}).get(key)
            if value is None:
                parts.append(torch.full((spec.n,), float("nan"), device=device, dtype=torch.float32))
            else:
                parts.append(value)
        merged[key] = torch.cat(parts)

    return merged


def _build_population_tensors(
    tensors_by_pop: Mapping[str, Mapping[str, Tensor]],
    pop_specs: Sequence[PopulationSpec],
    pop_states: Mapping[str, _PopulationState],
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for spec in pop_specs:
        pop_name = spec.name
        spikes = pop_states[pop_name].spikes
        tensors[f"pop/{pop_name}/spikes"] = spikes
        for key, value in tensors_by_pop.get(pop_name, {}).items():
            if key == "spikes":
                continue
            tensors[f"pop/{pop_name}/{key}"] = value
    return tensors


def _summarize_spikes(
    pop_specs: Sequence[PopulationSpec],
    pop_states: Mapping[str, _PopulationState],
) -> tuple[Tensor, Tensor]:
    torch = require_torch()
    total_spikes = None
    total_neurons = 0
    for spec in pop_specs:
        total_neurons += spec.n
        spikes = pop_states[spec.name].spikes
        count = spikes.sum()
        total_spikes = count if total_spikes is None else total_spikes + count
    if total_neurons <= 0 or total_spikes is None:
        device = None
        dtype = None
        if pop_specs:
            sample = pop_states[pop_specs[0].name].spikes
            device = sample.device
            dtype = sample.dtype
        return (
            torch.zeros((), device=device, dtype=dtype or torch.float32),
            torch.zeros((), device=device, dtype=dtype or torch.float32),
        )
    spike_fraction = total_spikes / float(total_neurons)
    return total_spikes, spike_fraction


def _validate_fast_mode_monitors(monitors: Sequence[IMonitor]) -> None:
    try:
        from biosnn.monitors.csv import NeuronCSVMonitor, SynapseCSVMonitor
        from biosnn.monitors.raster.spike_events_csv import SpikeEventsCSVMonitor
    except Exception:
        return

    incompatible = [
        monitor
        for monitor in monitors
        if isinstance(monitor, (NeuronCSVMonitor, SynapseCSVMonitor, SpikeEventsCSVMonitor))
    ]
    if incompatible:
        names = ", ".join(type(mon).__name__ for mon in incompatible)
        raise RuntimeError(
            "fast_mode=True does not support monitors that require merged tensors or global spikes. "
            f"Incompatible monitors: {names}"
        )


def _projection_tensors(
    proj_specs: Sequence[ProjectionSpec],
    proj_states: Mapping[str, _ProjectionState],
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for proj in proj_specs:
        state = proj_states[proj.name].state
        for key, value in proj.synapse.state_tensors(state).items():
            tensors[f"proj/{proj.name}/{key}"] = value
    return tensors


def _projection_weight_tensors(
    proj_specs: Sequence[ProjectionSpec],
    proj_states: Mapping[str, _ProjectionState],
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for proj in proj_specs:
        state = proj_states[proj.name].state
        weights = getattr(state, "weights", None)
        if weights is not None:
            tensors[f"proj/{proj.name}/weights"] = cast(Tensor, weights)
    return tensors


def _learning_tensors(
    proj_specs: Sequence[ProjectionSpec],
    proj_states: Mapping[str, _ProjectionState],
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for proj in proj_specs:
        if proj.learning is None:
            continue
        state = proj_states[proj.name].learning_state
        if state is None:
            continue
        for key, value in proj.learning.state_tensors(state).items():
            tensors[f"learn/{proj.name}/{key}"] = value
    return tensors


def _homeostasis_tensors(homeostasis: IHomeostasisRule | None) -> dict[str, Tensor]:
    if homeostasis is None:
        return {}
    return dict(homeostasis.state_tensors())


def _modulator_tensors(
    mod_specs: Sequence[ModulatorSpec],
    mod_states: Mapping[str, Any],
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for spec in mod_specs:
        state = mod_states[spec.name]
        for key, value in spec.field.state_tensors(state).items():
            tensors[f"mod/{spec.name}/{key}"] = value
    return tensors


def _sparse_edge_cap(proj: ProjectionSpec) -> int | None:
    if proj.meta and "sparse_max_edges" in proj.meta:
        try:
            cap = int(proj.meta["sparse_max_edges"])
        except (TypeError, ValueError):
            return None
        return cap if cap > 0 else None
    return None


def _subset_edge_mods(
    edge_mods: Mapping[ModulatorKind, Tensor] | None,
    edges: Tensor,
) -> Mapping[ModulatorKind, Tensor] | None:
    if edge_mods is None:
        return None
    return {kind: value.index_select(0, edges) for kind, value in edge_mods.items()}


def _build_learning_batch(
    *,
    proj: ProjectionSpec,
    pre_spikes: Tensor,
    post_spikes: Tensor,
    weights: Tensor,
    edge_mods: Mapping[ModulatorKind, Tensor] | None,
    device: Any,
    use_sparse: bool,
    scratch: _LearningScratch | None,
) -> tuple[LearningBatch, Tensor | None]:
    if use_sparse:
        active_edges = _active_edges_from_spikes(
            pre_spikes,
            proj.topology,
            device=device,
            max_edges=_sparse_edge_cap(proj),
            scratch=scratch,
        )
        if scratch is not None:
            n_active = active_edges.numel()
            if (
                scratch.size != n_active
                or scratch.edge_pre_idx.device != device
                or scratch.edge_pre.dtype != pre_spikes.dtype
                or scratch.edge_post.dtype != post_spikes.dtype
                or scratch.edge_weights.dtype != weights.dtype
            ):
                torch = require_torch()
                scratch.edge_pre_idx = torch.empty((n_active,), device=device, dtype=torch.long)
                scratch.edge_post_idx = torch.empty((n_active,), device=device, dtype=torch.long)
                scratch.edge_pre = torch.empty((n_active,), device=device, dtype=pre_spikes.dtype)
                scratch.edge_post = torch.empty((n_active,), device=device, dtype=post_spikes.dtype)
                scratch.edge_weights = torch.empty((n_active,), device=device, dtype=weights.dtype)
                scratch.size = n_active

            edge_pre_idx = scratch.edge_pre_idx
            edge_post_idx = scratch.edge_post_idx
            edge_pre = scratch.edge_pre
            edge_post = scratch.edge_post
            edge_weights = scratch.edge_weights

            torch = require_torch()
            torch.index_select(proj.topology.pre_idx, 0, active_edges, out=edge_pre_idx)
            torch.index_select(proj.topology.post_idx, 0, active_edges, out=edge_post_idx)
            torch.index_select(pre_spikes, 0, edge_pre_idx, out=edge_pre)
            torch.index_select(post_spikes, 0, edge_post_idx, out=edge_post)
            torch.index_select(weights, 0, active_edges, out=edge_weights)
        else:
            edge_pre_idx = proj.topology.pre_idx.index_select(0, active_edges)
            edge_post_idx = proj.topology.post_idx.index_select(0, active_edges)
            edge_pre = pre_spikes.index_select(0, edge_pre_idx)
            edge_post = post_spikes.index_select(0, edge_post_idx)
            edge_weights = weights.index_select(0, active_edges)

        mods = _subset_edge_mods(edge_mods, active_edges)
        batch = LearningBatch(
            pre_spikes=edge_pre,
            post_spikes=edge_post,
            weights=edge_weights,
            modulators=mods,
            extras={
                "pre_idx": edge_pre_idx,
                "post_idx": edge_post_idx,
                "active_edges": active_edges,
            },
        )
        return batch, active_edges

    edge_pre = pre_spikes[proj.topology.pre_idx]
    edge_post = post_spikes[proj.topology.post_idx]
    batch = LearningBatch(
        pre_spikes=edge_pre,
        post_spikes=edge_post,
        weights=weights,
        modulators=edge_mods,
        extras={
            "pre_idx": proj.topology.pre_idx,
            "post_idx": proj.topology.post_idx,
        },
    )
    return batch, None


def _require_pre_adjacency(topology: SynapseTopology, device: Any) -> tuple[Tensor, Tensor]:
    if not topology.meta:
        raise ValueError(
            "Topology meta missing pre adjacency; compile_topology(..., build_pre_adjacency=True) "
            "must be called."
        )
    pre_ptr = topology.meta.get("pre_ptr")
    edge_idx = topology.meta.get("edge_idx")
    if pre_ptr is None or edge_idx is None:
        raise ValueError(
            "Topology meta missing pre adjacency; compile_topology(..., build_pre_adjacency=True) "
            "must be called."
        )
    if pre_ptr.device != device or edge_idx.device != device:
        raise ValueError("Adjacency tensors must be on the projection device")
    return cast(Tensor, pre_ptr), cast(Tensor, edge_idx)


def _get_arange(scratch: _LearningScratch, needed: int, *, device: Any) -> Tensor:
    if scratch.arange_size < needed or scratch.arange_buf.device != device:
        torch = require_torch()
        scratch.arange_buf = torch.arange(needed, device=device, dtype=torch.long)
        scratch.arange_size = needed
    return scratch.arange_buf


def _gather_active_edges(
    active_pre: Tensor,
    pre_ptr: Tensor,
    edge_idx: Tensor,
    *,
    scratch: _LearningScratch | None = None,
) -> Tensor:
    torch = require_torch()
    starts = pre_ptr.index_select(0, active_pre)
    ends = pre_ptr.index_select(0, active_pre + 1)
    counts = ends - starts
    if counts.numel() == 0:
        return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))
    base = torch.repeat_interleave(starts, counts)
    total = base.numel()
    if total == 0:
        return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))
    prefix = torch.cumsum(counts, 0)
    group_start = torch.repeat_interleave(prefix - counts, counts)
    if scratch is not None:
        intra = _get_arange(scratch, total, device=edge_idx.device)[:total] - group_start
    else:
        intra = torch.arange(total, device=edge_idx.device) - group_start
    edge_pos = base + intra
    return cast(Tensor, edge_idx.index_select(0, edge_pos))


def _active_edges_from_spikes(
    pre_spikes: Tensor,
    topology: SynapseTopology,
    *,
    device: Any,
    max_edges: int | None = None,
    scratch: _LearningScratch | None = None,
) -> Tensor:
    torch = require_torch()
    pre_ptr, edge_idx = _require_pre_adjacency(topology, device)
    active_pre = pre_spikes.nonzero(as_tuple=False).flatten()
    if active_pre.numel() == 0:
        return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))
    edges = _gather_active_edges(active_pre, pre_ptr, edge_idx, scratch=scratch)
    if max_edges is not None and edges.numel() > max_edges:
        return edges[:max_edges]
    return edges


def _apply_learning_update(
    *,
    weights: Tensor,
    result: LearningStepResult,
    active_edges: Tensor | None,
    proj_name: str,
    synapse: Any | None = None,
    topology: SynapseTopology | None = None,
) -> None:
    if active_edges is None and result.extras is not None:
        maybe_edges = result.extras.get("active_edges")
        if isinstance(maybe_edges, Tensor):
            active_edges = maybe_edges

    def _apply_to_weights(target: Tensor) -> None:
        if active_edges is None:
            target.add_(result.d_weights)
            return
        if result.d_weights.numel() == active_edges.numel():
            target.index_add_(0, active_edges, result.d_weights)
            return
        if result.d_weights.numel() == target.numel():
            target.add_(result.d_weights)
            return
        raise RuntimeError(
            "Sparse learning update shape mismatch for "
            f"{proj_name}: d_weights={tuple(result.d_weights.shape)} "
            f"active_edges={tuple(active_edges.shape)} "
            f"weights={tuple(target.shape)}"
        )

    if synapse is not None and topology is not None:
        apply_fn = getattr(synapse, "apply_weight_updates", None)
        if callable(apply_fn):
            apply_fn(topology, active_edges, result.d_weights)
            topo_weights = topology.weights
            if topo_weights is None or topo_weights is not weights:
                _apply_to_weights(weights)
            return

    _apply_to_weights(weights)


def _ensure_topology_meta(
    topology: SynapseTopology,
    *,
    n_pre: int | None = None,
    n_post: int | None = None,
) -> SynapseTopology:
    meta = dict(topology.meta) if topology.meta else {}
    updated = False

    if n_pre is not None and "n_pre" not in meta:
        meta["n_pre"] = int(n_pre)
        updated = True
    if n_post is not None and "n_post" not in meta:
        meta["n_post"] = int(n_post)
        updated = True

    if topology.delay_steps is not None and "max_delay_steps" not in meta:
        delay_steps = topology.delay_steps
        max_delay = 0
        if hasattr(delay_steps, "numel") and delay_steps.numel():
            max_val = delay_steps.detach()
            if hasattr(max_val, "max"):
                max_val = max_val.max()
            if hasattr(max_val, "cpu"):
                max_val = max_val.cpu()
            if hasattr(max_val, "tolist"):
                max_list = max_val.tolist()
                if isinstance(max_list, list):
                    scalar = max_list[0] if max_list else 0
                    max_delay = int(cast(SupportsInt, scalar))
                else:
                    max_delay = int(cast(SupportsInt, max_list))
            else:
                max_delay = int(max_val)
        meta["max_delay_steps"] = max_delay
        updated = True

    if updated:
        return replace(topology, meta=meta)
    return topology


def _check_ring_buffer_budget(
    *,
    proj_name: str,
    topology: SynapseTopology,
    max_ring_mib: float | None,
    requires_ring: bool,
) -> None:
    if not requires_ring:
        return
    if max_ring_mib is None:
        return
    try:
        max_mib = float(max_ring_mib)
    except (TypeError, ValueError):
        return
    if max_mib <= 0:
        return
    meta = topology.meta or {}
    est_mib = meta.get("estimated_ring_mib")
    if est_mib is None:
        return
    try:
        est_val = float(est_mib)
    except (TypeError, ValueError):
        return
    if est_val <= max_mib:
        return
    ring_len = meta.get("ring_len", "unknown")
    n_post = meta.get("n_post", "unknown")
    if topology.weights is not None and hasattr(topology.weights, "dtype"):
        dtype = str(topology.weights.dtype)
    else:
        dtype = str(meta.get("dtype", "unknown"))
    raise RuntimeError(
        f"Projection '{proj_name}' ring buffer estimate {est_val:.2f} MiB exceeds "
        f"max_ring_mib={max_mib:.2f} MiB. "
        f"ring_len={ring_len}, n_post={n_post}, dtype={dtype}. "
        "Reduce max_delay_steps, reduce n_post, use smaller delays, or choose a smaller dtype "
        "to lower ring buffer memory, or increase max_ring_mib."
    )


__all__ = ["TorchNetworkEngine"]
