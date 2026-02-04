"""Torch-backed multi-population simulation engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, SupportsInt, cast

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.monitors import IMonitor, Scalar, StepEvent
from biosnn.contracts.neurons import Compartment, INeuronModel, NeuronInputs, StepContext
from biosnn.contracts.simulation import ISimulationEngine, SimulationConfig
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype
from biosnn.simulation.network.specs import ModulatorSpec, PopulationSpec, ProjectionSpec

ExternalDriveFn = Callable[[float, int, str, StepContext], Mapping[Compartment, Tensor]]
ReleasesFn = Callable[[float, int, StepContext], Sequence[ModulatorRelease]]


@dataclass(slots=True)
class _ProjectionState:
    state: Any
    learning_state: Any | None


@dataclass(slots=True)
class _PopulationState:
    state: Any
    spikes: Tensor


class TorchNetworkEngine(ISimulationEngine):
    """Multi-population engine with projections, modulators, and learning."""

    name = "torch_network"

    def __init__(
        self,
        *,
        populations: Sequence[PopulationSpec],
        projections: Sequence[ProjectionSpec],
        modulators: Sequence[ModulatorSpec] | None = None,
        external_drive_fn: ExternalDriveFn | None = None,
        releases_fn: ReleasesFn | None = None,
        fast_mode: bool = False,
        compiled_mode: bool = False,
    ) -> None:
        self._pop_specs = list(populations)
        self._proj_specs = list(projections)
        self._mod_specs = list(modulators) if modulators is not None else []
        self._external_drive_fn = external_drive_fn
        self._releases_fn = releases_fn
        self._fast_mode = bool(fast_mode)
        self._compiled_mode = bool(compiled_mode)

        self._ctx = StepContext()
        self._dt = 0.0
        self._t = 0.0
        self._step = 0
        self._device = None
        self._dtype = None

        self._pop_states: dict[str, _PopulationState] = {}
        self._proj_states: dict[str, _ProjectionState] = {}
        self._mod_states: dict[str, Any] = {}
        self._monitors: list[IMonitor] = []
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
        self._last_event: StepEvent | None = None

        self.last_projection_drive: dict[str, Mapping[Compartment, Tensor]] = {}
        self.last_d_weights: dict[str, Tensor] = {}

        self._pop_order = [spec.name for spec in self._pop_specs]
        self._pop_map = {spec.name: spec for spec in self._pop_specs}
        self._modulator_kinds = _collect_modulator_kinds(self._mod_specs)

        _validate_specs(self._pop_specs, self._proj_specs)

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

        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        self._pop_states.clear()
        for spec in self._pop_specs:
            state = spec.model.init_state(spec.n, ctx=self._ctx)
            spikes = torch.zeros((spec.n,), device=self._device, dtype=torch.bool)
            self._pop_states[spec.name] = _PopulationState(state=state, spikes=spikes)

        self._proj_states.clear()
        for proj in self._proj_specs:
            edge_count = _edge_count(proj.topology)
            syn_state = proj.synapse.init_state(edge_count, ctx=self._ctx)
            learn_state = None
            if proj.learning is not None:
                learn_state = proj.learning.init_state(edge_count, ctx=self._ctx)
            self._proj_states[proj.name] = _ProjectionState(state=syn_state, learning_state=learn_state)
            _copy_topology_weights(syn_state, proj.topology.weights)

        pop_map = {spec.name: spec for spec in self._pop_specs}
        for proj in self._proj_specs:
            _ensure_topology_meta(
                proj.topology,
                n_pre=pop_map[proj.pre].n,
                n_post=pop_map[proj.post].n,
            )
            build_edges, build_adj = _compile_flags_for_projection(proj)
            compile_topology(
                proj.topology,
                device=self._device,
                dtype=self._dtype,
                build_edges_by_delay=build_edges,
                build_pre_adjacency=build_adj,
            )

        self._mod_states.clear()
        for mod in self._mod_specs:
            self._mod_states[mod.name] = mod.field.init_state(ctx=self._ctx)

        pop_compartments = _collect_drive_compartments(self._pop_specs, self._proj_specs)
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

    def attach_monitors(self, monitors: Sequence[IMonitor]) -> None:
        monitor_list = list(monitors)
        if self._fast_mode:
            _validate_fast_mode_monitors(monitor_list)
        self._monitors = monitor_list

    def step(self) -> Mapping[str, Any]:
        torch = require_torch()
        if self._compiled_mode:
            return self._step_compiled()

        no_monitors = not _monitors_active(self._monitors)
        mod_by_pop, edge_mods_by_proj = _step_modulators(
            self._mod_specs,
            self._mod_states,
            self._pop_specs,
            self._proj_specs,
            self._t,
            self._step,
            self._dt,
            self._ctx,
            self._device,
            self._dtype,
            self._modulator_kinds,
            releases=self._resolve_releases(),
        )

        mod_by_pop = mod_by_pop or {}
        edge_mods_by_proj = edge_mods_by_proj or {}

        drive_acc = self._drive_buffers
        for drive_by_comp in drive_acc.values():
            for buffer in drive_by_comp.values():
                buffer.zero_()

        self.last_projection_drive = {}
        for proj in self._proj_specs:
            proj_state = self._proj_states[proj.name]
            pre_spikes = self._pop_states[proj.pre].spikes
            if hasattr(proj.synapse, "step_into"):
                proj.synapse.step_into(
                    proj_state.state,
                    pre_spikes,
                    drive_acc[proj.post],
                    t=self._t,
                    topology=proj.topology,
                    dt=self._dt,
                    ctx=self._ctx,
                )
                self.last_projection_drive[proj.name] = drive_acc[proj.post]
            else:
                proj_state.state, syn_result = proj.synapse.step(
                    proj_state.state,
                    proj.topology,
                    SynapseInputs(pre_spikes=pre_spikes),
                    dt=self._dt,
                    t=self._t,
                    ctx=self._ctx,
                )
                self.last_projection_drive[proj.name] = syn_result.post_drive
                _accumulate_drive(
                    drive_acc[proj.post],
                    syn_result.post_drive,
                )

        if self._external_drive_fn is not None:
            for spec in self._pop_specs:
                extra = self._external_drive_fn(self._t, self._step, spec.name, self._ctx)
                _accumulate_drive(drive_acc[spec.name], extra)

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
            if pop_state.spikes.shape == spikes.shape:
                pop_state.spikes.copy_(spikes.to(device=pop_state.spikes.device))
                spikes = pop_state.spikes
            else:
                pop_state.spikes = spikes.to(device=pop_state.spikes.device)
            self._pop_states[spec.name] = pop_state

            if not no_monitors:
                start = offset
                end = offset + spec.n
                pop_slices[spec.name] = (start, end)
                offset = end
                if not self._fast_mode:
                    spikes_concat.append(spikes)
                neuron_tensors_by_pop[spec.name] = spec.model.state_tensors(pop_state.state)

        self.last_d_weights = {}
        for proj in self._proj_specs:
            if proj.learning is None:
                continue
            if proj.learn_every <= 0 or (self._step % proj.learn_every) != 0:
                continue
            proj_state = self._proj_states[proj.name]
            learn_state = proj_state.learning_state
            if learn_state is None:
                continue

            weights = _require_weights(proj_state.state, proj.name)
            edge_pre = self._pop_states[proj.pre].spikes[proj.topology.pre_idx]
            edge_post = self._pop_states[proj.post].spikes[proj.topology.post_idx]
            batch = LearningBatch(
                pre_spikes=edge_pre,
                post_spikes=edge_post,
                weights=weights,
                modulators=edge_mods_by_proj.get(proj.name),
                extras={
                    "pre_idx": proj.topology.pre_idx,
                    "post_idx": proj.topology.post_idx,
                },
            )
            new_state, res = proj.learning.step(
                learn_state,
                batch,
                dt=self._dt,
                t=self._t,
                ctx=self._ctx,
            )
            proj_state.learning_state = new_state
            weights.add_(res.d_weights)
            _apply_weight_clamp(weights, proj)
            self.last_d_weights[proj.name] = res.d_weights

        if no_monitors:
            event = StepEvent(t=self._t, dt=self._dt)
            self._last_event = event
            t_now = self._t
            step_now = self._step
            self._t += self._dt
            self._step += 1
            return {"step": float(step_now), "t": float(t_now)}

        if self._fast_mode:
            spikes_global = None
            spike_count, spike_fraction = _summarize_spikes(self._pop_specs, self._pop_states)
            event_tensors = _build_population_tensors(
                neuron_tensors_by_pop,
                self._pop_specs,
                self._pop_states,
            )
        else:
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
            event_tensors = _merge_population_tensors(
                neuron_tensors_by_pop,
                self._pop_specs,
                self._device,
            )
        event_tensors.update(_projection_tensors(self._proj_specs, self._proj_states))
        event_tensors.update(_learning_tensors(self._proj_specs, self._proj_states))
        event_tensors.update(_modulator_tensors(self._mod_specs, self._mod_states))

        scalars: dict[str, Scalar] = {
            "step": float(self._step),
            "t": float(self._t),
            "spike_count_total": spike_count,
            "spike_fraction_total": spike_fraction,
        }
        for spec in self._pop_specs:
            scalars[f"spike_count/{spec.name}"] = self._pop_states[spec.name].spikes.sum()

        event = StepEvent(
            t=self._t,
            dt=self._dt,
            spikes=spikes_global,
            tensors=event_tensors,
            scalars=cast(Mapping[str, Scalar], scalars),
            meta={"population_slices": pop_slices},
        )
        self._last_event = event

        for monitor in self._monitors:
            monitor.on_step(event)

        self._t += self._dt
        self._step += 1

        return scalars

    def _step_compiled(self) -> Mapping[str, Any]:
        if self._spikes_global is None:
            raise RuntimeError("Engine must be reset before stepping.")

        torch = require_torch()
        no_monitors = not _monitors_active(self._monitors)
        if self._mod_specs:
            mod_by_pop, edge_mods_by_proj = _step_modulators_compiled(
                self._mod_specs,
                self._mod_states,
                self._pop_specs,
                self._proj_specs,
                self._t,
                self._step,
                self._dt,
                self._ctx,
                self._device,
                self._dtype,
                self._modulator_kinds,
                self._pop_map,
                self._compiled_mod_by_pop,
                self._compiled_edge_mods_by_proj,
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
        for proj in self._proj_specs:
            proj_state = self._proj_states[proj.name]
            pre_spikes = self._pop_states[proj.pre].spikes
            if hasattr(proj.synapse, "step_into"):
                proj.synapse.step_into(
                    proj_state.state,
                    pre_spikes,
                    self._drive_buffers[proj.post],
                    t=self._t,
                    topology=proj.topology,
                    dt=self._dt,
                    ctx=self._ctx,
                )
                self.last_projection_drive[proj.name] = self._drive_buffers[proj.post]
            else:
                proj_state.state, syn_result = proj.synapse.step(
                    proj_state.state,
                    proj.topology,
                    SynapseInputs(pre_spikes=pre_spikes),
                    dt=self._dt,
                    t=self._t,
                    ctx=self._ctx,
                )
                self.last_projection_drive[proj.name] = syn_result.post_drive
                _accumulate_drive(self._drive_buffers[proj.post], syn_result.post_drive)

        if self._external_drive_fn is not None:
            for spec in self._pop_specs:
                extra = self._external_drive_fn(self._t, self._step, spec.name, self._ctx)
                _accumulate_drive(self._drive_buffers[spec.name], extra)

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
            spikes_view.copy_(spikes.to(device=spikes_view.device))
            self._pop_states[spec.name] = pop_state

            if not no_monitors:
                if self._fast_mode:
                    if self._compiled_event_tensors is not None:
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

        self.last_d_weights = {}
        for proj in self._proj_specs:
            if proj.learning is None:
                continue
            if proj.learn_every <= 0 or (self._step % proj.learn_every) != 0:
                continue
            proj_state = self._proj_states[proj.name]
            learn_state = proj_state.learning_state
            if learn_state is None:
                continue

            weights = _require_weights(proj_state.state, proj.name)
            supports_sparse = bool(getattr(proj.learning, "supports_sparse", False))
            use_sparse = bool(proj.sparse_learning and supports_sparse)
            if use_sparse:
                active_edges = _active_edges_from_spikes(
                    self._pop_states[proj.pre].spikes,
                    proj.topology,
                    device=self._device,
                    max_edges=_sparse_edge_cap(proj),
                )
                edge_pre_idx = proj.topology.pre_idx.index_select(0, active_edges)
                edge_post_idx = proj.topology.post_idx.index_select(0, active_edges)
                edge_pre = self._pop_states[proj.pre].spikes.index_select(0, edge_pre_idx)
                edge_post = self._pop_states[proj.post].spikes.index_select(0, edge_post_idx)
                edge_weights = weights.index_select(0, active_edges)
                mods = _subset_edge_mods(edge_mods_by_proj.get(proj.name), active_edges)
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
            else:
                edge_pre = self._pop_states[proj.pre].spikes[proj.topology.pre_idx]
                edge_post = self._pop_states[proj.post].spikes[proj.topology.post_idx]
                batch = LearningBatch(
                    pre_spikes=edge_pre,
                    post_spikes=edge_post,
                    weights=weights,
                    modulators=edge_mods_by_proj.get(proj.name),
                    extras={
                        "pre_idx": proj.topology.pre_idx,
                        "post_idx": proj.topology.post_idx,
                    },
                )
            new_state, res = proj.learning.step(
                learn_state,
                batch,
                dt=self._dt,
                t=self._t,
                ctx=self._ctx,
            )
            proj_state.learning_state = new_state
            if use_sparse:
                if res.d_weights.numel() == active_edges.numel():
                    weights.index_add_(0, active_edges, res.d_weights)
                elif res.d_weights.numel() == weights.numel():
                    weights.add_(res.d_weights)
                else:
                    raise ValueError(
                        f"Sparse learning for {proj.name} returned d_weights with unexpected shape."
                    )
            else:
                weights.add_(res.d_weights)
            _apply_weight_clamp(weights, proj)
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

        event = StepEvent(
            t=self._t,
            dt=self._dt,
            spikes=spikes_global if not self._fast_mode else None,
            tensors=self._compiled_event_tensors or {},
            scalars=cast(Mapping[str, Scalar], scalars),
            meta=self._compiled_meta,
        )
        self._last_event = event

        for monitor in self._monitors:
            monitor.on_step(event)

        self._t += self._dt
        self._step += 1

        return scalars

    def run(self, steps: int) -> None:
        try:
            for _ in range(steps):
                self.step()
        finally:
            for monitor in self._monitors:
                monitor.flush()
            for monitor in self._monitors:
                monitor.close()

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
        if self._fast_mode:
            fast_event_tensors: dict[str, Tensor] = {}
            for spec in self._pop_specs:
                pop_state = self._pop_states[spec.name]
                fast_event_tensors[f"pop/{spec.name}/spikes"] = pop_state.spikes
                tensors = spec.model.state_tensors(pop_state.state)
                self._compiled_pop_tensor_keys[spec.name] = tuple(tensors.keys())
                for key, value in tensors.items():
                    if key == "spikes":
                        continue
                    fast_event_tensors[f"pop/{spec.name}/{key}"] = value
            fast_event_tensors.update(_projection_tensors(self._proj_specs, self._proj_states))
            fast_event_tensors.update(_learning_tensors(self._proj_specs, self._proj_states))
            fast_event_tensors.update(_modulator_tensors(self._mod_specs, self._mod_states))
            self._compiled_event_tensors = fast_event_tensors
        else:
            keys: set[str] = set()
            pop_keys: dict[str, set[str]] = {}
            pop_tensors: dict[str, Mapping[str, Tensor]] = {}
            for spec in self._pop_specs:
                tensors = spec.model.state_tensors(self._pop_states[spec.name].state)
                pop_tensors[spec.name] = tensors
                pop_key_set = set(tensors.keys())
                pop_keys[spec.name] = pop_key_set
                keys.update(pop_key_set)
            compiled_event_tensors: dict[str, Tensor] = {}
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

            compiled_event_tensors.update(_projection_tensors(self._proj_specs, self._proj_states))
            compiled_event_tensors.update(_learning_tensors(self._proj_specs, self._proj_states))
            compiled_event_tensors.update(_modulator_tensors(self._mod_specs, self._mod_states))
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
        self._compiled_meta = {"population_slices": self._pop_slice_tuples}

        self._compiled_mod_by_pop = {spec.name: {} for spec in self._pop_specs}
        self._compiled_edge_mods_by_proj = {proj.name: {} for proj in self._proj_specs}


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


def _compile_flags_for_projection(proj: ProjectionSpec) -> tuple[bool, bool]:
    build_edges_by_delay = False
    build_pre_adjacency = False
    try:
        from biosnn.synapses.dynamics.delayed_current import DelayedCurrentSynapse
    except Exception:
        return build_edges_by_delay, build_pre_adjacency

    if isinstance(proj.synapse, DelayedCurrentSynapse):
        params = proj.synapse.params
        adaptive = bool(getattr(params, "adaptive_event_driven", False))
        build_pre_adjacency = bool(params.event_driven or adaptive)
        build_edges_by_delay = bool(
            (not params.use_edge_buffer) and (adaptive or not params.event_driven)
        )
    supports_sparse = bool(getattr(proj.learning, "supports_sparse", False))
    if proj.sparse_learning and supports_sparse:
        build_pre_adjacency = True
    return build_edges_by_delay, build_pre_adjacency


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
        return
    state_weights = state.weights
    if not hasattr(state_weights, "shape"):
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


def _gather_active_edges(active_pre: Tensor, pre_ptr: Tensor, edge_idx: Tensor) -> Tensor:
    torch = require_torch()
    starts = pre_ptr.index_select(0, active_pre)
    ends = pre_ptr.index_select(0, active_pre + 1)
    counts = ends - starts
    if counts.numel() == 0:
        return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))
    total = counts.sum()
    base = torch.repeat_interleave(starts, counts)
    prefix = torch.cumsum(counts, 0)
    group_start = torch.repeat_interleave(prefix - counts, counts)
    intra = torch.arange(total, device=edge_idx.device) - group_start
    edge_pos = base + intra
    return cast(Tensor, edge_idx.index_select(0, edge_pos))


def _active_edges_from_spikes(
    pre_spikes: Tensor,
    topology: SynapseTopology,
    *,
    device: Any,
    max_edges: int | None = None,
) -> Tensor:
    torch = require_torch()
    pre_ptr, edge_idx = _require_pre_adjacency(topology, device)
    active_pre = pre_spikes.nonzero(as_tuple=False).flatten()
    if active_pre.numel() == 0:
        return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))
    edges = _gather_active_edges(active_pre, pre_ptr, edge_idx)
    if max_edges is not None and edges.numel() > max_edges:
        return edges[:max_edges]
    return edges


def _ensure_topology_meta(
    topology: SynapseTopology,
    *,
    n_pre: int | None = None,
    n_post: int | None = None,
) -> None:
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
        object.__setattr__(topology, "meta", meta)


__all__ = ["TorchNetworkEngine"]
