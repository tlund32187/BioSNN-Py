"""Torch-backed multi-population simulation engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from biosnn.biophysics.models._torch_utils import require_torch, resolve_device_dtype
from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.monitors import IMonitor, StepEvent
from biosnn.contracts.neurons import Compartment, INeuronModel, NeuronInputs, StepContext
from biosnn.contracts.simulation import ISimulationEngine, SimulationConfig
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
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
    ) -> None:
        self._pop_specs = list(populations)
        self._proj_specs = list(projections)
        self._mod_specs = list(modulators) if modulators is not None else []
        self._external_drive_fn = external_drive_fn
        self._releases_fn = releases_fn

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

        self.last_projection_drive: dict[str, Mapping[Compartment, Tensor]] = {}
        self.last_d_weights: dict[str, Tensor] = {}

        self._pop_order = [spec.name for spec in self._pop_specs]
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
            spikes = torch.zeros((spec.n,), device=self._device, dtype=self._dtype)
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

        self._mod_states.clear()
        for mod in self._mod_specs:
            self._mod_states[mod.name] = mod.field.init_state(ctx=self._ctx)

        _apply_initial_spikes(self._pop_states, self._pop_order, config.meta)

    def attach_monitors(self, monitors: Sequence[IMonitor]) -> None:
        self._monitors = list(monitors)

    def step(self) -> Mapping[str, Any]:
        torch = require_torch()

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

        drive_acc: dict[str, dict[Compartment, Tensor]] = {name: {} for name in self._pop_order}

        self.last_projection_drive = {}
        for proj in self._proj_specs:
            proj_state = self._proj_states[proj.name]
            pre_spikes = self._pop_states[proj.pre].spikes
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
            if not drive:
                drive = _zero_drive(spec.model, spec.n, self._device, self._dtype)
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
            spikes = (result.spikes > 0).to(device=pop_state.spikes.device, dtype=pop_state.spikes.dtype)
            pop_state.spikes = spikes
            self._pop_states[spec.name] = pop_state

            start = offset
            end = offset + spec.n
            pop_slices[spec.name] = (start, end)
            offset = end

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

        spikes_global = torch.cat(spikes_concat) if spikes_concat else torch.zeros((0,), device=self._device)
        spike_count = float(spikes_global.sum().item()) if spikes_global.numel() else 0.0
        spike_fraction = float(spikes_global.mean().item()) if spikes_global.numel() else 0.0

        event_tensors = _merge_population_tensors(
            neuron_tensors_by_pop,
            self._pop_specs,
            self._device,
        )
        event_tensors.update(_projection_tensors(self._proj_specs, self._proj_states))
        event_tensors.update(_learning_tensors(self._proj_specs, self._proj_states))
        event_tensors.update(_modulator_tensors(self._mod_specs, self._mod_states))

        scalars = {
            "step": float(self._step),
            "t": float(self._t),
            "spike_count_total": spike_count,
            "spike_fraction_total": spike_fraction,
        }
        for spec in self._pop_specs:
            count = float(self._pop_states[spec.name].spikes.sum().item())
            scalars[f"spike_count/{spec.name}"] = count

        event = StepEvent(
            t=self._t,
            dt=self._dt,
            spikes=spikes_global,
            tensors=event_tensors,
            scalars=scalars,
            meta={"population_slices": pop_slices},
        )

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
        spikes[idx_tensor] = 1.0
        return

    if isinstance(indices, (list, tuple)):
        for idx in indices:
            spikes[int(idx)] = 1.0
        return

    spikes[int(indices)] = 1.0


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
        if comp in target:
            target[comp] = target[comp] + tensor
        else:
            target[comp] = tensor


def _zero_drive(model: INeuronModel, n: int, device: Any, dtype: Any) -> dict[Compartment, Tensor]:
    torch = require_torch()
    return {comp: torch.zeros((n,), device=device, dtype=dtype) for comp in model.compartments}


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


__all__ = ["TorchNetworkEngine"]
