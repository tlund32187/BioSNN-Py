"""Delayed current synapse model with per-edge delays and ring buffer."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, cast

from biosnn.contracts.neurons import Compartment, StepContext, Tensor
from biosnn.contracts.synapses import (
    ISynapseModel,
    ReceptorKind,
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)
from biosnn.core.torch_utils import _resolve_dtype, require_torch, resolve_device_dtype
from biosnn.synapses.buffers.event_list_ring import EventListRing

_COMPARTMENT_ORDER = tuple(Compartment)
_COMPARTMENT_TO_ID = {comp: idx for idx, comp in enumerate(_COMPARTMENT_ORDER)}


@dataclass(frozen=True, slots=True)
class DelayedCurrentParams:
    """Simple delayed-current synapse parameters."""

    init_weight: float = 1e-9
    clamp_min: float | None = None
    clamp_max: float | None = None
    receptor_scale: Mapping[ReceptorKind, float] | None = None
    ring_dtype: str | None = None
    ring_strategy: Literal["dense", "event_list_proto", "event_bucketed"] = "dense"
    event_list_max_events: int = 1_000_000
    use_edge_buffer: bool = False
    event_driven: bool = False
    adaptive_event_driven: bool = False
    adaptive_threshold: float = 0.02


@dataclass(slots=True)
class DelayedCurrentState:
    """Synapse state (weights + optional delay buffer)."""

    weights: Tensor  # shape [E]
    delay_buffer: Tensor | None  # shape [D, E] (legacy)
    delay_tmp: Tensor | None  # shape [E] scratch for edge buffer
    post_ring: dict[Compartment, Tensor] | None  # shape [D, Npost] per compartment
    post_out: dict[Compartment, Tensor] | None  # reusable output buffers [Npost] per compartment
    post_event_ring: dict[Compartment, EventListRing] | None  # sparse event ring per compartment
    cursor: int


class DelayedCurrentSynapse(ISynapseModel):
    """Edge-population synapse model with integer step delays."""

    name = "delayed_current"

    def __init__(self, params: DelayedCurrentParams | None = None) -> None:
        self.params = params or DelayedCurrentParams()
        self._scale_cache: dict[tuple[object, object, tuple[ReceptorKind, ...]], Tensor] = {}

    def init_state(self, e: int, *, ctx: StepContext) -> DelayedCurrentState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        weights = torch.full((e,), self.params.init_weight, device=device, dtype=dtype)
        return DelayedCurrentState(
            weights=weights,
            delay_buffer=None,
            delay_tmp=None,
            post_ring=None,
            post_out=None,
            post_event_ring=None,
            cursor=0,
        )

    def reset_state(
        self,
        state: DelayedCurrentState,
        *,
        ctx: StepContext,
        edge_indices: Tensor | None = None,
    ) -> DelayedCurrentState:
        _ = ctx
        if edge_indices is None:
            state.weights.fill_(self.params.init_weight)
            if state.delay_buffer is not None:
                state.delay_buffer.zero_()
            if state.delay_tmp is not None:
                state.delay_tmp.zero_()
            if state.post_ring is not None:
                for ring in state.post_ring.values():
                    ring.zero_()
            if state.post_out is not None:
                for out in state.post_out.values():
                    out.zero_()
            if state.post_event_ring is not None:
                for event_list in state.post_event_ring.values():
                    event_list.clear()
            state.cursor = 0
            return state
        state.weights[edge_indices] = self.params.init_weight
        if state.delay_buffer is not None:
            state.delay_buffer[:, edge_indices] = 0.0
        if state.delay_tmp is not None:
            state.delay_tmp.zero_()
        if state.post_ring is not None:
            for ring in state.post_ring.values():
                ring[:, :] = 0.0
        if state.post_out is not None:
            for out in state.post_out.values():
                out[:] = 0.0
        if state.post_event_ring is not None:
            for event_list in state.post_event_ring.values():
                event_list.clear()
        return state

    def step(
        self,
        state: DelayedCurrentState,
        topology: SynapseTopology,
        inputs: SynapseInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[DelayedCurrentState, SynapseStepResult]:
        torch = require_torch()
        use_no_grad = _use_no_grad(ctx)
        with torch.no_grad() if use_no_grad else nullcontext():
            if (
                self.params.ring_strategy in {"event_list_proto", "event_bucketed"}
                and not (self.params.event_driven or self.params.adaptive_event_driven)
            ):
                raise RuntimeError(
                    "ring_strategy=event_bucketed requires event_driven or adaptive_event_driven."
                )
            weights = state.weights
            device = weights.device
            validate_shapes = _validate_shapes(ctx)
            require_on_device = _require_inputs_on_device(ctx)

            pre_spikes = _as_like(
                inputs.pre_spikes,
                weights,
                name="pre_spikes",
                validate_shapes=validate_shapes,
                expected_len=_infer_n_pre(topology),
                require_on_device=require_on_device,
            )

            pre_idx = _as_index(topology.pre_idx, device, name="pre_idx")
            post_idx = _as_index(topology.post_idx, device, name="post_idx")
            if self.params.use_edge_buffer:
                state, post_drive = _step_edge_buffer(
                    self,
                    state,
                    topology,
                    pre_spikes,
                    pre_idx,
                    post_idx,
                    device,
                    weights,
                )
                return state, SynapseStepResult(post_drive=post_drive)

            if self.params.adaptive_event_driven:
                spike_fraction = pre_spikes.float().mean()
                active_pre = pre_spikes.nonzero(as_tuple=False).flatten()
                use_event = active_pre.numel() < (
                    self.params.adaptive_threshold * pre_spikes.numel()
                )
                if use_event:
                    state, post_drive, extras = _step_post_ring_event_driven(
                        self,
                        state,
                        topology,
                        pre_spikes,
                        pre_idx,
                        post_idx,
                        device,
                        weights,
                        active_pre=active_pre,
                    )
                    extras = dict(extras)
                    extras["spike_fraction"] = spike_fraction
                    return state, SynapseStepResult(post_drive=post_drive, extras=extras)

                state, post_drive = _step_post_ring(
                    self,
                    state,
                    topology,
                    pre_spikes,
                    pre_idx,
                    post_idx,
                    device,
                    weights,
                )
                return state, SynapseStepResult(post_drive=post_drive)

            if self.params.event_driven:
                state, post_drive, extras = _step_post_ring_event_driven(
                    self,
                    state,
                    topology,
                    pre_spikes,
                    pre_idx,
                    post_idx,
                    device,
                    weights,
                )
                return state, SynapseStepResult(post_drive=post_drive, extras=extras)

            state, post_drive = _step_post_ring(
                self,
                state,
                topology,
                pre_spikes,
                pre_idx,
                post_idx,
                device,
                weights,
            )

            return state, SynapseStepResult(post_drive=post_drive)

    def step_into(
        self,
        state: DelayedCurrentState,
        pre_spikes: Tensor,
        out_drive: Mapping[Compartment, Tensor],
        t: int,
        **kwargs: Any,
    ) -> None:
        topology = kwargs.get("topology")
        ctx = kwargs.get("ctx")
        dt = kwargs.get("dt")
        if topology is None or ctx is None or dt is None:
            raise ValueError("step_into requires topology, dt, and ctx keyword arguments")

        torch = require_torch()
        use_no_grad = _use_no_grad(ctx)
        with torch.no_grad() if use_no_grad else nullcontext():
            weights = state.weights
            device = weights.device
            validate_shapes = _validate_shapes(ctx)
            require_on_device = _require_inputs_on_device(ctx)

            pre_spikes = _as_like(
                pre_spikes,
                weights,
                name="pre_spikes",
                validate_shapes=validate_shapes,
                expected_len=_infer_n_pre(topology),
                require_on_device=require_on_device,
            )

            pre_idx = _as_index(topology.pre_idx, device, name="pre_idx")
            post_idx = _as_index(topology.post_idx, device, name="post_idx")
            if self.params.use_edge_buffer:
                state, post_drive = _step_edge_buffer(
                    self,
                    state,
                    topology,
                    pre_spikes,
                    pre_idx,
                    post_idx,
                    device,
                    weights,
                )
                for comp, drive in post_drive.items():
                    out_drive[comp].add_(drive)
                return

            if self.params.adaptive_event_driven:
                _ = pre_spikes.float().mean()
                active_pre = pre_spikes.nonzero(as_tuple=False).flatten()
                use_event = active_pre.numel() < (
                    self.params.adaptive_threshold * pre_spikes.numel()
                )
                if use_event:
                    _step_post_ring_event_driven_into(
                        self,
                        state,
                        topology,
                        pre_spikes,
                        pre_idx,
                        post_idx,
                        device,
                        weights,
                        out_drive,
                        active_pre=active_pre,
                    )
                    return
                _step_post_ring_into(
                    self,
                    state,
                    topology,
                    pre_spikes,
                    pre_idx,
                    post_idx,
                    device,
                    weights,
                    out_drive,
                )
                return

            if self.params.event_driven:
                _step_post_ring_event_driven_into(
                    self,
                    state,
                    topology,
                    pre_spikes,
                    pre_idx,
                    post_idx,
                    device,
                    weights,
                    out_drive,
                )
                return

            _step_post_ring_into(
                self,
                state,
                topology,
                pre_spikes,
                pre_idx,
                post_idx,
                device,
                weights,
                out_drive,
            )

    def state_tensors(self, state: DelayedCurrentState) -> Mapping[str, Tensor]:
        tensors: dict[str, Tensor] = {"weights": state.weights}
        if state.delay_buffer is not None:
            tensors["delay_buffer"] = state.delay_buffer
        return tensors

    def compilation_requirements(self) -> Mapping[str, bool]:
        needs_pre_adjacency = bool(self.params.event_driven or self.params.adaptive_event_driven)
        needs_edges_by_delay = bool(
            (not self.params.use_edge_buffer)
            and (self.params.adaptive_event_driven or not self.params.event_driven)
        )
        return {
            "needs_edges_by_delay": needs_edges_by_delay,
            "needs_pre_adjacency": needs_pre_adjacency,
            "needs_sparse_delay_mats": False,
            "needs_bucket_edge_mapping": False,
            "wants_fused_sparse": False,
            "wants_by_delay_sparse": False,
            "wants_bucket_edge_mapping": False,
            "wants_weights_snapshot_each_step": False,
            "wants_projection_drive_tensor": False,
        }

    def _scale_table(self, like: Tensor, kinds: tuple[ReceptorKind, ...]) -> Tensor:
        key = (like.device, like.dtype, kinds)
        cached = self._scale_cache.get(key)
        if cached is not None:
            return cached
        torch = require_torch()
        scale_map = self.params.receptor_scale or {}
        values = [scale_map.get(kind, 1.0) for kind in kinds]
        table = cast(Tensor, torch.tensor(values, device=like.device, dtype=like.dtype))
        self._scale_cache[key] = table
        return table


def _apply_receptor_scale(
    model: DelayedCurrentSynapse,
    topology: SynapseTopology,
    edge_current: Tensor,
) -> Tensor:
    if topology.receptor is None:
        return edge_current
    torch = require_torch()
    receptor_ids = topology.receptor
    if receptor_ids.device != edge_current.device or receptor_ids.dtype != torch.long:
        raise ValueError("receptor ids must be on the same device with dtype long")
    kinds = topology.receptor_kinds or (ReceptorKind.AMPA, ReceptorKind.NMDA, ReceptorKind.GABA)
    table = model._scale_table(edge_current, kinds)
    edge_scale = table.index_select(0, receptor_ids)
    return edge_current * edge_scale


def _apply_receptor_scale_edges(
    model: DelayedCurrentSynapse,
    topology: SynapseTopology,
    edges: Tensor,
    edge_current: Tensor,
) -> Tensor:
    if topology.receptor is None:
        return edge_current
    torch = require_torch()
    receptor_ids = topology.receptor
    if receptor_ids.device != edge_current.device or receptor_ids.dtype != torch.long:
        raise ValueError("receptor ids must be on the same device with dtype long")
    receptor_ids = receptor_ids.index_select(0, edges)
    kinds = topology.receptor_kinds or (ReceptorKind.AMPA, ReceptorKind.NMDA, ReceptorKind.GABA)
    table = model._scale_table(edge_current, kinds)
    edge_scale = table.index_select(0, receptor_ids)
    return edge_current * edge_scale


def _accumulate_post_drive(
    edge_current: Tensor,
    post_idx: Tensor,
    topology: SynapseTopology,
    device: Any,
    dtype: Any,
) -> Mapping[Compartment, Tensor]:
    torch = require_torch()
    n_post = _infer_n_post(topology)
    post_drive: dict[Compartment, Tensor] = {}

    target_compartments = topology.target_compartments
    if target_compartments is None:
        comp = topology.target_compartment
        out = torch.zeros((n_post,), device=device, dtype=dtype)
        if post_idx.numel():
            out.scatter_add_(0, post_idx, edge_current)
        post_drive[comp] = out
        return post_drive

    comp_ids = target_compartments
    if comp_ids.device != device or comp_ids.dtype != require_torch().long:
        raise ValueError("target_compartments must be on the target device with dtype long")
    for comp, comp_id in _COMPARTMENT_TO_ID.items():
        mask = comp_ids == comp_id
        selected = mask.nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        out = torch.zeros((n_post,), device=device, dtype=dtype)
        out.scatter_add_(
            0,
            post_idx.index_select(0, selected),
            edge_current.index_select(0, selected),
        )
        post_drive[comp] = out
    return post_drive


def _step_edge_buffer(
    model: DelayedCurrentSynapse,
    state: DelayedCurrentState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    pre_idx: Tensor,
    post_idx: Tensor,
    device: Any,
    weights: Tensor,
) -> tuple[DelayedCurrentState, Mapping[Compartment, Tensor]]:
    torch = require_torch()
    edge_spikes = pre_spikes.index_select(0, pre_idx)
    edge_active_f = (
        edge_spikes
        if edge_spikes.dtype == weights.dtype
        else edge_spikes.to(dtype=weights.dtype)
    )

    delay_steps_opt = topology.delay_steps
    max_delay = _max_delay_steps(topology)
    if delay_steps_opt is not None:
        delay_steps = _as_index(delay_steps_opt, device, name="delay_steps")
    else:
        delay_steps = None
        max_delay = 0

    if max_delay > 0 and delay_steps is not None:
        depth = max_delay + 1
        if state.delay_buffer is None or state.delay_buffer.shape != (depth, pre_idx.numel()):
            state.delay_buffer = torch.zeros((depth, pre_idx.numel()), device=device, dtype=weights.dtype)
            state.cursor = 0
        if (
            state.delay_tmp is None
            or state.delay_tmp.shape != (pre_idx.numel(),)
            or state.delay_tmp.device != device
            or state.delay_tmp.dtype != weights.dtype
        ):
            state.delay_tmp = torch.empty((pre_idx.numel(),), device=device, dtype=weights.dtype)
        cursor = state.cursor % depth
        edge_idx = torch.arange(pre_idx.numel(), device=device)
        delayed = state.delay_tmp
        delayed.copy_(state.delay_buffer[cursor])
        state.delay_buffer[cursor].zero_()
        immediate_mask = cast(Tensor, delay_steps == 0)
        delayed = delayed + edge_active_f * immediate_mask.to(edge_active_f.dtype)
        delayed_idx = (~immediate_mask).nonzero(as_tuple=False).flatten()
        if delayed_idx.numel() > 0:
            target_rows = torch.remainder(
                delay_steps.index_select(0, delayed_idx) + cursor, depth
            )
            state.delay_buffer[
                target_rows, edge_idx.index_select(0, delayed_idx)
            ] = edge_active_f.index_select(0, delayed_idx)
        state.cursor = (cursor + 1) % depth
    else:
        delayed = edge_active_f

    weights = _maybe_clamp_weights(model, state, weights)
    edge_current = delayed * weights
    edge_current = _apply_receptor_scale(model, topology, edge_current)

    post_drive = _accumulate_post_drive(
        edge_current,
        post_idx,
        topology,
        device,
        weights.dtype,
    )
    return state, post_drive


def _step_post_ring(
    model: DelayedCurrentSynapse,
    state: DelayedCurrentState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    pre_idx: Tensor,
    post_idx: Tensor,
    device: Any,
    weights: Tensor,
) -> tuple[DelayedCurrentState, Mapping[Compartment, Tensor]]:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
    edges_by_delay = _edges_by_delay(topology, device)
    ring_dtype = _resolve_ring_dtype(model.params.ring_dtype, weights.dtype, device)
    post_ring, post_out = _ensure_post_ring(
        state,
        topology,
        depth,
        n_post,
        device,
        ring_dtype,
        weights.dtype,
    )

    cursor = state.cursor % depth
    post_drive: dict[Compartment, Tensor] = {}
    for comp, ring in post_ring.items():
        slot = ring[cursor]
        out = post_out[comp]
        _copy_ring_slot(slot, out)
        post_drive[comp] = out

    weights = _maybe_clamp_weights(model, state, weights)

    for delay, edges in enumerate(edges_by_delay):
        if edges.numel() == 0:
            continue
        edge_pre = pre_idx.index_select(0, edges)
        edge_post = post_idx.index_select(0, edges)
        edge_spikes = pre_spikes.index_select(0, edge_pre)
        edge_weights = weights.index_select(0, edges)
        edge_active_f = (
            edge_spikes
            if edge_spikes.dtype == edge_weights.dtype
            else edge_spikes.to(dtype=edge_weights.dtype)
        )
        edge_current = edge_active_f * edge_weights
        edge_current = _apply_receptor_scale_edges(model, topology, edges, edge_current)

        if topology.target_compartments is None:
            comp = topology.target_compartment
            if delay == 0:
                post_drive[comp].index_add_(0, edge_post, edge_current)
            else:
                target_row = (cursor + delay) % depth
                post_ring[comp][target_row].index_add_(
                    0,
                    edge_post,
                    _as_ring_dtype(edge_current, ring_dtype),
                )
            continue

        comp_ids = topology.target_compartments
        if comp_ids.device != device or comp_ids.dtype != torch.long:
            raise ValueError("target_compartments must be on the target device with dtype long")
        comp_ids = comp_ids.index_select(0, edges)
        for comp in post_drive:
            comp_id = _COMPARTMENT_TO_ID[comp]
            mask = comp_ids == comp_id
            selected = mask.nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            posts = edge_post.index_select(0, selected)
            currents = edge_current.index_select(0, selected)
            if delay == 0:
                post_drive[comp].index_add_(0, posts, currents)
            else:
                target_row = (cursor + delay) % depth
                post_ring[comp][target_row].index_add_(
                    0,
                    posts,
                    _as_ring_dtype(currents, ring_dtype),
                )

    state.cursor = (cursor + 1) % depth
    return state, post_drive


def _step_post_ring_into(
    model: DelayedCurrentSynapse,
    state: DelayedCurrentState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    pre_idx: Tensor,
    post_idx: Tensor,
    device: Any,
    weights: Tensor,
    out_drive: Mapping[Compartment, Tensor],
) -> None:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
    edges_by_delay = _edges_by_delay(topology, device)
    ring_dtype = _resolve_ring_dtype(model.params.ring_dtype, weights.dtype, device)
    post_ring, _ = _ensure_post_ring(
        state,
        topology,
        depth,
        n_post,
        device,
        ring_dtype,
        weights.dtype,
    )

    cursor = state.cursor % depth
    for comp, ring in post_ring.items():
        slot = ring[cursor]
        if comp not in out_drive:
            raise KeyError(f"Drive accumulator missing compartment {comp}")
        _add_ring_slot(slot, out_drive[comp])

    weights = _maybe_clamp_weights(model, state, weights)

    for delay, edges in enumerate(edges_by_delay):
        if edges.numel() == 0:
            continue
        edge_pre = pre_idx.index_select(0, edges)
        edge_post = post_idx.index_select(0, edges)
        edge_spikes = pre_spikes.index_select(0, edge_pre)
        edge_weights = weights.index_select(0, edges)
        edge_active_f = (
            edge_spikes
            if edge_spikes.dtype == edge_weights.dtype
            else edge_spikes.to(dtype=edge_weights.dtype)
        )
        edge_current = edge_active_f * edge_weights
        edge_current = _apply_receptor_scale_edges(model, topology, edges, edge_current)

        if topology.target_compartments is None:
            comp = topology.target_compartment
            if comp not in out_drive:
                raise KeyError(f"Drive accumulator missing compartment {comp}")
            if delay == 0:
                out_drive[comp].index_add_(0, edge_post, edge_current)
            else:
                target_row = (cursor + delay) % depth
                post_ring[comp][target_row].index_add_(
                    0,
                    edge_post,
                    _as_ring_dtype(edge_current, ring_dtype),
                )
            continue

        comp_ids = topology.target_compartments
        if comp_ids.device != device or comp_ids.dtype != torch.long:
            raise ValueError("target_compartments must be on the target device with dtype long")
        comp_ids = comp_ids.index_select(0, edges)
        for comp in out_drive:
            comp_id = _COMPARTMENT_TO_ID[comp]
            mask = comp_ids == comp_id
            selected = mask.nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            posts = edge_post.index_select(0, selected)
            currents = edge_current.index_select(0, selected)
            if delay == 0:
                out_drive[comp].index_add_(0, posts, currents)
            else:
                target_row = (cursor + delay) % depth
                post_ring[comp][target_row].index_add_(
                    0,
                    posts,
                    _as_ring_dtype(currents, ring_dtype),
                )

    state.cursor = (cursor + 1) % depth


def _step_post_ring_event_driven(
    model: DelayedCurrentSynapse,
    state: DelayedCurrentState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    pre_idx: Tensor,
    post_idx: Tensor,
    device: Any,
    weights: Tensor,
    *,
    active_pre: Tensor | None = None,
) -> tuple[DelayedCurrentState, Mapping[Compartment, Tensor], Mapping[str, Tensor]]:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
    ring_dtype = _resolve_ring_dtype(model.params.ring_dtype, weights.dtype, device)
    use_event_list = model.params.ring_strategy in {"event_list_proto", "event_bucketed"}
    event_ring: dict[Compartment, EventListRing] | None = None
    if use_event_list:
        post_out = _ensure_post_out(state, topology, n_post, device, weights.dtype)
        event_ring = _ensure_event_ring(
            state,
            topology,
            depth,
            device,
            ring_dtype,
            max_events=model.params.event_list_max_events,
        )
    else:
        post_ring, post_out = _ensure_post_ring(
            state,
            topology,
            depth,
            n_post,
            device,
            ring_dtype,
            weights.dtype,
        )

    cursor = state.cursor % depth
    post_drive: dict[Compartment, Tensor] = {}
    if use_event_list:
        assert event_ring is not None
        for comp, event_list in event_ring.items():
            out = post_out[comp]
            out.zero_()
            event_list.pop_into(cursor, out)
            post_drive[comp] = out
    else:
        for comp, ring in post_ring.items():
            slot = ring[cursor]
            out = post_out[comp]
            _copy_ring_slot(slot, out)
            post_drive[comp] = out

    weights = _maybe_clamp_weights(model, state, weights)

    if active_pre is None:
        active_pre = pre_spikes.nonzero(as_tuple=False).flatten()
    if active_pre.numel() == 0:
        state.cursor = (cursor + 1) % depth
        extras = {"processed_edges": weights.new_zeros(())}
        return state, post_drive, extras

    pre_ptr, edge_idx = _require_pre_adjacency(topology, device)
    edges = _gather_active_edges(active_pre, pre_ptr, edge_idx, topology)
    if edges.numel() == 0:
        state.cursor = (cursor + 1) % depth
        extras = {"processed_edges": weights.new_zeros(())}
        return state, post_drive, extras

    edge_pre = pre_idx.index_select(0, edges)
    edge_post = post_idx.index_select(0, edges)
    edge_spikes = pre_spikes.index_select(0, edge_pre)
    edge_weights = weights.index_select(0, edges)
    edge_active_f = (
        edge_spikes
        if edge_spikes.dtype == edge_weights.dtype
        else edge_spikes.to(dtype=edge_weights.dtype)
    )
    edge_current = edge_active_f * edge_weights
    edge_current = _apply_receptor_scale_edges(model, topology, edges, edge_current)

    delay_steps = topology.delay_steps
    delay_vals = None if delay_steps is None else delay_steps.index_select(0, edges)

    if topology.target_compartments is None:
        comp = topology.target_compartment
        if delay_vals is None:
            post_drive[comp].index_add_(0, edge_post, edge_current)
        else:
            immediate_mask = delay_vals == 0
            immediate_idx = immediate_mask.nonzero(as_tuple=False).flatten()
            if immediate_idx.numel() > 0:
                post_drive[comp].index_add_(
                    0,
                    edge_post.index_select(0, immediate_idx),
                    edge_current.index_select(0, immediate_idx),
                )
            delayed_idx = (~immediate_mask).nonzero(as_tuple=False).flatten()
            if delayed_idx.numel() > 0:
                target_rows = torch.remainder(
                    delay_vals.index_select(0, delayed_idx) + cursor, depth
                )
                delayed_posts = edge_post.index_select(0, delayed_idx)
                delayed_vals = edge_current.index_select(0, delayed_idx)
                if use_event_list:
                    assert event_ring is not None
                    event_ring[comp].schedule(
                        target_rows,
                        delayed_posts,
                        _as_ring_dtype(delayed_vals, ring_dtype),
                    )
                else:
                    post_ring[comp].index_put_(
                        (target_rows, delayed_posts),
                        _as_ring_dtype(delayed_vals, ring_dtype),
                        accumulate=True,
                    )
    else:
        comp_ids = topology.target_compartments
        if comp_ids.device != device or comp_ids.dtype != torch.long:
            raise ValueError("target_compartments must be on the target device with dtype long")
        comp_ids = comp_ids.index_select(0, edges)
        for comp in post_drive:
            comp_id = _COMPARTMENT_TO_ID[comp]
            comp_mask = comp_ids == comp_id
            comp_idx = comp_mask.nonzero(as_tuple=False).flatten()
            if comp_idx.numel() == 0:
                continue
            comp_posts = edge_post.index_select(0, comp_idx)
            comp_currents = edge_current.index_select(0, comp_idx)
            if delay_vals is None:
                post_drive[comp].index_add_(0, comp_posts, comp_currents)
                continue
            comp_delays = delay_vals.index_select(0, comp_idx)
            immediate_mask = comp_delays == 0
            immediate_idx = immediate_mask.nonzero(as_tuple=False).flatten()
            if immediate_idx.numel() > 0:
                post_drive[comp].index_add_(
                    0,
                    comp_posts.index_select(0, immediate_idx),
                    comp_currents.index_select(0, immediate_idx),
                )
            delayed_idx = (~immediate_mask).nonzero(as_tuple=False).flatten()
            if delayed_idx.numel() > 0:
                target_rows = torch.remainder(
                    comp_delays.index_select(0, delayed_idx) + cursor, depth
                )
                delayed_posts = comp_posts.index_select(0, delayed_idx)
                delayed_vals = comp_currents.index_select(0, delayed_idx)
                if use_event_list:
                    assert event_ring is not None
                    event_ring[comp].schedule(
                        target_rows,
                        delayed_posts,
                        _as_ring_dtype(delayed_vals, ring_dtype),
                    )
                else:
                    post_ring[comp].index_put_(
                        (target_rows, delayed_posts),
                        _as_ring_dtype(delayed_vals, ring_dtype),
                        accumulate=True,
                    )

    state.cursor = (cursor + 1) % depth
    extras = {"processed_edges": weights.new_tensor(edges.numel())}
    return state, post_drive, extras


def _step_post_ring_event_driven_into(
    model: DelayedCurrentSynapse,
    state: DelayedCurrentState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    pre_idx: Tensor,
    post_idx: Tensor,
    device: Any,
    weights: Tensor,
    out_drive: Mapping[Compartment, Tensor],
    *,
    active_pre: Tensor | None = None,
) -> None:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
    ring_dtype = _resolve_ring_dtype(model.params.ring_dtype, weights.dtype, device)
    use_event_list = model.params.ring_strategy in {"event_list_proto", "event_bucketed"}
    event_ring: dict[Compartment, EventListRing] | None = None
    if use_event_list:
        event_ring = _ensure_event_ring(
            state,
            topology,
            depth,
            device,
            ring_dtype,
            max_events=model.params.event_list_max_events,
        )
    else:
        post_ring, _ = _ensure_post_ring(
            state,
            topology,
            depth,
            n_post,
            device,
            ring_dtype,
            weights.dtype,
        )

    cursor = state.cursor % depth
    if use_event_list:
        assert event_ring is not None
        for comp, event_list in event_ring.items():
            if comp not in out_drive:
                raise KeyError(f"Drive accumulator missing compartment {comp}")
            event_list.pop_into(cursor, out_drive[comp])
    else:
        for comp, ring in post_ring.items():
            slot = ring[cursor]
            if comp not in out_drive:
                raise KeyError(f"Drive accumulator missing compartment {comp}")
            _add_ring_slot(slot, out_drive[comp])

    weights = _maybe_clamp_weights(model, state, weights)

    if active_pre is None:
        active_pre = pre_spikes.nonzero(as_tuple=False).flatten()
    if active_pre.numel() == 0:
        state.cursor = (cursor + 1) % depth
        return

    pre_ptr, edge_idx = _require_pre_adjacency(topology, device)
    edges = _gather_active_edges(active_pre, pre_ptr, edge_idx, topology)
    if edges.numel() == 0:
        state.cursor = (cursor + 1) % depth
        return

    edge_pre = pre_idx.index_select(0, edges)
    edge_post = post_idx.index_select(0, edges)
    edge_spikes = pre_spikes.index_select(0, edge_pre)
    edge_weights = weights.index_select(0, edges)
    edge_active_f = (
        edge_spikes
        if edge_spikes.dtype == edge_weights.dtype
        else edge_spikes.to(dtype=edge_weights.dtype)
    )
    edge_current = edge_active_f * edge_weights
    edge_current = _apply_receptor_scale_edges(model, topology, edges, edge_current)

    delay_steps = topology.delay_steps
    delay_vals = None if delay_steps is None else delay_steps.index_select(0, edges)

    if topology.target_compartments is None:
        comp = topology.target_compartment
        if comp not in out_drive:
            raise KeyError(f"Drive accumulator missing compartment {comp}")
        if delay_vals is None:
            out_drive[comp].index_add_(0, edge_post, edge_current)
        else:
            immediate_mask = delay_vals == 0
            immediate_idx = immediate_mask.nonzero(as_tuple=False).flatten()
            if immediate_idx.numel() > 0:
                out_drive[comp].index_add_(
                    0,
                    edge_post.index_select(0, immediate_idx),
                    edge_current.index_select(0, immediate_idx),
                )
            delayed_idx = (~immediate_mask).nonzero(as_tuple=False).flatten()
            if delayed_idx.numel() > 0:
                target_rows = torch.remainder(
                    delay_vals.index_select(0, delayed_idx) + cursor, depth
                )
                delayed_posts = edge_post.index_select(0, delayed_idx)
                delayed_vals = edge_current.index_select(0, delayed_idx)
                if use_event_list:
                    assert event_ring is not None
                    event_ring[comp].schedule(
                        target_rows,
                        delayed_posts,
                        _as_ring_dtype(delayed_vals, ring_dtype),
                    )
                else:
                    post_ring[comp].index_put_(
                        (target_rows, delayed_posts),
                        _as_ring_dtype(delayed_vals, ring_dtype),
                        accumulate=True,
                    )
    else:
        comp_ids = topology.target_compartments
        if comp_ids.device != device or comp_ids.dtype != torch.long:
            raise ValueError("target_compartments must be on the target device with dtype long")
        comp_ids = comp_ids.index_select(0, edges)
        for comp in out_drive:
            comp_id = _COMPARTMENT_TO_ID[comp]
            comp_mask = comp_ids == comp_id
            comp_idx = comp_mask.nonzero(as_tuple=False).flatten()
            if comp_idx.numel() == 0:
                continue
            comp_posts = edge_post.index_select(0, comp_idx)
            comp_currents = edge_current.index_select(0, comp_idx)
            if delay_vals is None:
                out_drive[comp].index_add_(0, comp_posts, comp_currents)
                continue
            comp_delays = delay_vals.index_select(0, comp_idx)
            immediate_mask = comp_delays == 0
            immediate_idx = immediate_mask.nonzero(as_tuple=False).flatten()
            if immediate_idx.numel() > 0:
                out_drive[comp].index_add_(
                    0,
                    comp_posts.index_select(0, immediate_idx),
                    comp_currents.index_select(0, immediate_idx),
                )
            delayed_idx = (~immediate_mask).nonzero(as_tuple=False).flatten()
            if delayed_idx.numel() > 0:
                target_rows = torch.remainder(
                    comp_delays.index_select(0, delayed_idx) + cursor, depth
                )
                delayed_posts = comp_posts.index_select(0, delayed_idx)
                delayed_vals = comp_currents.index_select(0, delayed_idx)
                if use_event_list:
                    assert event_ring is not None
                    event_ring[comp].schedule(
                        target_rows,
                        delayed_posts,
                        _as_ring_dtype(delayed_vals, ring_dtype),
                    )
                else:
                    post_ring[comp].index_put_(
                        (target_rows, delayed_posts),
                        _as_ring_dtype(delayed_vals, ring_dtype),
                        accumulate=True,
                    )

    state.cursor = (cursor + 1) % depth


def _ensure_post_ring(
    state: DelayedCurrentState,
    topology: SynapseTopology,
    depth: int,
    n_post: int,
    device: Any,
    ring_dtype: Any,
    out_dtype: Any,
) -> tuple[dict[Compartment, Tensor], dict[Compartment, Tensor]]:
    torch = require_torch()
    if state.post_ring is None:
        state.post_ring = {}
    if state.post_out is None:
        state.post_out = {}
    for comp in _ring_compartments(topology):
        ring = state.post_ring.get(comp)
        if (
            ring is None
            or ring.shape != (depth, n_post)
            or ring.device != device
            or ring.dtype != ring_dtype
        ):
            try:
                ring = torch.zeros((depth, n_post), device=device, dtype=ring_dtype)
            except Exception as exc:  # pragma: no cover - device/dtype dependent
                raise RuntimeError(
                    f"ring_dtype={ring_dtype} is not supported on device {device} for delay rings; "
                    "use ring_dtype=None/float32 or run on CUDA."
                ) from exc
            state.post_ring[comp] = ring
        out = state.post_out.get(comp)
        if out is None or out.shape != (n_post,) or out.device != device or out.dtype != out_dtype:
            out = torch.zeros((n_post,), device=device, dtype=out_dtype)
            state.post_out[comp] = out
    return state.post_ring, state.post_out


def _ensure_post_out(
    state: DelayedCurrentState,
    topology: SynapseTopology,
    n_post: int,
    device: Any,
    out_dtype: Any,
) -> dict[Compartment, Tensor]:
    torch = require_torch()
    if state.post_out is None:
        state.post_out = {}
    for comp in _ring_compartments(topology):
        out = state.post_out.get(comp)
        if out is None or out.shape != (n_post,) or out.device != device or out.dtype != out_dtype:
            out = torch.zeros((n_post,), device=device, dtype=out_dtype)
            state.post_out[comp] = out
    return state.post_out


def _ensure_event_ring(
    state: DelayedCurrentState,
    topology: SynapseTopology,
    depth: int,
    device: Any,
    ring_dtype: Any,
    *,
    max_events: int,
) -> dict[Compartment, EventListRing]:
    if state.post_event_ring is None:
        state.post_event_ring = {}
    for comp in _ring_compartments(topology):
        ring = state.post_event_ring.get(comp)
        if (
            ring is None
            or ring.depth != depth
            or ring.device != device
            or ring.dtype != ring_dtype
            or ring.max_events != int(max_events)
        ):
            ring = EventListRing(
                depth=depth,
                device=device,
                dtype=ring_dtype,
                max_events=max_events,
            )
            state.post_event_ring[comp] = ring
    return state.post_event_ring


def _resolve_ring_dtype(ring_dtype: str | None, weights_dtype: Any, device: Any) -> Any:
    if ring_dtype is None:
        return weights_dtype
    torch = require_torch()
    resolved = _resolve_dtype(torch, ring_dtype)
    _validate_ring_dtype_supported(torch=torch, resolved_dtype=resolved, device=device)
    return resolved


def _validate_ring_dtype_supported(*, torch: Any, resolved_dtype: Any, device: Any) -> None:
    if device is None or getattr(device, "type", None) != "cpu":
        return
    try:
        probe = torch.zeros((2, 2), device=device, dtype=resolved_dtype)
        probe.add_(1)
        idx = torch.tensor([0, 1], device=device, dtype=torch.long)
        probe.index_add_(0, idx, probe)
    except Exception as exc:
        raise RuntimeError(
            f"ring_dtype={resolved_dtype} is not supported on CPU for ring operations."
        ) from exc


def _copy_ring_slot(slot: Tensor, out: Tensor) -> None:
    if slot.dtype == out.dtype:
        out.copy_(slot)
    else:
        out.copy_(slot.to(dtype=out.dtype))
    slot.zero_()


def _add_ring_slot(slot: Tensor, out: Tensor) -> None:
    if slot.dtype == out.dtype:
        out.add_(slot)
    else:
        out.add_(slot.to(dtype=out.dtype))
    slot.zero_()


def _as_ring_dtype(tensor: Tensor, ring_dtype: Any) -> Tensor:
    if tensor.dtype == ring_dtype:
        return tensor
    return tensor.to(dtype=ring_dtype)


def _ring_compartments(topology: SynapseTopology) -> tuple[Compartment, ...]:
    if topology.target_compartments is None:
        return (topology.target_compartment,)
    if topology.meta and "target_comp_ids" in topology.meta:
        comp_ids = topology.meta.get("target_comp_ids")
        if isinstance(comp_ids, list):
            comps: list[Compartment] = []
            for comp_id in comp_ids:
                comp = _COMPARTMENT_ORDER[int(comp_id)]
                if comp not in comps:
                    comps.append(comp)
            return tuple(comps)
    return _COMPARTMENT_ORDER


def _edges_by_delay(topology: SynapseTopology, device: Any) -> list[Tensor]:
    meta = topology.meta or {}
    cached = meta.get("edges_by_delay")
    if isinstance(cached, list):
        if cached:
            first = cached[0]
            if hasattr(first, "device") and device is not None and first.device != device:
                raise RuntimeError(
                    "edges_by_delay is on a different device; "
                    "compile_topology(..., build_edges_by_delay=True) with the synapse device."
                )
        return cached
    raise RuntimeError(
        "edges_by_delay missing; compile_topology(..., build_edges_by_delay=True) before stepping."
    )


def _require_pre_adjacency(topology: SynapseTopology, device: Any) -> tuple[Tensor, Tensor]:
    if not topology.meta:
        raise ValueError(
            "Topology meta missing pre adjacency; compile_topology(..., build_pre_adjacency=True) must be called."
        )
    pre_ptr = topology.meta.get("pre_ptr")
    edge_idx = topology.meta.get("edge_idx")
    if pre_ptr is None or edge_idx is None:
        raise ValueError(
            "Topology meta missing pre adjacency; compile_topology(..., build_pre_adjacency=True) must be called."
        )
    if pre_ptr.device != device or edge_idx.device != device:
        raise ValueError("Adjacency tensors must be on the synapse device")
    return cast(Tensor, pre_ptr), cast(Tensor, edge_idx)


def _gather_active_edges(
    active_pre: Tensor,
    pre_ptr: Tensor,
    edge_idx: Tensor,
    topology: SynapseTopology,
) -> Tensor:
    torch = require_torch()
    if active_pre.numel() == 0:
        return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))

    starts = pre_ptr.index_select(0, active_pre)
    ends = pre_ptr.index_select(0, active_pre + 1)
    counts = ends - starts
    if counts.numel() == 0:
        return cast(Tensor, torch.empty((0,), device=edge_idx.device, dtype=torch.long))
    base = torch.repeat_interleave(starts, counts)
    prefix = torch.cumsum(counts, 0)
    group_start = torch.repeat_interleave(prefix - counts, counts)
    intra = torch.arange(group_start.numel(), device=edge_idx.device) - group_start
    edge_pos = base + intra
    return cast(Tensor, edge_idx.index_select(0, edge_pos))


def _maybe_clamp_weights(
    model: DelayedCurrentSynapse,
    state: DelayedCurrentState,
    weights: Tensor,
) -> Tensor:
    if model.params.clamp_min is not None or model.params.clamp_max is not None:
        weights = weights.clamp(
            min=model.params.clamp_min if model.params.clamp_min is not None else None,
            max=model.params.clamp_max if model.params.clamp_max is not None else None,
        )
        state.weights.copy_(weights)
    return weights


def _as_like(
    tensor: Tensor,
    like: Tensor,
    *,
    name: str,
    validate_shapes: bool,
    expected_len: int | None,
    require_on_device: bool,
) -> Tensor:
    if tensor.device != like.device:
        if require_on_device:
            raise ValueError(
                f"{name} must be on device {like.device} with dtype {like.dtype}, got {tensor.device}/{tensor.dtype}"
            )
        raise ValueError(
            f"{name} must be on device {like.device} with dtype {like.dtype}, got {tensor.device}/{tensor.dtype}"
        )
    if tensor.dtype != like.dtype:
        torch = require_torch()
        if tensor.dtype != torch.bool:
            if require_on_device:
                raise ValueError(
                    f"{name} must be on device {like.device} with dtype {like.dtype}, got {tensor.device}/{tensor.dtype}"
                )
            raise ValueError(
                f"{name} must be on device {like.device} with dtype {like.dtype}, got {tensor.device}/{tensor.dtype}"
            )
    if validate_shapes and expected_len is not None:
        _ensure_1d_len(tensor, expected_len, name)
    return tensor


def _as_index(tensor: Tensor, device: Any, *, name: str) -> Tensor:
    torch = require_torch()
    if tensor.device != device or tensor.dtype != torch.long:
        raise ValueError(f"{name} must be on device {device} with dtype long")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got {tuple(tensor.shape)}")
    return tensor


def _ensure_1d_len(tensor: Tensor, n: int, name: str) -> None:
    if tensor.dim() != 1 or tensor.shape[0] != n:
        raise ValueError(f"{name} must have shape [{n}], got {tuple(tensor.shape)}")


def _infer_n_pre(topology: SynapseTopology) -> int | None:
    if topology.meta and "n_pre" in topology.meta:
        return int(topology.meta["n_pre"])
    return None


def _infer_n_post(topology: SynapseTopology) -> int:
    if topology.meta and "n_post" in topology.meta:
        return int(topology.meta["n_post"])
    raise ValueError("Topology meta missing n_post; compile_topology must be called before stepping.")


def _max_delay_steps(topology: SynapseTopology) -> int:
    if topology.meta and "max_delay_steps" in topology.meta:
        return int(topology.meta["max_delay_steps"])
    if topology.delay_steps is None:
        return 0
    raise ValueError(
        "Topology meta missing max_delay_steps; compile_topology must be called before stepping."
    )


def _use_no_grad(ctx: StepContext) -> bool:
    if ctx.extras and "no_grad" in ctx.extras:
        return bool(ctx.extras["no_grad"])
    return not ctx.is_training


def _validate_shapes(ctx: StepContext) -> bool:
    if ctx.extras and "validate_shapes" in ctx.extras:
        return bool(ctx.extras["validate_shapes"])
    return True


def _require_inputs_on_device(ctx: StepContext) -> bool:
    if ctx.extras and "require_inputs_on_device" in ctx.extras:
        return bool(ctx.extras["require_inputs_on_device"])
    return False
