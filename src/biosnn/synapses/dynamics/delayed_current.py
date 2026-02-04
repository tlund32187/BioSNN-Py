"""Delayed current synapse model with per-edge delays and ring buffer."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

from biosnn.biophysics.models._torch_utils import require_torch, resolve_device_dtype
from biosnn.contracts.neurons import Compartment, StepContext, Tensor
from biosnn.contracts.synapses import (
    ISynapseModel,
    ReceptorKind,
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)

_COMPARTMENT_ORDER = tuple(Compartment)
_COMPARTMENT_TO_ID = {comp: idx for idx, comp in enumerate(_COMPARTMENT_ORDER)}


@dataclass(frozen=True, slots=True)
class DelayedCurrentParams:
    """Simple delayed-current synapse parameters."""

    init_weight: float = 1e-9
    clamp_min: float | None = None
    clamp_max: float | None = None
    receptor_scale: Mapping[ReceptorKind, float] | None = None
    use_edge_buffer: bool = False
    event_driven: bool = False


@dataclass(slots=True)
class DelayedCurrentState:
    """Synapse state (weights + optional delay buffer)."""

    weights: Tensor  # shape [E]
    delay_buffer: Tensor | None  # shape [D, E] (legacy)
    post_ring: dict[Compartment, Tensor] | None  # shape [D, Npost] per compartment
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
        return DelayedCurrentState(weights=weights, delay_buffer=None, post_ring=None, cursor=0)

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
            if state.post_ring is not None:
                for ring in state.post_ring.values():
                    ring.zero_()
            state.cursor = 0
            return state
        state.weights[edge_indices] = self.params.init_weight
        if state.delay_buffer is not None:
            state.delay_buffer[:, edge_indices] = 0.0
        if state.post_ring is not None:
            for ring in state.post_ring.values():
                ring[:, :] = 0.0
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

    def state_tensors(self, state: DelayedCurrentState) -> Mapping[str, Tensor]:
        tensors: dict[str, Tensor] = {"weights": state.weights}
        if state.delay_buffer is not None:
            tensors["delay_buffer"] = state.delay_buffer
        return tensors

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
        if mask.any():
            out = torch.zeros((n_post,), device=device, dtype=dtype)
            out.scatter_add_(0, post_idx[mask], edge_current[mask])
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
        cursor = state.cursor % depth
        edge_idx = torch.arange(pre_idx.numel(), device=device)
        delayed = state.delay_buffer[cursor].clone()
        state.delay_buffer[cursor].zero_()
        immediate_mask = cast(Tensor, delay_steps == 0)
        if immediate_mask.any():
            delayed = delayed + edge_spikes * immediate_mask.to(edge_spikes.dtype)
        if (~immediate_mask).any():
            target_rows = torch.remainder(delay_steps + cursor, depth)
            state.delay_buffer[target_rows[~immediate_mask], edge_idx[~immediate_mask]] = edge_spikes[
                ~immediate_mask
            ]
        state.cursor = (cursor + 1) % depth
    else:
        delayed = edge_spikes

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
    post_ring = _ensure_post_ring(state, topology, depth, n_post, device, weights.dtype)

    cursor = state.cursor % depth
    post_drive: dict[Compartment, Tensor] = {}
    for comp, ring in post_ring.items():
        slot = ring[cursor]
        out = slot.clone()
        slot.zero_()
        post_drive[comp] = out

    weights = _maybe_clamp_weights(model, state, weights)

    for delay, edges in enumerate(edges_by_delay):
        if edges.numel() == 0:
            continue
        edge_pre = pre_idx.index_select(0, edges)
        edge_post = post_idx.index_select(0, edges)
        edge_spikes = pre_spikes.index_select(0, edge_pre)
        edge_weights = weights.index_select(0, edges)
        edge_current = edge_spikes * edge_weights
        edge_current = _apply_receptor_scale_edges(model, topology, edges, edge_current)

        if topology.target_compartments is None:
            comp = topology.target_compartment
            if delay == 0:
                post_drive[comp].index_add_(0, edge_post, edge_current)
            else:
                target_row = (cursor + delay) % depth
                post_ring[comp][target_row].index_add_(0, edge_post, edge_current)
            continue

        comp_ids = topology.target_compartments
        if comp_ids.device != device or comp_ids.dtype != torch.long:
            raise ValueError("target_compartments must be on the target device with dtype long")
        comp_ids = comp_ids.index_select(0, edges)
        for comp in post_drive:
            comp_id = _COMPARTMENT_TO_ID[comp]
            mask = comp_ids == comp_id
            if mask.any():
                posts = edge_post[mask]
                currents = edge_current[mask]
                if delay == 0:
                    post_drive[comp].index_add_(0, posts, currents)
                else:
                    target_row = (cursor + delay) % depth
                    post_ring[comp][target_row].index_add_(0, posts, currents)

    state.cursor = (cursor + 1) % depth
    return state, post_drive


def _step_post_ring_event_driven(
    model: DelayedCurrentSynapse,
    state: DelayedCurrentState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    pre_idx: Tensor,
    post_idx: Tensor,
    device: Any,
    weights: Tensor,
) -> tuple[DelayedCurrentState, Mapping[Compartment, Tensor], Mapping[str, Tensor]]:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
    post_ring = _ensure_post_ring(state, topology, depth, n_post, device, weights.dtype)

    cursor = state.cursor % depth
    post_drive: dict[Compartment, Tensor] = {}
    for comp, ring in post_ring.items():
        slot = ring[cursor]
        out = slot.clone()
        slot.zero_()
        post_drive[comp] = out

    weights = _maybe_clamp_weights(model, state, weights)

    active_pre = pre_spikes.nonzero(as_tuple=False).flatten()
    if active_pre.numel() == 0:
        state.cursor = (cursor + 1) % depth
        extras = {"processed_edges": weights.new_zeros(())}
        return state, post_drive, extras

    pre_ptr, edge_idx = _require_pre_adjacency(topology, device)
    edges = _gather_active_edges(active_pre, pre_ptr, edge_idx)
    if edges.numel() == 0:
        state.cursor = (cursor + 1) % depth
        extras = {"processed_edges": weights.new_zeros(())}
        return state, post_drive, extras

    edge_pre = pre_idx.index_select(0, edges)
    edge_post = post_idx.index_select(0, edges)
    edge_spikes = pre_spikes.index_select(0, edge_pre)
    edge_weights = weights.index_select(0, edges)
    edge_current = edge_spikes * edge_weights
    edge_current = _apply_receptor_scale_edges(model, topology, edges, edge_current)

    delay_steps = topology.delay_steps
    delay_vals = None if delay_steps is None else delay_steps.index_select(0, edges)

    if topology.target_compartments is None:
        comp = topology.target_compartment
        if delay_vals is None:
            post_drive[comp].index_add_(0, edge_post, edge_current)
        else:
            immediate_mask = delay_vals == 0
            if immediate_mask.any():
                post_drive[comp].index_add_(0, edge_post[immediate_mask], edge_current[immediate_mask])
            delayed_mask = ~immediate_mask
            if delayed_mask.any():
                target_rows = torch.remainder(delay_vals[delayed_mask] + cursor, depth)
                post_ring[comp].index_put_(
                    (target_rows, edge_post[delayed_mask]),
                    edge_current[delayed_mask],
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
            if not comp_mask.any():
                continue
            comp_posts = edge_post[comp_mask]
            comp_currents = edge_current[comp_mask]
            if delay_vals is None:
                post_drive[comp].index_add_(0, comp_posts, comp_currents)
                continue
            comp_delays = delay_vals[comp_mask]
            immediate_mask = comp_delays == 0
            if immediate_mask.any():
                post_drive[comp].index_add_(
                    0, comp_posts[immediate_mask], comp_currents[immediate_mask]
                )
            delayed_mask = ~immediate_mask
            if delayed_mask.any():
                target_rows = torch.remainder(comp_delays[delayed_mask] + cursor, depth)
                post_ring[comp].index_put_(
                    (target_rows, comp_posts[delayed_mask]),
                    comp_currents[delayed_mask],
                    accumulate=True,
                )

    state.cursor = (cursor + 1) % depth
    extras = {"processed_edges": weights.new_tensor(edges.numel())}
    return state, post_drive, extras


def _ensure_post_ring(
    state: DelayedCurrentState,
    topology: SynapseTopology,
    depth: int,
    n_post: int,
    device: Any,
    dtype: Any,
) -> dict[Compartment, Tensor]:
    torch = require_torch()
    if state.post_ring is None:
        state.post_ring = {}
    for comp in _ring_compartments(topology):
        ring = state.post_ring.get(comp)
        if ring is None or ring.shape != (depth, n_post) or ring.device != device or ring.dtype != dtype:
            ring = torch.zeros((depth, n_post), device=device, dtype=dtype)
            state.post_ring[comp] = ring
    return state.post_ring


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
    if isinstance(cached, list) and cached:
        return cached
    torch = require_torch()
    delay_steps = topology.delay_steps
    edge_count = topology.pre_idx.numel() if hasattr(topology.pre_idx, "numel") else 0
    if delay_steps is None or edge_count == 0:
        edges = torch.arange(edge_count, device=device, dtype=torch.long)
        edges_by_delay = [edges]
    else:
        max_delay = _max_delay_steps(topology)
        edges_by_delay = []
        for delay in range(max_delay + 1):
            mask = delay_steps == delay
            if mask.any():
                edges_by_delay.append(mask.nonzero(as_tuple=False).flatten())
            else:
                edges_by_delay.append(torch.empty((0,), device=device, dtype=torch.long))
    meta = dict(meta)
    meta["edges_by_delay"] = edges_by_delay
    object.__setattr__(topology, "meta", meta)
    return edges_by_delay


def _require_pre_adjacency(topology: SynapseTopology, device: Any) -> tuple[Tensor, Tensor]:
    if not topology.meta:
        raise ValueError("Topology meta missing pre adjacency; compile_topology must be called.")
    pre_ptr = topology.meta.get("pre_ptr")
    edge_idx = topology.meta.get("edge_idx")
    if pre_ptr is None or edge_idx is None:
        raise ValueError("Topology meta missing pre adjacency; compile_topology must be called.")
    if pre_ptr.device != device or edge_idx.device != device:
        raise ValueError("Adjacency tensors must be on the synapse device")
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
    if tensor.device != like.device or tensor.dtype != like.dtype:
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
