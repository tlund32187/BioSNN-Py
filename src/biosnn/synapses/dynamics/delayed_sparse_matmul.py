"""Delayed synapse backend using sparse matmul per delay bucket."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, cast

from biosnn.contracts.neurons import Compartment, StepContext, Tensor
from biosnn.contracts.synapses import (
    ISynapseModel,
    ISynapseModelInplace,
    ReceptorKind,
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)
from biosnn.core.torch_utils import _resolve_dtype, require_torch, resolve_device_dtype
from biosnn.synapses.buffers import BucketedEventRing
from biosnn.synapses.receptors import ReceptorProfile, resolve_profile_value

_COMPARTMENT_ORDER = tuple(Compartment)
_COMPARTMENT_TO_ID = {comp: idx for idx, comp in enumerate(_COMPARTMENT_ORDER)}
_EventRing = BucketedEventRing


@dataclass(frozen=True, slots=True)
class DelayedSparseMatmulParams:
    init_weight: float = 1e-9
    clamp_min: float | None = None
    clamp_max: float | None = None
    receptor_scale: Mapping[ReceptorKind, float] | None = None
    ring_dtype: str | None = None
    enable_sparse_updates: bool = False
    learning_update_target: Literal["both", "fused_only"] = "both"
    fused_layout: Literal["auto", "coo", "csr"] = "auto"
    store_sparse_by_delay: bool | None = None
    backend: Literal["spmm_fused", "event_driven"] = "spmm_fused"
    ring_strategy: Literal["dense", "event_bucketed"] = "dense"
    ring_capacity_max: int = 2_000_000
    receptor_profile: ReceptorProfile | None = None
    receptor_state_dtype: str | None = None


@dataclass(slots=True)
class DelayedSparseMatmulState:
    weights: Tensor
    post_ring: dict[Compartment, Tensor] | None
    post_event_ring: dict[Compartment, _EventRing] | None
    post_out: dict[Compartment, Tensor] | None
    cursor: int
    bind_weights_to_topology: bool = True
    pre_activity_buf: Tensor | None = None
    receptor_g: dict[Compartment, Tensor] | None = None
    receptor_decay: Tensor | None = None
    receptor_mix: Tensor | None = None
    receptor_sign: Tensor | None = None
    receptor_sign_values: tuple[float, ...] | None = None
    receptor_profile_kinds: tuple[ReceptorKind, ...] | None = None
    receptor_profile_dt: float | None = None
    receptor_profile_dtype: Any | None = None
    receptor_input_buf: dict[Compartment, Tensor] | None = None


class DelayedSparseMatmulSynapse(ISynapseModel, ISynapseModelInplace):
    """Sparse matmul delayed synapse (no per-edge enumeration per step).

    Note: if learning updates weights dynamically, call apply_weight_updates to keep
    the compiled sparse matrices in sync with the topology weights.
    """

    name = "delayed_sparse_matmul"

    def __init__(self, params: DelayedSparseMatmulParams | None = None) -> None:
        self.params = params or DelayedSparseMatmulParams()
        self._scale_cache: dict[
            tuple[object, object, tuple[ReceptorKind, ...], tuple[float, ...]],
            Tensor,
        ] = {}

    def init_state(self, e: int, *, ctx: StepContext) -> DelayedSparseMatmulState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        weights = torch.full((e,), self.params.init_weight, device=device, dtype=dtype)
        return DelayedSparseMatmulState(
            weights=weights,
            post_ring=None,
            post_event_ring=None,
            post_out=None,
            cursor=0,
            pre_activity_buf=None,
        )

    def reset_state(
        self,
        state: DelayedSparseMatmulState,
        *,
        ctx: StepContext,
        edge_indices: Tensor | None = None,
    ) -> DelayedSparseMatmulState:
        _ = ctx
        if edge_indices is None:
            state.weights.fill_(self.params.init_weight)
            if state.post_ring is not None:
                for ring in state.post_ring.values():
                    ring.zero_()
            if state.post_out is not None:
                for out in state.post_out.values():
                    out.zero_()
            if state.post_event_ring is not None:
                for event_ring in state.post_event_ring.values():
                    event_ring.clear()
            state.cursor = 0
            if state.pre_activity_buf is not None:
                state.pre_activity_buf.zero_()
            if state.receptor_g is not None:
                for g in state.receptor_g.values():
                    g.zero_()
            if state.receptor_input_buf is not None:
                for buf in state.receptor_input_buf.values():
                    buf.zero_()
            return state
        state.weights[edge_indices] = self.params.init_weight
        if state.post_ring is not None:
            for ring in state.post_ring.values():
                ring[:, :] = 0.0
        if state.post_out is not None:
            for out in state.post_out.values():
                out[:] = 0.0
        if state.post_event_ring is not None:
            for event_ring in state.post_event_ring.values():
                event_ring.clear()
        if state.pre_activity_buf is not None:
            state.pre_activity_buf.zero_()
        if state.receptor_g is not None:
            for g in state.receptor_g.values():
                g.zero_()
        if state.receptor_input_buf is not None:
            for buf in state.receptor_input_buf.values():
                buf.zero_()
        return state

    def step(
        self,
        state: DelayedSparseMatmulState,
        topology: SynapseTopology,
        inputs: SynapseInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[DelayedSparseMatmulState, SynapseStepResult]:
        torch = require_torch()
        use_no_grad = _use_no_grad(ctx)
        with torch.no_grad() if use_no_grad else nullcontext():
            _ensure_supported_topology(topology)
            _validate_backend_config(self.params)
            weights = _resolve_weights(state, topology, ctx)
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

            if self.params.backend == "event_driven":
                post_drive, extras = _step_event_driven(
                    self,
                    state,
                    topology,
                    pre_spikes,
                    device,
                    weights,
                    dt,
                )
                return state, SynapseStepResult(post_drive=post_drive, extras=extras)
            post_drive = _step_sparse_matmul(
                self,
                state,
                topology,
                pre_spikes,
                device,
                weights,
                dt,
            )
            return state, SynapseStepResult(post_drive=post_drive)

    def step_into(
        self,
        state: DelayedSparseMatmulState,
        pre_spikes: Tensor,
        out_drive: MutableMapping[Compartment, Tensor],
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
            _ensure_supported_topology(topology)
            _validate_backend_config(self.params)
            weights = _resolve_weights(state, topology, ctx)
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
            if self.params.backend == "event_driven":
                _step_event_driven_into(
                    self,
                    state,
                    topology,
                    pre_spikes,
                    device,
                    weights,
                    out_drive,
                    dt,
                )
                return
            _step_sparse_matmul_into(
                self,
                state,
                topology,
                pre_spikes,
                device,
                weights,
                out_drive,
                dt,
            )

    def state_tensors(self, state: DelayedSparseMatmulState) -> Mapping[str, Tensor]:
        return {"weights": state.weights}

    def compilation_requirements(self) -> Mapping[str, bool | str | None]:
        use_event_driven = self.params.backend == "event_driven"
        wants_by_delay_sparse = bool(self.params.store_sparse_by_delay) and not use_event_driven
        wants_fused_sparse = not use_event_driven
        requirements: dict[str, bool | str | None] = {
            "needs_edges_by_delay": False,
            "needs_pre_adjacency": use_event_driven,
            "needs_sparse_delay_mats": not use_event_driven,
            "needs_bucket_edge_mapping": False,
            "wants_fused_sparse": wants_fused_sparse,
            "wants_by_delay_sparse": wants_by_delay_sparse,
            "wants_bucket_edge_mapping": bool(self.params.enable_sparse_updates),
            "wants_weights_snapshot_each_step": False,
            "wants_projection_drive_tensor": False,
            "wants_fused_layout": self.params.fused_layout if wants_fused_sparse else "auto",
            "ring_strategy": self.params.ring_strategy if use_event_driven else "dense",
            "ring_dtype": self.params.ring_dtype,
        }
        if wants_fused_sparse and self.params.fused_layout == "csr":
            requirements["wants_fused_csr"] = True
        elif wants_fused_sparse and self.params.fused_layout == "coo":
            requirements["wants_fused_csr"] = False
        if self.params.store_sparse_by_delay is not None and not use_event_driven:
            requirements["store_sparse_by_delay"] = bool(self.params.store_sparse_by_delay)
        return requirements

    def _scale_table(
        self,
        like: Tensor,
        kinds: tuple[ReceptorKind, ...],
        scale_map: Mapping[ReceptorKind, float],
    ) -> Tensor:
        values = tuple(
            resolve_profile_value(scale_map, kind, default=1.0)
            for kind in kinds
        )
        key = (like.device, like.dtype, kinds, values)
        cached = self._scale_cache.get(key)
        if cached is not None:
            return cached
        torch = require_torch()
        table = cast(Tensor, torch.tensor(values, device=like.device, dtype=like.dtype))
        self._scale_cache[key] = table
        return table

    def apply_weight_updates(
        self,
        topology: SynapseTopology,
        active_edges: Tensor | None,
        d_weights: Tensor,
    ) -> None:
        update_target = self.params.learning_update_target
        weights = topology.weights
        if weights is None:
            raise RuntimeError("DelayedSparseMatmulSynapse requires topology.weights for updates")

        update_all = d_weights.numel() == weights.numel()
        if update_all:
            weights.add_(d_weights)
            edge_ids = None
            delta = d_weights
        elif active_edges is not None and d_weights.numel() == active_edges.numel():
            weights.index_add_(0, active_edges, d_weights)
            edge_ids = active_edges
            delta = d_weights
        else:
            raise RuntimeError(
                "Sparse matmul weight update shape mismatch: "
                f"d_weights={tuple(d_weights.shape)} active_edges={tuple(active_edges.shape) if active_edges is not None else None} "
                f"weights={tuple(weights.shape)}"
            )

        if edge_ids is None:
            _sync_sparse_values(topology)
            return

        (
            values_by_comp,
            edge_bucket_comp,
            edge_bucket_delay,
            edge_bucket_pos,
            edge_scale,
            fused_values_by_comp,
            edge_bucket_fused_pos,
        ) = _require_sparse_update_meta(topology)
        comp_ids = edge_bucket_comp.index_select(0, edge_ids)
        delay_ids = edge_bucket_delay.index_select(0, edge_ids)
        pos_ids = edge_bucket_pos.index_select(0, edge_ids)
        scale_vals = edge_scale.index_select(0, edge_ids) if edge_scale is not None else None

        if update_target == "fused_only":
            _apply_sparse_value_updates(
                values_by_comp=None,
                comp_ids=comp_ids,
                delay_ids=delay_ids,
                pos_ids=pos_ids,
                delta=delta,
                scale_vals=scale_vals,
                fused_values_by_comp=fused_values_by_comp,
                fused_pos_ids=edge_bucket_fused_pos.index_select(0, edge_ids)
                if edge_bucket_fused_pos is not None
                else None,
            )
        else:
            _apply_sparse_value_updates(
                values_by_comp=values_by_comp,
                comp_ids=comp_ids,
                delay_ids=delay_ids,
                pos_ids=pos_ids,
                delta=delta,
                scale_vals=scale_vals,
                fused_values_by_comp=fused_values_by_comp,
                fused_pos_ids=edge_bucket_fused_pos.index_select(0, edge_ids)
                if edge_bucket_fused_pos is not None
                else None,
            )

    def sync_sparse_values(self, topology: SynapseTopology) -> None:
        _sync_sparse_values(topology)


def _step_sparse_matmul(
    model: DelayedSparseMatmulSynapse,
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    device: Any,
    weights: Tensor,
    dt: float,
) -> Mapping[Compartment, Tensor]:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
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

    weights = _maybe_clamp_weights(model, state, topology, weights)
    pre_activity = _fill_pre_activity_buf(state, pre_spikes, weights, device)

    # Preferred path: fused delay buckets (single sparse matmul per compartment).
    fused_mats, fused_delays, fused_n_post, fused_d, fused_immediate_idx, fused_delayed_idx = (
        _fused_sparse_by_comp(topology, device, fused_layout=model.params.fused_layout)
    )
    if fused_mats:
        for comp, fused in fused_mats.items():
            delays = fused_delays.get(comp)
            if delays is None or delays.numel() == 0:
                continue
            immediate_idx = fused_immediate_idx.get(comp)
            delayed_idx = fused_delayed_idx.get(comp)
            n_post_comp = fused_n_post.get(comp, n_post)
            d_val = fused_d.get(comp, int(delays.numel()))
            contrib_flat = torch.sparse.mm(fused, pre_activity.unsqueeze(1)).squeeze(1)
            if contrib_flat.numel() == 0:
                continue
            contrib = contrib_flat.view(int(d_val), n_post_comp)
            if immediate_idx is not None and immediate_idx.numel() > 0:
                post_drive[comp].add_(contrib.index_select(0, immediate_idx).sum(dim=0))
            if delayed_idx is not None and delayed_idx.numel() > 0:
                ring_indices = torch.remainder(delays + cursor, depth)
                contrib_ring = _as_ring_dtype(contrib, ring_dtype)
                post_ring[comp].index_add_(
                    0,
                    ring_indices.index_select(0, delayed_idx),
                    contrib_ring.index_select(0, delayed_idx),
                )
    else:
        mats_by_comp = _nonempty_mats_by_comp(topology, device)
        for comp, mats in mats_by_comp.items():
            for delay, mat in mats:
                contrib = torch.sparse.mm(mat, pre_activity.unsqueeze(1)).squeeze(1)
                target_row = (cursor + delay) % depth
                if delay == 0:
                    post_drive[comp].add_(contrib)
                else:
                    post_ring[comp][target_row].add_(_as_ring_dtype(contrib, ring_dtype))

    _apply_receptor_profile_if_enabled(
        model=model,
        state=state,
        topology=topology,
        drive_by_comp=post_drive,
        dt=dt,
        n_post=n_post,
        device=device,
        weights_dtype=weights.dtype,
    )

    state.cursor = (cursor + 1) % depth
    return post_drive


def _step_sparse_matmul_into(
    model: DelayedSparseMatmulSynapse,
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    device: Any,
    weights: Tensor,
    out_drive: MutableMapping[Compartment, Tensor],
    dt: float,
) -> None:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
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

    use_receptor_profile = model.params.receptor_profile is not None
    cursor = state.cursor % depth
    proj_drive: MutableMapping[Compartment, Tensor]
    if use_receptor_profile:
        proj_drive = post_out
        for comp, ring in post_ring.items():
            slot = ring[cursor]
            out = proj_drive[comp]
            out.zero_()
            _copy_ring_slot(slot, out)
    else:
        proj_drive = out_drive
        for comp, ring in post_ring.items():
            slot = ring[cursor]
            if comp not in out_drive:
                raise KeyError(f"Drive accumulator missing compartment {comp}")
            _add_ring_slot(slot, out_drive[comp])

    weights = _maybe_clamp_weights(model, state, topology, weights)
    pre_activity = _fill_pre_activity_buf(state, pre_spikes, weights, device)

    # Preferred path: fused delay buckets (single sparse matmul per compartment).
    fused_mats, fused_delays, fused_n_post, fused_d, fused_immediate_idx, fused_delayed_idx = (
        _fused_sparse_by_comp(topology, device, fused_layout=model.params.fused_layout)
    )
    if fused_mats:
        for comp, fused in fused_mats.items():
            if comp not in proj_drive:
                raise KeyError(f"Drive accumulator missing compartment {comp}")
            delays = fused_delays.get(comp)
            if delays is None or delays.numel() == 0:
                continue
            immediate_idx = fused_immediate_idx.get(comp)
            delayed_idx = fused_delayed_idx.get(comp)
            n_post_comp = fused_n_post.get(comp, n_post)
            d_val = fused_d.get(comp, int(delays.numel()))
            contrib_flat = torch.sparse.mm(fused, pre_activity.unsqueeze(1)).squeeze(1)
            if contrib_flat.numel() == 0:
                continue
            contrib = contrib_flat.view(int(d_val), n_post_comp)
            if immediate_idx is not None and immediate_idx.numel() > 0:
                proj_drive[comp].add_(contrib.index_select(0, immediate_idx).sum(dim=0))
            if delayed_idx is not None and delayed_idx.numel() > 0:
                ring_indices = torch.remainder(delays + cursor, depth)
                contrib_ring = _as_ring_dtype(contrib, ring_dtype)
                post_ring[comp].index_add_(
                    0,
                    ring_indices.index_select(0, delayed_idx),
                    contrib_ring.index_select(0, delayed_idx),
                )
    else:
        mats_by_comp = _nonempty_mats_by_comp(topology, device)
        for comp, mats in mats_by_comp.items():
            if comp not in proj_drive:
                raise KeyError(f"Drive accumulator missing compartment {comp}")
            for delay, mat in mats:
                contrib = torch.sparse.mm(mat, pre_activity.unsqueeze(1)).squeeze(1)
                target_row = (cursor + delay) % depth
                if delay == 0:
                    proj_drive[comp].add_(contrib)
                else:
                    post_ring[comp][target_row].add_(_as_ring_dtype(contrib, ring_dtype))

    if use_receptor_profile:
        _apply_receptor_profile_if_enabled(
            model=model,
            state=state,
            topology=topology,
            drive_by_comp=proj_drive,
            dt=dt,
            n_post=n_post,
            device=device,
            weights_dtype=weights.dtype,
        )
        for comp, proj_out in proj_drive.items():
            if comp not in out_drive:
                raise KeyError(f"Drive accumulator missing compartment {comp}")
            out_drive[comp].add_(proj_out)

    state.cursor = (cursor + 1) % depth


def _step_event_driven(
    model: DelayedSparseMatmulSynapse,
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    device: Any,
    weights: Tensor,
    dt: float,
) -> tuple[Mapping[Compartment, Tensor], Mapping[str, Tensor]]:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
    ring_dtype = _resolve_ring_dtype(model.params.ring_dtype, weights.dtype, device)
    use_bucketed_ring = model.params.ring_strategy == "event_bucketed"
    post_ring: dict[Compartment, Tensor] | None = None
    event_ring: dict[Compartment, _EventRing] | None = None
    if use_bucketed_ring:
        post_out = _ensure_post_out(
            state,
            topology,
            n_post,
            device,
            weights.dtype,
        )
        event_ring = _ensure_event_ring(
            state,
            topology,
            depth,
            device,
            ring_dtype,
            capacity_max=int(model.params.ring_capacity_max),
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
    if use_bucketed_ring:
        assert event_ring is not None
        for comp, bucket_ring in event_ring.items():
            out = post_out[comp]
            out.zero_()
            bucket_ring.pop_into(cursor, out)
            post_drive[comp] = out
    else:
        assert post_ring is not None
        for comp, dense_ring in post_ring.items():
            slot = dense_ring[cursor]
            out = post_out[comp]
            _copy_ring_slot(slot, out)
            post_drive[comp] = out

    weights = _maybe_clamp_weights(model, state, topology, weights)
    pre_activity = _fill_pre_activity_buf(state, pre_spikes, weights, device)
    active_pre = pre_activity.nonzero(as_tuple=False).flatten()
    if active_pre.numel() == 0:
        _apply_receptor_profile_if_enabled(
            model=model,
            state=state,
            topology=topology,
            drive_by_comp=post_drive,
            dt=dt,
            n_post=n_post,
            device=device,
            weights_dtype=weights.dtype,
        )
        state.cursor = (cursor + 1) % depth
        return post_drive, {"processed_edges": weights.new_zeros(())}

    pre_ptr, edge_idx = _require_pre_adjacency(topology, device)
    edges = _gather_active_edges(active_pre, pre_ptr, edge_idx)
    if edges.numel() == 0:
        _apply_receptor_profile_if_enabled(
            model=model,
            state=state,
            topology=topology,
            drive_by_comp=post_drive,
            dt=dt,
            n_post=n_post,
            device=device,
            weights_dtype=weights.dtype,
        )
        state.cursor = (cursor + 1) % depth
        return post_drive, {"processed_edges": weights.new_zeros(())}

    edge_pre = topology.pre_idx.index_select(0, edges)
    edge_post = topology.post_idx.index_select(0, edges)
    edge_activity = pre_activity.index_select(0, edge_pre)
    edge_weights = weights.index_select(0, edges)
    edge_current = edge_activity * edge_weights
    edge_current = _apply_receptor_scale_edges(model, topology, edges, edge_current)

    delay_steps = topology.delay_steps
    delay_vals = None if delay_steps is None else delay_steps.index_select(0, edges)
    post_ring_opt = None if use_bucketed_ring else post_ring
    if topology.target_compartments is None:
        comp = topology.target_compartment
        _route_currents_for_comp(
            posts=edge_post,
            currents=edge_current,
            delays=delay_vals,
            cursor=cursor,
            depth=depth,
            out=post_drive[comp],
            dense_ring=post_ring_opt[comp] if post_ring_opt is not None else None,
            event_ring=event_ring[comp] if event_ring is not None else None,
            ring_dtype=ring_dtype,
        )
    else:
        comp_ids = topology.target_compartments
        if comp_ids.device != device or comp_ids.dtype != torch.long:
            raise ValueError("target_compartments must be on the target device with dtype long")
        comp_ids = comp_ids.index_select(0, edges)
        for comp, out in post_drive.items():
            comp_mask = comp_ids == _COMPARTMENT_TO_ID[comp]
            comp_idx = comp_mask.nonzero(as_tuple=False).flatten()
            if comp_idx.numel() == 0:
                continue
            comp_posts = edge_post.index_select(0, comp_idx)
            comp_currents = edge_current.index_select(0, comp_idx)
            comp_delays = None if delay_vals is None else delay_vals.index_select(0, comp_idx)
            _route_currents_for_comp(
                posts=comp_posts,
                currents=comp_currents,
                delays=comp_delays,
                cursor=cursor,
                depth=depth,
                out=out,
                dense_ring=post_ring_opt[comp] if post_ring_opt is not None else None,
                event_ring=event_ring[comp] if event_ring is not None else None,
                ring_dtype=ring_dtype,
            )

    _apply_receptor_profile_if_enabled(
        model=model,
        state=state,
        topology=topology,
        drive_by_comp=post_drive,
        dt=dt,
        n_post=n_post,
        device=device,
        weights_dtype=weights.dtype,
    )

    state.cursor = (cursor + 1) % depth
    return post_drive, {"processed_edges": weights.new_tensor(edges.numel())}


def _step_event_driven_into(
    model: DelayedSparseMatmulSynapse,
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    device: Any,
    weights: Tensor,
    out_drive: MutableMapping[Compartment, Tensor],
    dt: float,
) -> None:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
    ring_dtype = _resolve_ring_dtype(model.params.ring_dtype, weights.dtype, device)
    use_bucketed_ring = model.params.ring_strategy == "event_bucketed"
    use_receptor_profile = model.params.receptor_profile is not None
    post_ring: dict[Compartment, Tensor] | None = None
    event_ring: dict[Compartment, _EventRing] | None = None
    post_out: dict[Compartment, Tensor] | None = None
    if use_bucketed_ring:
        if use_receptor_profile:
            post_out = _ensure_post_out(
                state,
                topology,
                n_post,
                device,
                weights.dtype,
            )
        event_ring = _ensure_event_ring(
            state,
            topology,
            depth,
            device,
            ring_dtype,
            capacity_max=int(model.params.ring_capacity_max),
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
    proj_drive: MutableMapping[Compartment, Tensor]
    if use_receptor_profile:
        assert post_out is not None
        proj_drive = post_out
        if use_bucketed_ring:
            assert event_ring is not None
            for comp, bucket_ring in event_ring.items():
                out = proj_drive[comp]
                out.zero_()
                bucket_ring.pop_into(cursor, out)
        else:
            assert post_ring is not None
            for comp, dense_ring in post_ring.items():
                out = proj_drive[comp]
                out.zero_()
                _copy_ring_slot(dense_ring[cursor], out)
    else:
        proj_drive = out_drive
        if use_bucketed_ring:
            assert event_ring is not None
            for comp, bucket_ring in event_ring.items():
                if comp not in out_drive:
                    raise KeyError(f"Drive accumulator missing compartment {comp}")
                bucket_ring.pop_into(cursor, out_drive[comp])
        else:
            assert post_ring is not None
            for comp, dense_ring in post_ring.items():
                if comp not in out_drive:
                    raise KeyError(f"Drive accumulator missing compartment {comp}")
                _add_ring_slot(dense_ring[cursor], out_drive[comp])

    weights = _maybe_clamp_weights(model, state, topology, weights)
    pre_activity = _fill_pre_activity_buf(state, pre_spikes, weights, device)
    active_pre = pre_activity.nonzero(as_tuple=False).flatten()
    if active_pre.numel() == 0:
        if use_receptor_profile:
            _apply_receptor_profile_if_enabled(
                model=model,
                state=state,
                topology=topology,
                drive_by_comp=proj_drive,
                dt=dt,
                n_post=n_post,
                device=device,
                weights_dtype=weights.dtype,
            )
            for comp, proj_out in proj_drive.items():
                if comp not in out_drive:
                    raise KeyError(f"Drive accumulator missing compartment {comp}")
                out_drive[comp].add_(proj_out)
        state.cursor = (cursor + 1) % depth
        return

    pre_ptr, edge_idx = _require_pre_adjacency(topology, device)
    edges = _gather_active_edges(active_pre, pre_ptr, edge_idx)
    if edges.numel() == 0:
        if use_receptor_profile:
            _apply_receptor_profile_if_enabled(
                model=model,
                state=state,
                topology=topology,
                drive_by_comp=proj_drive,
                dt=dt,
                n_post=n_post,
                device=device,
                weights_dtype=weights.dtype,
            )
            for comp, proj_out in proj_drive.items():
                if comp not in out_drive:
                    raise KeyError(f"Drive accumulator missing compartment {comp}")
                out_drive[comp].add_(proj_out)
        state.cursor = (cursor + 1) % depth
        return

    edge_pre = topology.pre_idx.index_select(0, edges)
    edge_post = topology.post_idx.index_select(0, edges)
    edge_activity = pre_activity.index_select(0, edge_pre)
    edge_weights = weights.index_select(0, edges)
    edge_current = edge_activity * edge_weights
    edge_current = _apply_receptor_scale_edges(model, topology, edges, edge_current)

    delay_steps = topology.delay_steps
    delay_vals = None if delay_steps is None else delay_steps.index_select(0, edges)
    post_ring_opt = None if use_bucketed_ring else post_ring
    if topology.target_compartments is None:
        comp = topology.target_compartment
        if comp not in proj_drive:
            raise KeyError(f"Drive accumulator missing compartment {comp}")
        _route_currents_for_comp(
            posts=edge_post,
            currents=edge_current,
            delays=delay_vals,
            cursor=cursor,
            depth=depth,
            out=proj_drive[comp],
            dense_ring=post_ring_opt[comp] if post_ring_opt is not None else None,
            event_ring=event_ring[comp] if event_ring is not None else None,
            ring_dtype=ring_dtype,
        )
    else:
        comp_ids = topology.target_compartments
        if comp_ids.device != device or comp_ids.dtype != torch.long:
            raise ValueError("target_compartments must be on the target device with dtype long")
        comp_ids = comp_ids.index_select(0, edges)
        for comp, out in proj_drive.items():
            comp_mask = comp_ids == _COMPARTMENT_TO_ID[comp]
            comp_idx = comp_mask.nonzero(as_tuple=False).flatten()
            if comp_idx.numel() == 0:
                continue
            comp_posts = edge_post.index_select(0, comp_idx)
            comp_currents = edge_current.index_select(0, comp_idx)
            comp_delays = None if delay_vals is None else delay_vals.index_select(0, comp_idx)
            _route_currents_for_comp(
                posts=comp_posts,
                currents=comp_currents,
                delays=comp_delays,
                cursor=cursor,
                depth=depth,
                out=out,
                dense_ring=post_ring_opt[comp] if post_ring_opt is not None else None,
                event_ring=event_ring[comp] if event_ring is not None else None,
                ring_dtype=ring_dtype,
            )

    if use_receptor_profile:
        _apply_receptor_profile_if_enabled(
            model=model,
            state=state,
            topology=topology,
            drive_by_comp=proj_drive,
            dt=dt,
            n_post=n_post,
            device=device,
            weights_dtype=weights.dtype,
        )
        for comp, proj_out in proj_drive.items():
            if comp not in out_drive:
                raise KeyError(f"Drive accumulator missing compartment {comp}")
            out_drive[comp].add_(proj_out)

    state.cursor = (cursor + 1) % depth


def _route_currents_for_comp(
    *,
    posts: Tensor,
    currents: Tensor,
    delays: Tensor | None,
    cursor: int,
    depth: int,
    out: Tensor,
    dense_ring: Tensor | None,
    event_ring: _EventRing | None,
    ring_dtype: Any,
) -> None:
    torch = require_torch()
    if delays is None:
        out.index_add_(0, posts, currents)
        return
    immediate_mask = delays == 0
    immediate_idx = immediate_mask.nonzero(as_tuple=False).flatten()
    if immediate_idx.numel() > 0:
        out.index_add_(
            0,
            posts.index_select(0, immediate_idx),
            currents.index_select(0, immediate_idx),
        )
    delayed_idx = (~immediate_mask).nonzero(as_tuple=False).flatten()
    if delayed_idx.numel() == 0:
        return
    target_rows = torch.remainder(delays.index_select(0, delayed_idx) + cursor, depth)
    delayed_posts = posts.index_select(0, delayed_idx)
    delayed_vals = _as_ring_dtype(currents.index_select(0, delayed_idx), ring_dtype)
    if event_ring is not None:
        event_ring.schedule(target_rows, delayed_posts, delayed_vals)
        return
    if dense_ring is None:
        raise RuntimeError("Delayed sparse ring target missing for delayed currents.")
    dense_ring.index_put_(
        (target_rows, delayed_posts),
        delayed_vals,
        accumulate=True,
    )


def _ensure_post_ring(
    state: DelayedSparseMatmulState,
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
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    n_post: int,
    device: Any,
    dtype: Any,
) -> dict[Compartment, Tensor]:
    torch = require_torch()
    if state.post_out is None:
        state.post_out = {}
    for comp in _ring_compartments(topology):
        out = state.post_out.get(comp)
        if out is None or out.shape != (n_post,) or out.device != device or out.dtype != dtype:
            out = torch.zeros((n_post,), device=device, dtype=dtype)
            state.post_out[comp] = out
    return state.post_out


def _ensure_event_ring(
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    depth: int,
    device: Any,
    ring_dtype: Any,
    *,
    capacity_max: int,
) -> dict[Compartment, _EventRing]:
    if state.post_event_ring is None:
        state.post_event_ring = {}
    for comp in _ring_compartments(topology):
        ring = state.post_event_ring.get(comp)
        if (
            ring is None
            or ring.depth != depth
            or ring.device != device
            or ring.dtype != ring_dtype
            or ring.capacity_max != capacity_max
        ):
            ring = BucketedEventRing(
                depth=depth,
                device=device,
                dtype=ring_dtype,
                capacity_max=capacity_max,
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


def _apply_receptor_profile_if_enabled(
    *,
    model: DelayedSparseMatmulSynapse,
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    drive_by_comp: Mapping[Compartment, Tensor],
    dt: float,
    n_post: int,
    device: Any,
    weights_dtype: Any,
) -> None:
    profile = model.params.receptor_profile
    if profile is None:
        return
    _ensure_receptor_profile_state(
        model=model,
        state=state,
        topology=topology,
        dt=dt,
        n_post=n_post,
        device=device,
        weights_dtype=weights_dtype,
    )
    if state.receptor_g is None or state.receptor_decay is None or state.receptor_mix is None:
        return
    signs = state.receptor_sign_values or ()
    for comp, out in drive_by_comp.items():
        g = state.receptor_g.get(comp)
        if g is None:
            continue
        receptor_in = out
        if receptor_in.dtype != g.dtype:
            receptor_in = _ensure_receptor_input_buf(
                state=state,
                comp=comp,
                n_post=n_post,
                device=device,
                dtype=g.dtype,
            )
            receptor_in.copy_(out)
        g.mul_(state.receptor_decay)
        g.addcmul_(state.receptor_mix, receptor_in.unsqueeze(0))
        out.zero_()
        if out.dtype == g.dtype:
            for ridx, sign in enumerate(signs):
                if sign != 0.0:
                    out.add_(g[ridx], alpha=sign)
        else:
            for ridx, sign in enumerate(signs):
                if sign != 0.0:
                    out.add_(g[ridx].to(dtype=out.dtype), alpha=sign)


def _ensure_receptor_profile_state(
    *,
    model: DelayedSparseMatmulSynapse,
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    dt: float,
    n_post: int,
    device: Any,
    weights_dtype: Any,
) -> None:
    profile = model.params.receptor_profile
    if profile is None:
        return
    torch = require_torch()
    kinds = tuple(profile.kinds)
    if len(kinds) == 0:
        raise ValueError("receptor_profile.kinds must be non-empty.")
    receptor_dtype = _resolve_receptor_profile_dtype(
        receptor_state_dtype=model.params.receptor_state_dtype,
        weights_dtype=weights_dtype,
    )
    if state.receptor_g is None:
        state.receptor_g = {}
    for comp in _ring_compartments(topology):
        g = state.receptor_g.get(comp)
        if (
            g is None
            or g.shape != (len(kinds), n_post)
            or g.device != device
            or g.dtype != receptor_dtype
        ):
            state.receptor_g[comp] = torch.zeros(
                (len(kinds), n_post), device=device, dtype=receptor_dtype
            )

    cache_invalid = (
        state.receptor_decay is None
        or state.receptor_mix is None
        or state.receptor_sign is None
        or state.receptor_profile_kinds != kinds
        or state.receptor_profile_dt != float(dt)
        or state.receptor_profile_dtype != receptor_dtype
        or state.receptor_decay.device != device
        or state.receptor_mix.device != device
        or state.receptor_sign.device != device
    )
    if cache_invalid:
        tau_vals = [
            max(resolve_profile_value(profile.tau, kind, default=1.0), 1e-9)
            for kind in kinds
        ]
        mix_vals = [resolve_profile_value(profile.mix, kind, default=1.0) for kind in kinds]
        sign_vals = [resolve_profile_value(profile.sign, kind, default=1.0) for kind in kinds]
        tau = cast(
            Tensor,
            torch.tensor(tau_vals, device=device, dtype=receptor_dtype),
        )
        decay = cast(Tensor, torch.exp(-float(dt) / tau)).unsqueeze(1)
        mix = cast(
            Tensor,
            torch.tensor(mix_vals, device=device, dtype=receptor_dtype),
        ).unsqueeze(1)
        sign = cast(
            Tensor,
            torch.tensor(sign_vals, device=device, dtype=receptor_dtype),
        ).unsqueeze(1)
        state.receptor_decay = decay
        state.receptor_mix = mix
        state.receptor_sign = sign
        state.receptor_sign_values = tuple(float(v) for v in sign_vals)
        state.receptor_profile_kinds = kinds
        state.receptor_profile_dt = float(dt)
        state.receptor_profile_dtype = receptor_dtype


def _ensure_receptor_input_buf(
    *,
    state: DelayedSparseMatmulState,
    comp: Compartment,
    n_post: int,
    device: Any,
    dtype: Any,
) -> Tensor:
    torch = require_torch()
    if state.receptor_input_buf is None:
        state.receptor_input_buf = {}
    buf = state.receptor_input_buf.get(comp)
    if buf is None or buf.shape != (n_post,) or buf.device != device or buf.dtype != dtype:
        buf = torch.zeros((n_post,), device=device, dtype=dtype)
        state.receptor_input_buf[comp] = buf
    return buf


def _resolve_receptor_profile_dtype(
    *,
    receptor_state_dtype: str | None,
    weights_dtype: Any,
) -> Any:
    if receptor_state_dtype is None:
        return weights_dtype
    torch = require_torch()
    return _resolve_dtype(torch, receptor_state_dtype)


def _resolve_receptor_scale_map(
    model: DelayedSparseMatmulSynapse,
    topology: SynapseTopology,
) -> Mapping[ReceptorKind, float] | None:
    meta = topology.meta
    if isinstance(meta, Mapping):
        scale_meta = meta.get("receptor_scale")
        if isinstance(scale_meta, Mapping):
            return cast(Mapping[ReceptorKind, float], scale_meta)
    return model.params.receptor_scale


def _apply_receptor_scale_edges(
    model: DelayedSparseMatmulSynapse,
    topology: SynapseTopology,
    edges: Tensor,
    edge_current: Tensor,
) -> Tensor:
    if topology.receptor is None:
        return edge_current
    scale_map = _resolve_receptor_scale_map(model, topology)
    if scale_map is None:
        return edge_current
    torch = require_torch()
    receptor_ids = topology.receptor
    if receptor_ids.device != edge_current.device or receptor_ids.dtype != torch.long:
        raise ValueError("receptor ids must be on the same device with dtype long")
    receptor_ids = receptor_ids.index_select(0, edges)
    kinds = topology.receptor_kinds or (ReceptorKind.AMPA, ReceptorKind.NMDA, ReceptorKind.GABA)
    table = model._scale_table(edge_current, kinds, scale_map)
    edge_scale = table.index_select(0, receptor_ids)
    return edge_current * edge_scale


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


def _sparse_mats_by_delay_by_comp(
    topology: SynapseTopology, device: Any
) -> dict[Compartment, list[Tensor | None]]:
    meta = topology.meta or {}
    cached = meta.get("W_by_delay_by_comp")
    if isinstance(cached, dict):
        return _validate_sparse_mats_by_comp(cached, device)
    fallback = meta.get("W_by_delay")
    if isinstance(fallback, list):
        comp = topology.target_compartment
        return {comp: _validate_sparse_mats(fallback, device, "W_by_delay")}
    raise RuntimeError(
        "W_by_delay_by_comp missing; compile_topology(..., build_sparse_delay_mats=True) before stepping."
    )


def _nonempty_mats_by_comp(
    topology: SynapseTopology, device: Any
) -> dict[Compartment, list[tuple[int, Tensor]]]:
    meta = topology.meta or {}
    cached = meta.get("nonempty_mats_by_comp_csr")
    if isinstance(cached, dict):
        out_csr: dict[Compartment, list[tuple[int, Tensor]]] = {}
        for comp, entries in cached.items():
            if not isinstance(entries, list):
                continue
            valid_csr: list[tuple[int, Tensor]] = []
            for entry in entries:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    continue
                delay, mat = entry
                if mat is None:
                    continue
                if hasattr(mat, "device") and device is not None and mat.device != device:
                    raise RuntimeError(
                        "nonempty_mats_by_comp is on a different device; "
                        "compile_topology(..., build_sparse_delay_mats=True) with the synapse device."
                    )
                valid_csr.append((int(delay), cast(Tensor, mat)))
            out_csr[cast(Compartment, comp)] = valid_csr
        if out_csr:
            return out_csr

    mats_csr = meta.get("W_by_delay_by_comp_csr")
    if isinstance(mats_csr, dict):
        return {
            comp: [(delay, mat) for delay, mat in enumerate(mats) if mat is not None]
            for comp, mats in mats_csr.items()
        }

    cached = meta.get("nonempty_mats_by_comp")
    if isinstance(cached, dict):
        out_coo: dict[Compartment, list[tuple[int, Tensor]]] = {}
        for comp, entries in cached.items():
            if not isinstance(entries, list):
                continue
            valid_coo: list[tuple[int, Tensor]] = []
            for entry in entries:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    continue
                delay, mat = entry
                if mat is None:
                    continue
                if hasattr(mat, "device") and device is not None and mat.device != device:
                    raise RuntimeError(
                        "nonempty_mats_by_comp is on a different device; "
                        "compile_topology(..., build_sparse_delay_mats=True) with the synapse device."
                    )
                valid_coo.append((int(delay), cast(Tensor, mat)))
            out_coo[cast(Compartment, comp)] = valid_coo
        if out_coo:
            return out_coo

    mats_by_comp = _sparse_mats_by_delay_by_comp(topology, device)
    return {
        comp: [(delay, mat) for delay, mat in enumerate(mats) if mat is not None]
        for comp, mats in mats_by_comp.items()
    }


def _fused_sparse_by_comp(
    topology: SynapseTopology,
    device: Any,
    *,
    fused_layout: Literal["auto", "coo", "csr"],
) -> tuple[
    dict[Compartment, Tensor],
    dict[Compartment, Tensor],
    dict[Compartment, int],
    dict[Compartment, int],
    dict[Compartment, Tensor],
    dict[Compartment, Tensor],
]:
    meta = topology.meta or {}
    fused_coo = meta.get("fused_W_by_comp_coo") or meta.get("fused_W_by_comp")
    fused_csr = meta.get("fused_W_by_comp_csr")
    delays = meta.get("fused_W_delays_long_by_comp") or meta.get("fused_W_delays_by_comp")
    n_post_by_comp = meta.get("fused_W_n_post_by_comp")
    d_by_comp_meta = meta.get("fused_D_by_comp") or meta.get("fused_W_n_blocks_by_comp")
    immediate_idx_by_comp = meta.get("fused_immediate_blocks_idx_by_comp")
    delayed_idx_by_comp = meta.get("fused_delayed_blocks_idx_by_comp")
    if not isinstance(delays, dict) or not isinstance(n_post_by_comp, dict):
        return {}, {}, {}, {}, {}, {}
    if not isinstance(immediate_idx_by_comp, dict) or not isinstance(delayed_idx_by_comp, dict):
        raise RuntimeError(
            "fused routing metadata missing; "
            "compile_topology(..., build_sparse_delay_mats=True) to precompute fused routing indices."
        )

    coo_dict = fused_coo if isinstance(fused_coo, dict) else {}
    csr_dict = fused_csr if isinstance(fused_csr, dict) else {}

    d_meta = d_by_comp_meta if isinstance(d_by_comp_meta, dict) else {}
    use_csr_default = fused_layout == "csr"
    if fused_layout == "auto" and (
        (device is not None and getattr(device, "type", None) == "cpu" and csr_dict)
        or (meta.get("fused_layout_preference") == "csr" and csr_dict)
    ):
        use_csr_default = True

    out_fused: dict[Compartment, Tensor] = {}
    out_delays: dict[Compartment, Tensor] = {}
    out_n_post: dict[Compartment, int] = {}
    out_d: dict[Compartment, int] = {}
    out_immediate_idx: dict[Compartment, Tensor] = {}
    out_delayed_idx: dict[Compartment, Tensor] = {}

    all_comps = set(coo_dict) | set(csr_dict)
    for comp in all_comps:
        mat = None
        if fused_layout == "coo":
            mat = coo_dict.get(comp)
            if mat is None:
                mat = csr_dict.get(comp)
        elif fused_layout == "csr":
            mat = csr_dict.get(comp)
            if mat is None:
                mat = coo_dict.get(comp)
        else:
            if use_csr_default:
                mat = csr_dict.get(comp)
                if mat is None:
                    mat = coo_dict.get(comp)
            else:
                mat = coo_dict.get(comp)
                if mat is None:
                    mat = csr_dict.get(comp)
        if mat is None:
            continue
        if hasattr(mat, "device") and device is not None and mat.device != device:
            raise RuntimeError(
                "fused_W_by_comp is on a different device; "
                "compile_topology(..., build_sparse_delay_mats=True) with the synapse device."
            )
        delay_vec = delays.get(comp)
        if delay_vec is None:
            continue
        if hasattr(delay_vec, "device") and device is not None and delay_vec.device != device:
            raise RuntimeError(
                "fused_W_delays_by_comp is on a different device; "
                "compile_topology(..., build_sparse_delay_mats=True) with the synapse device."
            )
        immediate_idx = immediate_idx_by_comp.get(comp)
        delayed_idx = delayed_idx_by_comp.get(comp)
        if immediate_idx is None or delayed_idx is None:
            raise RuntimeError(
                "fused routing metadata missing; "
                "compile_topology(..., build_sparse_delay_mats=True) to precompute fused routing indices."
            )
        if hasattr(immediate_idx, "device") and device is not None and immediate_idx.device != device:
            raise RuntimeError(
                "fused_immediate_blocks_idx_by_comp is on a different device; "
                "compile_topology(..., build_sparse_delay_mats=True) with the synapse device."
            )
        if hasattr(delayed_idx, "device") and device is not None and delayed_idx.device != device:
            raise RuntimeError(
                "fused_delayed_blocks_idx_by_comp is on a different device; "
                "compile_topology(..., build_sparse_delay_mats=True) with the synapse device."
            )
        out_fused[cast(Compartment, comp)] = cast(Tensor, mat)
        out_delays[cast(Compartment, comp)] = cast(Tensor, delay_vec)
        out_immediate_idx[cast(Compartment, comp)] = cast(Tensor, immediate_idx)
        out_delayed_idx[cast(Compartment, comp)] = cast(Tensor, delayed_idx)
        n_post_val = n_post_by_comp.get(comp)
        if n_post_val is None:
            continue
        try:
            out_n_post[cast(Compartment, comp)] = int(n_post_val)
        except Exception:
            continue
        d_val = d_meta.get(comp)
        if d_val is None:
            try:
                d_val = int(delay_vec.numel())
            except Exception:
                d_val = None
        if d_val is not None:
            out_d[cast(Compartment, comp)] = int(d_val)
    return out_fused, out_delays, out_n_post, out_d, out_immediate_idx, out_delayed_idx


def _validate_sparse_mats_by_comp(
    mats: dict[Any, Any], device: Any
) -> dict[Compartment, list[Tensor | None]]:
    out: dict[Compartment, list[Tensor | None]] = {}
    for comp, value in mats.items():
        if not isinstance(value, list):
            continue
        out[cast(Compartment, comp)] = _validate_sparse_mats(value, device, "W_by_delay_by_comp")
    return out


def _validate_sparse_mats(
    mats: list[Any], device: Any, label: str
) -> list[Tensor | None]:
    if mats:
        first = mats[0]
        if hasattr(first, "device") and device is not None and first.device != device:
            raise RuntimeError(
                f"{label} is on a different device; "
                "compile_topology(..., build_sparse_delay_mats=True) with the synapse device."
            )
    return cast(list[Tensor | None], mats)


def _maybe_clamp_weights(
    model: DelayedSparseMatmulSynapse,
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    weights: Tensor,
) -> Tensor:
    if model.params.clamp_min is not None or model.params.clamp_max is not None:
        weights.clamp_(
            min=model.params.clamp_min if model.params.clamp_min is not None else None,
            max=model.params.clamp_max if model.params.clamp_max is not None else None,
        )
        if state.weights is not weights:
            state.weights = weights
        _sync_sparse_values(topology)
    return weights


def _ensure_supported_topology(topology: SynapseTopology) -> None:
    _ = topology


def _validate_backend_config(params: DelayedSparseMatmulParams) -> None:
    if params.backend == "event_driven":
        if params.ring_strategy not in {"dense", "event_bucketed"}:
            raise RuntimeError(f"Unsupported ring_strategy={params.ring_strategy}.")
        return
    if params.ring_strategy != "dense":
        raise RuntimeError(
            f"ring_strategy={params.ring_strategy} requires backend='event_driven'."
        )


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


def _resolve_weights(
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    ctx: StepContext,
) -> Tensor:
    weights = topology.weights
    if weights is None:
        weights = state.weights
    elif weights is not state.weights:
        state.weights = weights
    return weights


def _ensure_pre_activity_buf(
    state: DelayedSparseMatmulState,
    n_pre: int,
    device: Any,
    dtype: Any,
) -> Tensor:
    torch = require_torch()
    buf = state.pre_activity_buf
    if buf is None or buf.shape != (n_pre,) or buf.device != device or buf.dtype != dtype:
        buf = torch.zeros((n_pre,), device=device, dtype=dtype)
        state.pre_activity_buf = buf
    return buf


def _fill_pre_activity_buf(
    state: DelayedSparseMatmulState,
    pre_spikes: Tensor,
    weights: Tensor,
    device: Any,
) -> Tensor:
    buf = _ensure_pre_activity_buf(state, int(pre_spikes.shape[0]), device, weights.dtype)
    try:
        buf.copy_(pre_spikes)
    except Exception:
        buf.zero_()
        buf.masked_fill_(pre_spikes, 1.0)
    return buf


def _sync_sparse_values(topology: SynapseTopology) -> None:
    (
        values_by_comp,
        edge_bucket_comp,
        edge_bucket_delay,
        edge_bucket_pos,
        edge_scale,
        fused_values_by_comp,
        edge_bucket_fused_pos,
    ) = _require_sparse_update_meta(topology)
    weights = topology.weights
    if weights is None:
        raise RuntimeError("Topology weights missing; cannot sync sparse values.")
    comp_ids = edge_bucket_comp
    delay_ids = edge_bucket_delay
    pos_ids = edge_bucket_pos
    scale_vals = edge_scale

    _apply_sparse_value_updates(
        values_by_comp=values_by_comp,
        comp_ids=comp_ids,
        delay_ids=delay_ids,
        pos_ids=pos_ids,
        delta=weights,
        scale_vals=scale_vals,
        replace=True,
        fused_values_by_comp=fused_values_by_comp,
        fused_pos_ids=edge_bucket_fused_pos,
    )


def _apply_sparse_value_updates(
    *,
    values_by_comp: Mapping[Compartment, list[Tensor | None]] | None,
    comp_ids: Tensor,
    delay_ids: Tensor,
    pos_ids: Tensor,
    delta: Tensor,
    scale_vals: Tensor | None,
    replace: bool = False,
    fused_values_by_comp: Mapping[Compartment, Tensor] | None = None,
    fused_pos_ids: Tensor | None = None,
) -> None:
    comp_order = tuple(Compartment)
    if fused_values_by_comp is not None and fused_pos_ids is not None:
        for comp, fused_values in fused_values_by_comp.items():
            if comp not in comp_order:
                continue
            comp_id = comp_order.index(comp)
            comp_mask = comp_ids == comp_id
            fused_mask = comp_mask & (fused_pos_ids >= 0)
            selected = fused_mask.nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            pos = fused_pos_ids.index_select(0, selected)
            vals = delta.index_select(0, selected)
            if scale_vals is not None:
                vals = vals * scale_vals.index_select(0, selected)
            if replace:
                fused_values.index_copy_(0, pos, vals)
            else:
                fused_values.index_add_(0, pos, vals)

    if values_by_comp is None:
        return
    for comp, values_by_delay in values_by_comp.items():
        if comp not in comp_order:
            continue
        comp_id = comp_order.index(comp)
        comp_mask = comp_ids == comp_id
        for delay, values in enumerate(values_by_delay):
            if values is None:
                continue
            mask = comp_mask & (delay_ids == delay)
            selected = mask.nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            pos = pos_ids.index_select(0, selected)
            vals = delta.index_select(0, selected)
            if scale_vals is not None:
                vals = vals * scale_vals.index_select(0, selected)
            if replace:
                values.index_copy_(0, pos, vals)
            else:
                values.index_add_(0, pos, vals)


def _require_sparse_update_meta(
    topology: SynapseTopology,
) -> tuple[
    Mapping[Compartment, list[Tensor | None]] | None,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    dict[Compartment, Tensor] | None,
    Tensor | None,
]:
    meta = topology.meta or {}
    values_by_comp = meta.get("values_by_comp")
    edge_bucket_comp = meta.get("edge_bucket_comp")
    edge_bucket_delay = meta.get("edge_bucket_delay")
    edge_bucket_pos = meta.get("edge_bucket_pos")
    edge_scale = meta.get("edge_scale")
    fused_by_comp = meta.get("fused_W_by_comp")
    edge_bucket_fused_pos = meta.get("edge_bucket_fused_pos")
    has_values = isinstance(values_by_comp, dict)
    has_fused = isinstance(fused_by_comp, dict) and edge_bucket_fused_pos is not None
    if not has_values and not has_fused:
        raise RuntimeError(
            "values_by_comp missing; compile_topology(..., build_sparse_delay_mats=True) "
            "must be called before apply_weight_updates."
        )
    if edge_bucket_comp is None or edge_bucket_delay is None or edge_bucket_pos is None:
        raise RuntimeError(
            "Edge bucket mappings missing; compile_topology(..., build_bucket_edge_mapping=True) "
            "must be called before apply_weight_updates."
        )
    fused_values_by_comp = None
    if isinstance(fused_by_comp, dict) and edge_bucket_fused_pos is not None:
        fused_values_by_comp = {
            cast(Compartment, comp): cast(Tensor, mat).values()
            for comp, mat in fused_by_comp.items()
            if mat is not None
        }
    return (
        cast(Mapping[Compartment, list[Tensor | None]] | None, values_by_comp) if has_values else None,
        cast(Tensor, edge_bucket_comp),
        cast(Tensor, edge_bucket_delay),
        cast(Tensor, edge_bucket_pos),
        cast(Tensor | None, edge_scale),
        fused_values_by_comp,
        cast(Tensor | None, edge_bucket_fused_pos),
    )


__all__ = [
    "DelayedSparseMatmulParams",
    "DelayedSparseMatmulState",
    "DelayedSparseMatmulSynapse",
]
