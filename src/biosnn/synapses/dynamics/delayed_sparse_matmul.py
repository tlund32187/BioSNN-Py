"""Delayed synapse backend using sparse matmul per delay bucket."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.neurons import Compartment, StepContext, Tensor
from biosnn.contracts.synapses import (
    ISynapseModel,
    ISynapseModelInplace,
    ReceptorKind,
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)
from biosnn.core.torch_utils import require_torch, resolve_device_dtype

_COMPARTMENT_ORDER = tuple(Compartment)
_COMPARTMENT_TO_ID = {comp: idx for idx, comp in enumerate(_COMPARTMENT_ORDER)}


@dataclass(frozen=True, slots=True)
class DelayedSparseMatmulParams:
    init_weight: float = 1e-9
    clamp_min: float | None = None
    clamp_max: float | None = None
    receptor_scale: Mapping[ReceptorKind, float] | None = None


@dataclass(slots=True)
class DelayedSparseMatmulState:
    weights: Tensor
    post_ring: dict[Compartment, Tensor] | None
    post_out: dict[Compartment, Tensor] | None
    cursor: int


class DelayedSparseMatmulSynapse(ISynapseModel, ISynapseModelInplace):
    """Sparse matmul delayed synapse (no per-edge enumeration per step).

    Note: if learning updates weights dynamically, call apply_weight_updates to keep
    the compiled sparse matrices in sync with the topology weights.
    """

    name = "delayed_sparse_matmul"

    def __init__(self, params: DelayedSparseMatmulParams | None = None) -> None:
        self.params = params or DelayedSparseMatmulParams()

    def init_state(self, e: int, *, ctx: StepContext) -> DelayedSparseMatmulState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        weights = torch.full((e,), self.params.init_weight, device=device, dtype=dtype)
        return DelayedSparseMatmulState(
            weights=weights,
            post_ring=None,
            post_out=None,
            cursor=0,
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
            state.cursor = 0
            return state
        state.weights[edge_indices] = self.params.init_weight
        if state.post_ring is not None:
            for ring in state.post_ring.values():
                ring[:, :] = 0.0
        if state.post_out is not None:
            for out in state.post_out.values():
                out[:] = 0.0
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

            post_drive = _step_sparse_matmul(
                self,
                state,
                topology,
                pre_spikes,
                device,
                weights,
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
            _step_sparse_matmul_into(
                self,
                state,
                topology,
                pre_spikes,
                device,
                weights,
                out_drive,
            )

    def state_tensors(self, state: DelayedSparseMatmulState) -> Mapping[str, Tensor]:
        return {"weights": state.weights}

    def compilation_requirements(self) -> Mapping[str, bool]:
        return {
            "needs_edges_by_delay": False,
            "needs_pre_adjacency": False,
            "needs_sparse_delay_mats": True,
            "needs_bucket_edge_mapping": True,
        }

    def apply_weight_updates(
        self,
        topology: SynapseTopology,
        active_edges: Tensor,
        d_weights: Tensor,
    ) -> None:
        weights = topology.weights
        if weights is None:
            raise RuntimeError("DelayedSparseMatmulSynapse requires topology.weights for updates")

        update_all = d_weights.numel() == weights.numel()
        if update_all:
            weights.add_(d_weights)
            edge_ids = None
            delta = d_weights
        elif d_weights.numel() == active_edges.numel():
            weights.index_add_(0, active_edges, d_weights)
            edge_ids = active_edges
            delta = d_weights
        else:
            raise RuntimeError(
                "Sparse matmul weight update shape mismatch: "
                f"d_weights={tuple(d_weights.shape)} active_edges={tuple(active_edges.shape)} "
                f"weights={tuple(weights.shape)}"
            )

        values_by_comp, edge_bucket_comp, edge_bucket_delay, edge_bucket_pos, edge_scale = (
            _require_sparse_update_meta(topology)
        )
        if edge_ids is None:
            comp_ids = edge_bucket_comp
            delay_ids = edge_bucket_delay
            pos_ids = edge_bucket_pos
            scale_vals = edge_scale
        else:
            comp_ids = edge_bucket_comp.index_select(0, edge_ids)
            delay_ids = edge_bucket_delay.index_select(0, edge_ids)
            pos_ids = edge_bucket_pos.index_select(0, edge_ids)
            scale_vals = edge_scale.index_select(0, edge_ids) if edge_scale is not None else None

        comp_order = tuple(Compartment)
        for comp, values_by_delay in values_by_comp.items():
            if comp not in comp_order:
                continue
            comp_id = comp_order.index(comp)
            for delay, values in enumerate(values_by_delay):
                if values is None:
                    continue
                mask = (comp_ids == comp_id) & (delay_ids == delay)
                selected = mask.nonzero(as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                pos = pos_ids.index_select(0, selected)
                delta_vals = delta.index_select(0, selected)
                if scale_vals is not None:
                    delta_vals = delta_vals * scale_vals.index_select(0, selected)
                values.index_add_(0, pos, delta_vals)


def _step_sparse_matmul(
    model: DelayedSparseMatmulSynapse,
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    pre_spikes: Tensor,
    device: Any,
    weights: Tensor,
) -> Mapping[Compartment, Tensor]:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
    post_ring, post_out = _ensure_post_ring(state, topology, depth, n_post, device, weights.dtype)

    cursor = state.cursor % depth
    post_drive: dict[Compartment, Tensor] = {}
    for comp, ring in post_ring.items():
        slot = ring[cursor]
        out = post_out[comp]
        out.copy_(slot)
        slot.zero_()
        post_drive[comp] = out

    weights = _maybe_clamp_weights(model, state, weights)
    pre_activity = (
        pre_spikes
        if pre_spikes.dtype == weights.dtype
        else pre_spikes.to(dtype=weights.dtype)
    )

    mats_by_comp = _sparse_mats_by_delay_by_comp(topology, device)
    for comp, mats in mats_by_comp.items():
        for delay, mat in enumerate(mats):
            if mat is None:
                continue
            contrib = torch.sparse.mm(mat, pre_activity.unsqueeze(1)).squeeze(1)
            target_row = (cursor + delay) % depth
            if delay == 0:
                post_drive[comp].add_(contrib)
            else:
                post_ring[comp][target_row].add_(contrib)

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
) -> None:
    torch = require_torch()
    max_delay = _max_delay_steps(topology)
    depth = max_delay + 1
    n_post = _infer_n_post(topology)
    post_ring, _ = _ensure_post_ring(state, topology, depth, n_post, device, weights.dtype)

    cursor = state.cursor % depth
    for comp, ring in post_ring.items():
        slot = ring[cursor]
        if comp not in out_drive:
            raise KeyError(f"Drive accumulator missing compartment {comp}")
        out_drive[comp].add_(slot)
        slot.zero_()

    weights = _maybe_clamp_weights(model, state, weights)
    pre_activity = (
        pre_spikes
        if pre_spikes.dtype == weights.dtype
        else pre_spikes.to(dtype=weights.dtype)
    )

    mats_by_comp = _sparse_mats_by_delay_by_comp(topology, device)
    for comp, mats in mats_by_comp.items():
        if comp not in out_drive:
            raise KeyError(f"Drive accumulator missing compartment {comp}")
        for delay, mat in enumerate(mats):
            if mat is None:
                continue
            contrib = torch.sparse.mm(mat, pre_activity.unsqueeze(1)).squeeze(1)
            target_row = (cursor + delay) % depth
            if delay == 0:
                out_drive[comp].add_(contrib)
            else:
                post_ring[comp][target_row].add_(contrib)

    state.cursor = (cursor + 1) % depth


def _ensure_post_ring(
    state: DelayedSparseMatmulState,
    topology: SynapseTopology,
    depth: int,
    n_post: int,
    device: Any,
    dtype: Any,
) -> tuple[dict[Compartment, Tensor], dict[Compartment, Tensor]]:
    torch = require_torch()
    if state.post_ring is None:
        state.post_ring = {}
    if state.post_out is None:
        state.post_out = {}
    for comp in _ring_compartments(topology):
        ring = state.post_ring.get(comp)
        if ring is None or ring.shape != (depth, n_post) or ring.device != device or ring.dtype != dtype:
            ring = torch.zeros((depth, n_post), device=device, dtype=dtype)
            state.post_ring[comp] = ring
        out = state.post_out.get(comp)
        if out is None or out.shape != (n_post,) or out.device != device or out.dtype != dtype:
            out = torch.zeros((n_post,), device=device, dtype=dtype)
            state.post_out[comp] = out
    return state.post_ring, state.post_out


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
    weights: Tensor,
) -> Tensor:
    if model.params.clamp_min is not None or model.params.clamp_max is not None:
        weights = weights.clamp(
            min=model.params.clamp_min if model.params.clamp_min is not None else None,
            max=model.params.clamp_max if model.params.clamp_max is not None else None,
        )
        state.weights.copy_(weights)
    return weights


def _ensure_supported_topology(topology: SynapseTopology) -> None:
    _ = topology


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


def _require_sparse_update_meta(
    topology: SynapseTopology,
) -> tuple[
    dict[Compartment, list[Tensor | None]],
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
]:
    meta = topology.meta or {}
    values_by_comp = meta.get("values_by_comp")
    edge_bucket_comp = meta.get("edge_bucket_comp")
    edge_bucket_delay = meta.get("edge_bucket_delay")
    edge_bucket_pos = meta.get("edge_bucket_pos")
    edge_scale = meta.get("edge_scale")
    if not isinstance(values_by_comp, dict):
        raise RuntimeError(
            "values_by_comp missing; compile_topology(..., build_sparse_delay_mats=True) "
            "must be called before apply_weight_updates."
        )
    if edge_bucket_comp is None or edge_bucket_delay is None or edge_bucket_pos is None:
        raise RuntimeError(
            "Edge bucket mappings missing; compile_topology(..., build_bucket_edge_mapping=True) "
            "must be called before apply_weight_updates."
        )
    return (
        cast(dict[Compartment, list[Tensor | None]], values_by_comp),
        cast(Tensor, edge_bucket_comp),
        cast(Tensor, edge_bucket_delay),
        cast(Tensor, edge_bucket_pos),
        cast(Tensor | None, edge_scale),
    )


__all__ = [
    "DelayedSparseMatmulParams",
    "DelayedSparseMatmulState",
    "DelayedSparseMatmulSynapse",
]
