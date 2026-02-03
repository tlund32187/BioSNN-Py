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


@dataclass(slots=True)
class DelayedCurrentState:
    """Synapse state (weights + optional delay buffer)."""

    weights: Tensor  # shape [E]
    delay_buffer: Tensor | None  # shape [D, E]
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
        return DelayedCurrentState(weights=weights, delay_buffer=None, cursor=0)

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
            state.cursor = 0
            return state
        state.weights[edge_indices] = self.params.init_weight
        if state.delay_buffer is not None:
            state.delay_buffer[:, edge_indices] = 0.0
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
            edge_spikes = pre_spikes.index_select(0, pre_idx)

            delay_steps_opt = topology.delay_steps
            if delay_steps_opt is not None:
                delay_steps = _as_index(delay_steps_opt, device, name="delay_steps")
                max_delay = int(delay_steps.max().item()) if delay_steps.numel() else 0
            else:
                delay_steps = None
                max_delay = 0

            if max_delay > 0 and delay_steps is not None:
                depth = max_delay + 1
                if state.delay_buffer is None or state.delay_buffer.shape != (depth, pre_idx.numel()):
                    state.delay_buffer = torch.zeros(
                        (depth, pre_idx.numel()), device=device, dtype=weights.dtype
                    )
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
                    state.delay_buffer[target_rows[~immediate_mask], edge_idx[~immediate_mask]] = (
                        edge_spikes[~immediate_mask]
                    )
                state.cursor = (cursor + 1) % depth
            else:
                delayed = edge_spikes

            if self.params.clamp_min is not None or self.params.clamp_max is not None:
                weights = weights.clamp(
                    min=self.params.clamp_min if self.params.clamp_min is not None else None,
                    max=self.params.clamp_max if self.params.clamp_max is not None else None,
                )
                state.weights.copy_(weights)

            edge_current = delayed * weights
            edge_current = _apply_receptor_scale(self, topology, edge_current)

            post_drive = _accumulate_post_drive(
                edge_current,
                post_idx,
                topology,
                device,
                weights.dtype,
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
    receptor_ids = receptor_ids.to(device=edge_current.device, dtype=torch.long)
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

    comp_ids = target_compartments.to(device=device, dtype=torch.long)
    for comp, comp_id in _COMPARTMENT_TO_ID.items():
        mask = comp_ids == comp_id
        if mask.any():
            out = torch.zeros((n_post,), device=device, dtype=dtype)
            out.scatter_add_(0, post_idx[mask], edge_current[mask])
            post_drive[comp] = out
    return post_drive


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
        if require_on_device and tensor.device != like.device:
            raise ValueError(f"{name} must be on device {like.device}, got {tensor.device}")
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    if validate_shapes and expected_len is not None:
        _ensure_1d_len(tensor, expected_len, name)
    return tensor


def _as_index(tensor: Tensor, device: Any, *, name: str) -> Tensor:
    torch = require_torch()
    if tensor.device != device or tensor.dtype != torch.long:
        tensor = tensor.to(device=device, dtype=torch.long)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got {tuple(tensor.shape)}")
    return tensor


def _ensure_1d_len(tensor: Tensor, n: int, name: str) -> None:
    if tensor.dim() != 1 or tensor.shape[0] != n:
        raise ValueError(f"{name} must have shape [{n}], got {tuple(tensor.shape)}")


def _infer_n_pre(topology: SynapseTopology) -> int | None:
    if topology.meta and "n_pre" in topology.meta:
        return int(topology.meta["n_pre"])
    if topology.pre_idx.numel():
        return int(topology.pre_idx.max().item()) + 1
    return None


def _infer_n_post(topology: SynapseTopology) -> int:
    if topology.meta and "n_post" in topology.meta:
        return int(topology.meta["n_post"])
    if topology.post_idx.numel():
        return int(topology.post_idx.max().item()) + 1
    return 0


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
