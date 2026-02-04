"""Base scaffolding for neuron model implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(frozen=True, slots=True)
class StateTensorSpec:
    """Expected tensor shape/dtype spec for validation."""

    shape: tuple[int | None, ...]
    dtype: str  # "float", "bool", or "int"


class NeuronModelBase(INeuronModel):
    """Optional base class for guided neuron model implementations."""

    name = "base"
    compartments: frozenset[Compartment]

    def __init__(self) -> None:
        self.compartments = frozenset(_coerce_compartments(self.required_compartments()))

    # ---- Hooks for subclasses -------------------------------------------------
    def required_compartments(self) -> tuple[str, ...]:
        return ("soma",)

    def state_tensors_spec(self) -> Mapping[str, StateTensorSpec]:
        return {}

    def init_state_tensors(self, n: int, *, device: Any, dtype: Any) -> Any:
        raise NotImplementedError

    def step_state(
        self,
        state: Any,
        drive: Mapping[Compartment, Tensor],
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[Any, NeuronStepResult]:
        raise NotImplementedError

    # ---- INeuronModel ---------------------------------------------------------
    def init_state(self, n: int, *, ctx: StepContext) -> Any:
        device, dtype = resolve_device_dtype(ctx)
        state = self.init_state_tensors(n, device=device, dtype=dtype)
        self._validate_state(state, n=n, device=device, dtype=dtype)
        return state

    def reset_state(self, state: Any, *, ctx: StepContext, indices: Tensor | None = None) -> Any:
        n = _infer_n(self.state_tensors(state))
        fresh = self.init_state(n, ctx=ctx)
        state_tensors = self.state_tensors(state)
        fresh_tensors = self.state_tensors(fresh)
        if indices is None:
            for key, tensor in state_tensors.items():
                tensor.copy_(fresh_tensors[key])
        else:
            for key, tensor in state_tensors.items():
                tensor.index_copy_(0, indices, fresh_tensors[key].index_select(0, indices))
        return state

    def step(
        self,
        state: Any,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[Any, NeuronStepResult]:
        self._validate_inputs(inputs.drive, state)
        state, result = self.step_state(state, inputs.drive, dt=dt, t=t, ctx=ctx)
        self._validate_result(result, state)
        return state, result

    # ---- Validation helpers ---------------------------------------------------
    def _validate_state(self, state: Any, *, n: int, device: Any, dtype: Any) -> None:
        specs = self.state_tensors_spec()
        tensors = self.state_tensors(state)
        if specs:
            for name, spec in specs.items():
                if name not in tensors:
                    raise ValueError(f"State missing tensor '{name}'")
                _validate_tensor(tensors[name], spec, n=n, device=device, dtype=dtype)

    def _validate_inputs(self, drive: Mapping[Compartment, Tensor], state: Any) -> None:
        tensors = self.state_tensors(state)
        n = _infer_n(tensors)
        device, dtype = _infer_device_dtype(tensors)
        for comp in self.compartments:
            if comp not in drive:
                raise ValueError(f"Missing drive for compartment '{comp}'")
            _validate_tensor_shape(drive[comp], (n,), name=f"drive[{comp}]", device=device)
            if drive[comp].dtype != dtype:
                raise ValueError(f"drive[{comp}] dtype mismatch: {drive[comp].dtype} vs {dtype}")

    def _validate_result(self, result: NeuronStepResult, state: Any) -> None:
        tensors = self.state_tensors(state)
        n = _infer_n(tensors)
        device, _ = _infer_device_dtype(tensors)
        if result.spikes.shape != (n,):
            raise ValueError(f"spikes shape must be ({n},), got {tuple(result.spikes.shape)}")
        torch = require_torch()
        if result.spikes.dtype is not torch.bool:
            raise ValueError(f"spikes must be bool, got {result.spikes.dtype}")
        if result.spikes.device != device:
            raise ValueError("spikes device does not match state device")


def _coerce_compartments(values: tuple[str, ...]) -> tuple[Compartment, ...]:
    comps: list[Compartment] = []
    for value in values:
        try:
            comps.append(Compartment(value))
        except Exception as exc:
            raise ValueError(f"Unknown compartment '{value}'") from exc
    return tuple(comps)


def _infer_n(tensors: Mapping[str, Tensor]) -> int:
    for tensor in tensors.values():
        return int(tensor.shape[0])
    raise ValueError("Unable to infer n from empty state tensors")


def _infer_device_dtype(tensors: Mapping[str, Tensor]) -> tuple[Any, Any]:
    for tensor in tensors.values():
        return tensor.device, tensor.dtype
    raise ValueError("Unable to infer device/dtype from empty state tensors")


def _validate_tensor(
    tensor: Tensor,
    spec: StateTensorSpec,
    *,
    n: int,
    device: Any,
    dtype: Any,
) -> None:
    _validate_tensor_shape(tensor, spec.shape, name="state", device=device)
    torch = require_torch()
    if spec.dtype == "float":
        if not torch.is_floating_point(tensor) or tensor.dtype != dtype:
            raise ValueError("State tensor expected float dtype")
    elif spec.dtype == "bool":
        if tensor.dtype is not torch.bool:
            raise ValueError("State tensor expected bool dtype")
    elif spec.dtype == "int" and (torch.is_floating_point(tensor) or tensor.dtype is torch.bool):
        raise ValueError("State tensor expected integer dtype")


def _validate_tensor_shape(
    tensor: Tensor,
    shape: tuple[int | None, ...],
    *,
    name: str,
    device: Any,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} device mismatch: {tensor.device} vs {device}")
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} expected {len(shape)} dims, got {tensor.ndim}")
    for dim, expected in zip(tensor.shape, shape, strict=True):
        if expected is not None and dim != expected:
            raise ValueError(f"{name} expected dim {expected}, got {dim}")


__all__ = ["NeuronModelBase", "StateTensorSpec"]
