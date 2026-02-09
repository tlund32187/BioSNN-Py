"""Truth-table datasets and samplers for logic-gate tasks."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from biosnn.core.torch_utils import require_torch


class LogicGate(StrEnum):
    AND = "and"
    OR = "or"
    XOR = "xor"
    NAND = "nand"
    NOR = "nor"
    XNOR = "xnor"


SamplingMethod = Literal["sequential", "random_balanced"]

_TRUTH_TABLE_INPUTS = (
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
)

_TARGET_ROWS: dict[LogicGate, tuple[float, float, float, float]] = {
    LogicGate.AND: (0.0, 0.0, 0.0, 1.0),
    LogicGate.OR: (0.0, 1.0, 1.0, 1.0),
    LogicGate.XOR: (0.0, 1.0, 1.0, 0.0),
    LogicGate.NAND: (1.0, 1.0, 1.0, 0.0),
    LogicGate.NOR: (1.0, 0.0, 0.0, 0.0),
    LogicGate.XNOR: (1.0, 0.0, 0.0, 1.0),
}


def coerce_gate(gate: LogicGate | str) -> LogicGate:
    if isinstance(gate, LogicGate):
        return gate
    try:
        return LogicGate(gate.strip().lower())
    except Exception as exc:
        supported = ", ".join(member.name for member in LogicGate)
        raise ValueError(f"Unsupported logic gate '{gate}'. Supported: {supported}") from exc


def make_truth_table(
    gate: LogicGate | str,
    *,
    device: str | None = None,
    dtype: str | None = "float32",
) -> tuple[Any, Any]:
    """Return canonical logic-gate truth table tensors.

    Returns:
        inputs: tensor with shape [4, 2]
        targets: tensor with shape [4, 1]
    """

    torch = require_torch()
    gate_enum = coerce_gate(gate)
    device_obj = torch.device(device) if device is not None else None
    dtype_obj = _resolve_dtype(torch, dtype)

    inputs = torch.tensor(_TRUTH_TABLE_INPUTS, device=device_obj, dtype=dtype_obj)
    targets = torch.tensor(
        _TARGET_ROWS[gate_enum],
        device=device_obj,
        dtype=dtype_obj,
    ).unsqueeze(1)
    return inputs, targets


def sample_case_indices(
    num_trials: int,
    *,
    method: SamplingMethod = "sequential",
    generator: Any | None = None,
    device: str | None = "cpu",
) -> Any:
    """Sample truth-table row indices in ``[0, 4)``."""

    mode = method.strip().lower()
    if mode == "sequential":
        return sample_case_indices_sequential(num_trials=num_trials, device=device)
    if mode == "random_balanced":
        return sample_case_indices_random_balanced(
            num_trials=num_trials,
            generator=generator,
            device=device,
        )
    raise ValueError(f"Unknown sampling method '{method}'. Use sequential or random_balanced.")


def sample_case_indices_sequential(num_trials: int, *, device: str | None = "cpu") -> Any:
    torch = require_torch()
    if num_trials < 0:
        raise ValueError("num_trials must be >= 0")
    device_obj = torch.device(device) if device is not None else None
    indices = torch.arange(num_trials, device=device_obj, dtype=torch.long)
    if num_trials > 0:
        indices.remainder_(4)
    return indices


def sample_case_indices_random_balanced(
    num_trials: int,
    *,
    generator: Any | None = None,
    device: str | None = "cpu",
) -> Any:
    torch = require_torch()
    if num_trials < 0:
        raise ValueError("num_trials must be >= 0")
    sampled = torch.randint(0, 4, (num_trials,), generator=generator, device="cpu", dtype=torch.long)
    if device is None or str(device).lower() == "cpu":
        return sampled
    return sampled.to(torch.device(device))


def _resolve_dtype(torch: Any, dtype: str | Any | None) -> Any:
    if dtype is None:
        return torch.get_default_dtype()
    if isinstance(dtype, str):
        name = dtype.split(".", 1)[-1]
        if hasattr(torch, name):
            return getattr(torch, name)
        raise ValueError(f"Unknown torch dtype '{dtype}'")
    return dtype


__all__ = [
    "LogicGate",
    "SamplingMethod",
    "coerce_gate",
    "make_truth_table",
    "sample_case_indices",
    "sample_case_indices_random_balanced",
    "sample_case_indices_sequential",
]
