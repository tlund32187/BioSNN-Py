"""Scalar conversion helpers for monitors."""

from __future__ import annotations

from biosnn.contracts.monitors import Scalar
from biosnn.core.torch_utils import require_torch


def scalar_to_float(x: Scalar) -> float:
    """Convert a scalar (float/int/torch.Tensor) to Python float.

    Note: This is intended for monitor/sink usage only.
    """

    if isinstance(x, (float, int)):
        return float(x)
    torch = require_torch()
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"Expected a 0-d or 1-element tensor scalar, got shape {tuple(x.shape)}")
        return float(x.detach().float().cpu().item())
    raise TypeError(f"Unsupported scalar type: {type(x)!r}")


__all__ = ["scalar_to_float"]
