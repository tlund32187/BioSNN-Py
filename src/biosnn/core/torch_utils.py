"""Torch utilities shared across the codebase."""

from __future__ import annotations

import importlib
from typing import Any

from biosnn.contracts.neurons import StepContext


def require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Torch is required to use BioSNN torch components.") from exc


def resolve_device_dtype(ctx: StepContext | None) -> tuple[Any, Any]:
    t = require_torch()
    device = t.device(ctx.device) if ctx and ctx.device else None
    dtype = _resolve_dtype(t, ctx.dtype) if ctx and ctx.dtype else t.get_default_dtype()
    return device, dtype


def _resolve_dtype(t: Any, dtype_str: str | None) -> Any:
    if dtype_str is None:
        return t.get_default_dtype()
    name = dtype_str
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    if not hasattr(t, name):
        raise ValueError(f"Unknown torch dtype: {dtype_str}")
    return getattr(t, name)


__all__ = ["require_torch", "resolve_device_dtype", "_resolve_dtype"]
