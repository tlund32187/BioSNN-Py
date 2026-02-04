"""Deprecated: use biosnn.core.torch_utils instead."""

from __future__ import annotations

from biosnn.core.torch_utils import _resolve_dtype, require_torch, resolve_device_dtype

__all__ = ["require_torch", "resolve_device_dtype", "_resolve_dtype"]
