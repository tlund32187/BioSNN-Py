"""Tensor typing utilities (torch optional)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import torch  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    Tensor: TypeAlias = "torch.Tensor"
else:
    Tensor: TypeAlias = Any

__all__ = ["Tensor"]
