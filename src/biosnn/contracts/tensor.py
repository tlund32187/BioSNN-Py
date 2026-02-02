"""Tensor typing utilities (torch optional)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    type Tensor = torch.Tensor
else:
    type Tensor = Any

__all__ = ["Tensor"]
