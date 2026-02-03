"""Delay utilities for axonal conduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.tensor import Tensor


def _require_torch() -> Any:
    try:
        import importlib

        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Torch is required for delay computations.") from exc


@dataclass(frozen=True, slots=True)
class DelayParams:
    """Parameters for converting distances to delay steps."""

    base_velocity: float = 1.0  # meters/second
    myelin_scale: float = 1.0
    min_delay: float = 0.0
    max_delay: float | None = None
    use_ceil: bool = True


def compute_delay_steps(
    pre_pos: Tensor,
    post_pos: Tensor,
    pre_idx: Tensor,
    post_idx: Tensor,
    *,
    dt: float,
    myelin: Tensor | None = None,
    params: DelayParams | None = None,
) -> Tensor:
    """Compute per-edge delay steps from positions and myelin factors.

    pre_pos/post_pos: [N, 3] coordinates in meters.
    pre_idx/post_idx: [E] edge index arrays.
    myelin: optional [E] factor (0..1 typical) scaling conduction speed.
    """

    torch = _require_torch()
    p = params or DelayParams()
    if dt <= 0:
        raise ValueError("dt must be positive")

    if pre_pos.dim() != 2 or pre_pos.shape[1] != 3:
        raise ValueError("pre_pos must have shape [Npre, 3]")
    if post_pos.dim() != 2 or post_pos.shape[1] != 3:
        raise ValueError("post_pos must have shape [Npost, 3]")

    pre = pre_pos.index_select(0, pre_idx.to(device=pre_pos.device, dtype=torch.long))
    post = post_pos.index_select(0, post_idx.to(device=post_pos.device, dtype=torch.long))
    dist = torch.linalg.norm(post - pre, dim=1)

    velocity = dist.new_tensor(p.base_velocity)
    if myelin is not None:
        if myelin.shape != dist.shape:
            raise ValueError("myelin must have shape [E]")
        velocity = velocity * (1.0 + p.myelin_scale * myelin)

    delay_sec = dist / velocity
    if p.max_delay is not None:
        delay_sec = torch.clamp(delay_sec, min=p.min_delay, max=p.max_delay)
    else:
        delay_sec = torch.clamp(delay_sec, min=p.min_delay)

    steps = delay_sec / dt
    steps = torch.ceil(steps) if p.use_ceil else torch.round(steps)
    return cast(Tensor, steps.to(dtype=torch.int32))
