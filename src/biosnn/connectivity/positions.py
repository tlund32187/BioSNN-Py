"""Deterministic population position generation utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from biosnn.core.torch_utils import require_torch
from biosnn.simulation.network.specs import NeuronPosition, PopulationFrame


def generate_positions(
    frame: PopulationFrame | Mapping[str, Any],
    n: int,
) -> list[NeuronPosition]:
    """Return CPU-side positions as ``NeuronPosition`` records."""

    frame_obj = coerce_population_frame(frame)
    points = positions_tensor(frame_obj, n, device="cpu", dtype="float32")
    if int(points.numel()) == 0:
        return []
    rows = points.tolist()
    return [NeuronPosition(float(row[0]), float(row[1]), float(row[2])) for row in rows]


def positions_tensor(
    frame: PopulationFrame | Mapping[str, Any],
    n: int,
    device: Any,
    dtype: Any,
) -> Any:
    """Generate positions directly as a tensor on the target device."""

    torch = require_torch()
    frame_obj = coerce_population_frame(frame)
    n = int(n)
    device_obj = torch.device(device) if device is not None else None
    dtype_obj = _resolve_dtype(torch, dtype)
    if n <= 0:
        return torch.empty((0, 3), device=device_obj, dtype=dtype_obj)

    origin = torch.tensor(frame_obj.origin, device=device_obj, dtype=dtype_obj)
    extent = torch.tensor(frame_obj.extent, device=device_obj, dtype=dtype_obj)
    layout = frame_obj.layout

    if layout == "line":
        y_vals = _linspace_tensor(origin[1], origin[1] + extent[1], n, device=device_obj, dtype=dtype_obj)
        x_vals = torch.full((n,), origin[0] + 0.5 * extent[0], device=device_obj, dtype=dtype_obj)
        z_vals = torch.full((n,), origin[2] + 0.5 * extent[2], device=device_obj, dtype=dtype_obj)
        return torch.stack((x_vals, y_vals, z_vals), dim=1)

    if layout == "grid":
        cols, rows = _grid_dims(n, float(extent[0].item()), float(extent[1].item()))
        xs = _linspace_tensor(origin[0], origin[0] + extent[0], cols, device=device_obj, dtype=dtype_obj)
        ys = _linspace_tensor(origin[1], origin[1] + extent[1], rows, device=device_obj, dtype=dtype_obj)
        grid_x = xs.repeat(rows)
        grid_y = ys.repeat_interleave(cols)
        z_val = origin[2] + 0.5 * extent[2]
        grid_z = torch.full((grid_x.shape[0],), z_val, device=device_obj, dtype=dtype_obj)
        points = torch.stack((grid_x, grid_y, grid_z), dim=1)
        return points[:n]

    if layout == "ring":
        center_x = origin[0] + 0.5 * extent[0]
        center_y = origin[1] + 0.5 * extent[1]
        center_z = origin[2] + 0.5 * extent[2]
        radius = 0.5 * min(abs(float(extent[0].item())), abs(float(extent[1].item())))
        if radius == 0.0:
            center = torch.tensor((center_x, center_y, center_z), device=device_obj, dtype=dtype_obj)
            return center.unsqueeze(0).expand(n, -1).clone()

        angles = torch.arange(n, device=device_obj, dtype=dtype_obj)
        angles = angles * (2.0 * math.pi / max(n, 1))
        x_vals = center_x + radius * torch.cos(angles)
        y_vals = center_y + radius * torch.sin(angles)
        z_vals = torch.full((n,), center_z, device=device_obj, dtype=dtype_obj)
        return torch.stack((x_vals, y_vals, z_vals), dim=1)

    if layout == "random":
        if frame_obj.seed is None:
            noise = torch.rand((n, 3), device=device_obj, dtype=dtype_obj)
        else:
            generator = torch.Generator(device=device_obj)
            generator.manual_seed(int(frame_obj.seed))
            noise = torch.rand((n, 3), device=device_obj, dtype=dtype_obj, generator=generator)
        return origin.unsqueeze(0) + noise * extent.unsqueeze(0)

    raise ValueError(f"Unknown population layout: {layout}")


def coerce_population_frame(frame: PopulationFrame | Mapping[str, Any]) -> PopulationFrame:
    if isinstance(frame, PopulationFrame):
        return frame

    if not isinstance(frame, Mapping):
        raise TypeError("frame must be PopulationFrame or Mapping[str, Any]")

    origin = _as_xyz_tuple(frame.get("origin"), name="origin")
    extent = _as_xyz_tuple(frame.get("extent"), name="extent")
    layout_raw = frame.get("layout")
    if layout_raw is None:
        raise ValueError("frame.layout is required")
    layout = str(layout_raw).strip().lower()
    if layout not in {"grid", "random", "ring", "line"}:
        raise ValueError("frame.layout must be one of: grid, random, ring, line")

    seed_raw = frame.get("seed")
    seed = None if seed_raw is None else int(seed_raw)
    return PopulationFrame(origin=origin, extent=extent, layout=cast(Any, layout), seed=seed)


def _resolve_dtype(torch: Any, dtype: Any) -> Any:
    if dtype is None:
        return torch.get_default_dtype()
    if isinstance(dtype, str):
        if not hasattr(torch, dtype):
            raise ValueError(f"Unknown dtype '{dtype}'")
        return getattr(torch, dtype)
    return dtype


def _as_xyz_tuple(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"frame.{name} must be a sequence of length 3")
    return (float(value[0]), float(value[1]), float(value[2]))


def _linspace_tensor(start: Any, end: Any, count: int, *, device: Any, dtype: Any) -> Any:
    torch = require_torch()
    count = int(count)
    if count <= 1:
        center = start + 0.5 * (end - start)
        return torch.full((1,), center, device=device, dtype=dtype)
    return torch.linspace(start, end, count, device=device, dtype=dtype)


def _grid_dims(n: int, extent_x: float, extent_y: float) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    aspect = abs(extent_x / extent_y) if extent_y not in (0.0, -0.0) else 1.0
    cols = max(1, int(math.ceil(math.sqrt(n * aspect))))
    rows = max(1, int(math.ceil(n / cols)))
    return cols, rows


__all__ = ["coerce_population_frame", "generate_positions", "positions_tensor"]

