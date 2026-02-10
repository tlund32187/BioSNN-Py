"""JSON monitor for neuromodulator diffusion grids."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.monitors.metrics.scalar_utils import scalar_to_float


class ModulatorGridJsonMonitor(IMonitor):
    """Write downsampled modulator grids to `modgrid.json`."""

    name = "modulator_grid_json"

    def __init__(
        self,
        path: str | Path,
        *,
        every_n_steps: int = 1,
        max_side: int = 64,
        allow_cuda_sync: bool = False,
        modulator_names: list[str] | None = None,
        write_initial: bool = True,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._every_n_steps = max(1, int(every_n_steps))
        self._max_side = max(1, int(max_side))
        self._allow_cuda_sync = bool(allow_cuda_sync)
        self._modulator_names = {name.strip() for name in modulator_names or [] if name.strip()}
        self._warned_cuda = False
        self._event_count = 0

        if write_initial:
            self._write_payload(
                {
                    "step": 0,
                    "t": 0.0,
                    "available": False,
                    "grids": {},
                }
            )

    def on_step(self, event: StepEvent) -> None:
        self._event_count += 1
        if self._event_count % self._every_n_steps != 0:
            return

        step = int(scalar_to_float(event.scalars.get("step", 0))) if event.scalars else 0
        payload: dict[str, Any] = {
            "step": step,
            "t": float(event.t),
            "available": False,
            "grids": {},
        }
        if not event.tensors:
            self._write_payload(payload)
            return

        for key, value in event.tensors.items():
            mod_name = _parse_mod_grid_key(key)
            if mod_name is None:
                continue
            if self._modulator_names and mod_name not in self._modulator_names:
                continue
            downsampled = self._downsample_grid(value)
            if downsampled is None:
                continue
            payload["grids"][mod_name] = downsampled

        payload["available"] = bool(payload["grids"])
        self._write_payload(payload)

    def requirements(self) -> MonitorRequirements:
        return MonitorRequirements(
            needs_modulators=True,
            needs_scalars=True,
        )

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _downsample_grid(self, grid: Tensor) -> list[Any] | None:
        torch = require_torch()
        if (
            hasattr(grid, "device")
            and getattr(grid.device, "type", None) == "cuda"
            and not self._allow_cuda_sync
        ):
            if not self._warned_cuda:
                warnings.warn(
                    "ModulatorGridJsonMonitor disabled on CUDA unless allow_cuda_sync=True",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_cuda = True
            return None
        if grid.ndim not in (2, 3):
            return None

        sample = grid.unsqueeze(0).unsqueeze(0) if grid.ndim == 2 else grid.unsqueeze(0)

        sample = sample.detach()
        if not getattr(sample.dtype, "is_floating_point", False):
            sample = sample.to(dtype=torch.float32)

        h, w = int(sample.shape[-2]), int(sample.shape[-1])
        target_h, target_w = _target_size(h, w, max_side=self._max_side)
        if target_h != h or target_w != w:
            sample = torch.nn.functional.interpolate(
                sample,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        output = sample.squeeze(0)
        if hasattr(output, "to") and getattr(output.device, "type", None) != "cpu":
            output = output.to("cpu")
        return output.tolist()

    def _write_payload(self, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True)
        tmp_path = self._path.with_name(f"{self._path.name}.tmp")
        tmp_path.write_text(f"{text}\n", encoding="utf-8")
        tmp_path.replace(self._path)


def _target_size(h: int, w: int, *, max_side: int) -> tuple[int, int]:
    if h <= max_side and w <= max_side:
        return h, w
    if h <= 0 or w <= 0:
        return max(1, h), max(1, w)
    scale = min(float(max_side) / float(h), float(max_side) / float(w))
    target_h = max(1, int(round(h * scale)))
    target_w = max(1, int(round(w * scale)))
    return target_h, target_w


def _parse_mod_grid_key(key: str) -> str | None:
    parts = key.split("/")
    if len(parts) == 3 and parts[0] == "mod" and parts[2] == "grid":
        name = parts[1].strip()
        return name or None
    return None


__all__ = ["ModulatorGridJsonMonitor"]
