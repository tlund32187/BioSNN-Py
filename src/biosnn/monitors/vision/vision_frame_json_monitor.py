"""JSON monitor for visual frame exports."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal, cast

from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.monitors.metrics.scalar_utils import scalar_to_float


class VisionFrameJsonMonitor(IMonitor):
    """Write a downsampled vision frame to `vision.json`."""

    name = "vision_frame_json"

    def __init__(
        self,
        path: str | Path,
        *,
        tensor_key: str = "vision/frame",
        every_n_steps: int = 1,
        max_side: int = 64,
        output_dtype: Literal["uint8", "float16"] = "uint8",
        allow_cuda_sync: bool = False,
        write_initial: bool = True,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tensor_key = str(tensor_key)
        self._every_n_steps = max(1, int(every_n_steps))
        self._max_side = max(1, int(max_side))
        self._output_dtype = output_dtype
        self._allow_cuda_sync = bool(allow_cuda_sync)
        self._warned_cuda = False
        self._event_count = 0

        if write_initial:
            self._write_payload(
                {
                    "step": 0,
                    "t": 0.0,
                    "tensor_key": self._tensor_key,
                    "available": False,
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
            "tensor_key": self._tensor_key,
            "available": False,
        }
        if not event.tensors:
            self._write_payload(payload)
            return

        frame = self._resolve_frame(event.tensors)
        if frame is None:
            self._write_payload(payload)
            return

        transformed = self._downsample_frame(frame)
        if transformed is None:
            self._write_payload(payload)
            return

        payload.update(
            {
                "available": True,
                "dtype": str(transformed.dtype).replace("torch.", ""),
                "shape": [int(dim) for dim in transformed.shape],
                "data": transformed.tolist(),
            }
        )
        self._write_payload(payload)

    def requirements(self) -> MonitorRequirements:
        return MonitorRequirements(
            needs_population_state=True,
            needs_scalars=True,
        )

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _resolve_frame(self, tensors: dict[str, Tensor] | Any) -> Tensor | None:
        value = tensors.get(self._tensor_key)
        if value is not None:
            return value
        for key, tensor in tensors.items():
            if str(key).endswith("/frame"):
                return tensor
        return None

    def _downsample_frame(self, frame: Tensor) -> Tensor | None:
        torch = require_torch()
        if (
            hasattr(frame, "device")
            and getattr(frame.device, "type", None) == "cuda"
            and not self._allow_cuda_sync
        ):
            if not self._warned_cuda:
                warnings.warn(
                    "VisionFrameJsonMonitor disabled on CUDA unless allow_cuda_sync=True",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_cuda = True
            return None

        sample = frame.detach()
        if sample.ndim == 4:
            sample = sample[0]
        if sample.ndim == 2:
            sample = sample.unsqueeze(0)
        elif sample.ndim == 3:
            if sample.shape[-1] <= 4 and sample.shape[0] > 4:
                sample = sample.permute(2, 0, 1)
        else:
            return None

        sample = sample.unsqueeze(0)
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
        sample = sample.squeeze(0)
        if self._output_dtype == "float16":
            output = sample.to(dtype=torch.float16)
        else:
            output = _to_uint8(sample)
        if hasattr(output, "to") and getattr(output.device, "type", None) != "cpu":
            output = output.to("cpu")
        return output

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


def _to_uint8(value: Tensor) -> Tensor:
    torch = require_torch()
    sample = value.to(dtype=torch.float32)
    min_val = sample.min()
    max_val = sample.max()
    if float((max_val - min_val).abs().item()) < 1e-12:
        normalized = torch.zeros_like(sample)
    elif float(min_val.item()) < 0.0 or float(max_val.item()) > 1.0:
        normalized = (sample - min_val) / (max_val - min_val)
    else:
        normalized = torch.clamp(sample, 0.0, 1.0)
    out = torch.clamp(normalized * 255.0, min=0.0, max=255.0).round().to(dtype=torch.uint8)
    return cast(Tensor, out)


__all__ = ["VisionFrameJsonMonitor"]
