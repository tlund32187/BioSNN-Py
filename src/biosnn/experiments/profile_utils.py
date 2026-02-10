"""Shared profiling helpers for demo runners."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from biosnn.core.torch_utils import require_torch


def maybe_write_profile_trace(
    *,
    enabled: bool,
    engine: Any,
    steps: int,
    device: str,
    out_path: Path,
    step_fn: Callable[[], None] | None = None,
) -> bool:
    """Write a chrome trace profile when explicitly enabled."""

    if not enabled:
        return False

    torch = require_torch()
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        print("Profiler unavailable; skipping profile run.")
        return False

    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    run_step = step_fn if step_fn is not None else getattr(engine, "step", None)
    if not callable(run_step):
        raise ValueError("Profiling requires an engine with a callable step()")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with profile(activities=activities) as prof:
        for _ in range(max(1, int(steps))):
            run_step()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_path))
    return True


__all__ = ["maybe_write_profile_trace"]
