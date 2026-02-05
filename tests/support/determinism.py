from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

from biosnn.core.torch_utils import require_torch


@contextmanager
def set_deterministic_cpu(seed: int) -> Iterator[None]:
    """Force deterministic CPU settings for golden tests."""
    torch = require_torch()
    prev_rng_state = torch.get_rng_state()
    prev_num_threads = torch.get_num_threads()
    prev_num_interop_threads = torch.get_num_interop_threads()
    deterministic_enabled_fn = getattr(
        torch,
        "are_deterministic_algorithms_enabled",
        getattr(torch, "is_deterministic_algorithms_enabled", None),
    )
    prev_deterministic = (
        bool(deterministic_enabled_fn()) if deterministic_enabled_fn is not None else False
    )
    try:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        try:
            torch.set_num_threads(1)
        except RuntimeError as exc:  # pragma: no cover - torch guardrails
            warnings.warn(
                f"Could not set torch num threads: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:  # pragma: no cover - torch guardrails
            warnings.warn(
                f"Could not set torch interop threads: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        yield
    finally:
        torch.set_rng_state(prev_rng_state)
        torch.use_deterministic_algorithms(prev_deterministic)
        try:
            torch.set_num_threads(prev_num_threads)
        except RuntimeError as exc:  # pragma: no cover - torch guardrails
            warnings.warn(
                f"Could not restore torch num threads: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            torch.set_num_interop_threads(prev_num_interop_threads)
        except RuntimeError as exc:  # pragma: no cover - torch guardrails
            warnings.warn(
                f"Could not restore torch interop threads: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
