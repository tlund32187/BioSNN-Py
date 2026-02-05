from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress

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
        with suppress(RuntimeError):
            torch.set_num_threads(1)
        if prev_num_interop_threads != 1:
            with suppress(RuntimeError):
                torch.set_num_interop_threads(1)
        yield
    finally:
        torch.set_rng_state(prev_rng_state)
        torch.use_deterministic_algorithms(prev_deterministic)
        with suppress(RuntimeError):
            torch.set_num_threads(prev_num_threads)
        # torch.set_num_interop_threads can only be called once per process.
        # If we set it above, we intentionally leave it fixed at 1.
