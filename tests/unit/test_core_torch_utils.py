from __future__ import annotations

import pytest

from biosnn.core.torch_utils import require_torch

pytestmark = pytest.mark.unit

def test_require_torch_imports() -> None:
    torch = pytest.importorskip("torch")
    assert require_torch() is torch
