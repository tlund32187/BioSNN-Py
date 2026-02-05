from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit


from biosnn.core.torch_utils import require_torch


def test_require_torch_imports() -> None:
    torch = pytest.importorskip("torch")
    assert require_torch() is torch
