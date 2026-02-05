import sys

import pytest
pytestmark = pytest.mark.unit



def test_api_import_does_not_eagerly_import_torch() -> None:
    if "torch" in sys.modules:
        pytest.skip("torch already imported in test session")
    __import__("biosnn.api")
    assert "torch" not in sys.modules
