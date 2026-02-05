import pytest

pytestmark = pytest.mark.unit

def test_import_and_version():
    import biosnn
    import biosnn.api as api

    assert hasattr(biosnn, "__version__")
    assert isinstance(biosnn.__version__, str)
    assert api.__version__ == biosnn.__version__
