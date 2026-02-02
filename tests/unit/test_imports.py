def test_import_and_version():
    import biosnn

    assert hasattr(biosnn, "__version__")
    assert isinstance(biosnn.__version__, str)
